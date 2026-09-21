import runtime
from runtime import ROOT, CHECKPOINT, CHECKPOINT_REVISION, CLASSES, GATES, PROMPT, TRADITIONAL_PROMPT, TRADITIONAL_MAX_NEW_TOKENS
import json, random, time, hashlib, re
import numpy as np
import xgboost as xgb
from train import probs, metrics

def normalize(answer):
    a=answer.lower().strip().replace('_',' ')
    synonyms={'ladybird':'ladybug','cattle':'cow','bull':'cow','calf':'cow','hen':'chicken','rooster':'chicken','cockerel':'chicken','crocodylus':'crocodile','canis familiaris':'dog','canis lupus familiaris':'dog','felis catus':'cat','panthera leo':'lion','panthera tigris':'tiger','ursus':'bear','equus caballus':'horse','ovis aries':'sheep','capra hircus':'goat','sus scrofa':'pig','bald eagle':'eagle','honeybee':'bee','budgerigar':'parrot','cockatoo':'parrot','macaw':'parrot'}
    for word,label in synonyms.items():
        if re.search(r'\b'+re.escape(word)+r'\b',a): return label
    additional={
      'dog':['retriever','terrier','spaniel','bulldog','beagle','chihuahua','poodle','collie','shepherd','husky','labrador','doberman','akita','shiba','malamute','pug','rottweiler','dachshund'],
      'cat':['maine coon','siamese','domestic shorthair','british shorthair'],
      'beetle':['weevil','scarab','stag beetle','rhinoceros beetle','firefly'],
      'bee':['bumblebee','bumble bee','honeybee'],
      'parrot':['parakeet','kea','kaka','kakapo','lorikeet','cockatiel','galah'],
      'duck':['mallard','pintail','wigeon','teal'],
      'lizard':['gecko','iguana','skink','agama','anole','chameleon','lacerta'],
      'butterfly':['monarch','painted lady','red admiral','swallowtail','fritillary'],
      'zebra':['quagga'],
    }
    for label,words in additional.items():
        if any(re.search(r'\b'+re.escape(w)+r'\b',a) for w in words):return label
    # More specific beetle subgroup takes precedence.
    for label in ['ladybug']+CLASSES:
        if re.search(r'\b'+label+r's?\b',a): return label
    return 'unmapped'

class Classifier:
    def __init__(self):
        self.model=xgb.Booster();self.model.load_model(ROOT/'ammonix-xgboost.ubj');self.model.set_param({'nthread':1,'device':'cpu'})
        self.config=json.loads((ROOT/'training.json').read_text()); self.novel=dict(np.load(ROOT/'novelty.npz'))
    def predict(self,x):
        t=time.perf_counter()
        p=probs(self.model.inplace_predict(x[None,:],predict_type='margin'),self.config['temperature'])[0]
        z=(x-self.novel['mean'])/self.novel['scale']; z/=np.linalg.norm(z)
        score=float(p.max()*np.clip((self.novel['centers']@z).max(),0,1))
        label=CLASSES[int(p.argmax())]; gate=GATES[label]; rejected=score<self.config['rejection_threshold']
        elapsed=(time.perf_counter()-t)*1000
        return dict(prediction=label,gate=gate,confidence=float(p.max()),novelty_score=score,escalated=bool(rejected),classifier_ms=elapsed),p

def fallback(engine,path):
    prompt='Identify the animal in this image and its broad animal group. Return only JSON with keys "animal" and "group". The group must be mammal, bird, reptile, arthropod, or other.'
    raw,latency=engine.generate(path,prompt,max_new_tokens=64)
    try:
        parsed=json.loads(raw[raw.index('{'):raw.rindex('}')+1]);label=str(parsed['animal']);group=str(parsed['group']).lower()
        gate={'mammal':'MAMMALS','bird':'BIRDS & REPTILES','reptile':'BIRDS & REPTILES','arthropod':'ARTHROPODS'}.get(group)
    except (ValueError,KeyError,TypeError): label='Review required';gate=None
    return dict(final_label=label,final_gate=gate,fallback_ms=latency,fallback_raw=raw)

def traditional(engine,path):
    answer,latency=engine.generate(path,TRADITIONAL_PROMPT,max_new_tokens=TRADITIONAL_MAX_NEW_TOKENS)
    completion=dict(engine.last_generation)
    if not completion['complete']:
        raise RuntimeError(f'Traditional LLM answer reached the token limit for {path}; refusing to publish a truncated answer.')
    return answer,latency,completion

def run():
    from engine import Engine
    rows=json.loads((ROOT/'manifest.json').read_text('utf-8'))
    test=[r for r in rows if r['split']=='test'];random.Random(70135).shuffle(test)
    assert len(test)==70 and all(sum(r['species']==c for r in test)==2 for c in CLASSES)
    test += [r for r in rows if r['split']=='ood_test']
    engine=Engine();classifier=Classifier()
    for _ in range(3):
        x,_=engine.feature(ROOT/test[0]['image']);classifier.predict(x);traditional(engine,ROOT/test[0]['image'])
    records=[]; probabilities=[]; raw_trials=[]
    for i,row in enumerate(test):
        path=ROOT/row['image'];trials=[];results=[]
        for repeat in range(3):
            # Sequential, alternating order: no GPU contention and no reused feature/KV cache.
            if (i+repeat)%2==0:
                x,t=engine.feature(path); result,p=classifier.predict(x);answer,gms,completion=traditional(engine,path)
            else:
                answer,gms,completion=traditional(engine,path);x,t=engine.feature(path);result,p=classifier.predict(x)
            result.update(t);result['ammonix_ms']=t['feature_ms']+result['classifier_ms']
            result['traditional_ms']=gms;result['traditional_answer']=answer
            result['traditional_completion']=completion
            # Compatibility aliases retain the unmodified answer, never a dictionary match.
            result['qwen_ms']=gms;result['qwen_answer']=answer
            trials.append(result);results.append(p)
        # All visual timings are medians of the same three independent observations.
        rec=dict(trials[0]);
        for key in ['preprocess_ms','traditional_ms','qwen_ms','feature_ms','classifier_ms','ammonix_ms']:rec[key]=float(np.median([r[key] for r in trials]))
        rec.update(id=row['id'],species=row['species'],image=row['image'],source=row['source'],ood=row['split']=='ood_test',correct=rec['prediction']==row['species'])
        rec['final_label']=rec['prediction'];rec['final_gate']=rec['gate'];rec['fallback_ms']=0
        if rec['escalated']:rec.update(fallback(engine,path))
        if not rec['ood']:
            rec['final_correct']=normalize(rec['final_label'])==row['species']
        records.append(rec);probabilities.append(results[0]);raw_trials.append(dict(id=row['id'],trials=trials))
        print(i+1,row['species'],'->',rec['prediction'],round(rec['ammonix_ms']),round(rec['traditional_ms']),'escalated',rec['escalated'],flush=True)
        (ROOT/'benchmark-progress.json').write_text(json.dumps(records,indent=2))
    report=metrics(np.array([CLASSES.index(r['species']) for r in test[:70]]),np.stack(probabilities[:70]))
    report.update(median_ammonix_ms=float(np.median([r['ammonix_ms'] for r in records[:70]])),median_traditional_ms=float(np.median([r['traditional_ms'] for r in records[:70]])),median_qwen_ms=float(np.median([r['traditional_ms'] for r in records[:70]])),median_classifier_ms=float(np.median([r['classifier_ms'] for r in records[:70]])),p95_ammonix_ms=float(np.percentile([r['ammonix_ms'] for r in records[:70]],95)),known_escalations=sum(r['escalated'] for r in records[:70]),kangaroo_rejected=records[-1]['escalated'],kangaroo_answer=records[-1]['final_label'],test_images=70)
    result=dict(measured=True,model='Qwen3.8-27B',checkpoint=CHECKPOINT,checkpoint_revision=CHECKPOINT_REVISION,execution='NF4 checkpoint reconstructed to FP16; torch SDPA; eager linear attention; batch 1; max image pixels 147456; no thinking',gpu='NVIDIA RTX PRO 6000 Blackwell 96 GB',feature='final layer normalized hidden state at last prompt position; predicts first answer token; 5120 values',feature_prompt=PROMPT,traditional_prompt=TRADITIONAL_PROMPT,traditional_max_new_tokens=TRADITIONAL_MAX_NEW_TOKENS,traditional_answer_processing='Full decoded answer; no normalization or parsing.',repetitions=3,manifest_sha256=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest(),classifier_sha256=hashlib.sha256((ROOT/'ammonix-xgboost.ubj').read_bytes()).hexdigest(),novelty_sha256=hashlib.sha256((ROOT/'novelty.npz').read_bytes()).hexdigest(),report=report,rows=records)
    (ROOT/'results.json').write_text(json.dumps(result,indent=2))
    (ROOT/'raw-trials.json').write_text(json.dumps(raw_trials,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':run()
