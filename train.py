"""Fit and calibrate the 35-class decision layer from the shipped states. No photos, no language model.

Writes into retrained/ by default so the released classifier, which the recorded
results are bound to, stays untouched.
"""
import runtime
from runtime import ROOT, CLASSES, load_features as load
import argparse, json, time
import numpy as np
import xgboost as xgb

def probs(logits,temp):
    a=logits/temp; a=a-a.max(axis=1,keepdims=True); p=np.exp(a); return p/p.sum(axis=1,keepdims=True)
def metrics(y,p):
    pred=p.argmax(1); f1=[]; auc=[]
    for c in range(35):
        tp=np.sum((y==c)&(pred==c)); fp=np.sum((y!=c)&(pred==c)); fn=np.sum((y==c)&(pred!=c))
        f1.append(float(2*tp/max(1,2*tp+fp+fn)))
        pos=p[y==c,c]; neg=p[y!=c,c]; auc.append(float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg))))
    n=len(y); acc=float(np.mean(y==pred)); z=1.96; center=(acc+z*z/(2*n))/(1+z*z/n); half=z*np.sqrt(acc*(1-acc)/n+z*z/(4*n*n))/(1+z*z/n)
    return dict(top1_accuracy=acc,macro_f1=float(np.mean(f1)),macro_ovr_auroc=float(np.mean(auc)),accuracy_wilson95=[float(center-half),float(center+half)])

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda',help='The released classifier was fitted with cuda; cpu needs no GPU and may differ slightly.')
    parser.add_argument('--output',default=str(ROOT/'retrained'))
    args=parser.parse_args()
    from pathlib import Path
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    rows=json.loads((ROOT/'manifest.json').read_text('utf-8'))
    train=[r for r in rows if r['split']=='train']; val=[r for r in rows if r['split']=='validation']; ood=[r for r in rows if r['split']=='ood_validation']
    X=load(train); V=load(val); O=load(ood)
    y=np.array([CLASSES.index(r['species']) for r in train]); vy=np.array([CLASSES.index(r['species']) for r in val])
    params=dict(objective='multi:softprob',num_class=35,max_depth=3,eta=.06,subsample=.9,colsample_bytree=.7,min_child_weight=1,reg_lambda=5,tree_method='hist',device=args.device,nthread=8,seed=27035,eval_metric='mlogloss')
    start=time.perf_counter()
    model=xgb.train(params,xgb.DMatrix(X,label=y),num_boost_round=350,evals=[(xgb.DMatrix(V,label=vy),'validation')],early_stopping_rounds=30,verbose_eval=50)
    model=model[:model.best_iteration+1]
    model.save_model(out/'ammonix-xgboost.ubj')
    logits=model.inplace_predict(V,predict_type='margin')
    temperatures=np.linspace(.5,3,101)
    temp=float(min(temperatures,key=lambda t:-np.log(probs(logits,t)[np.arange(len(vy)),vy]+1e-12).mean()))
    P=probs(logits,temp); OP=probs(model.inplace_predict(O,predict_type='margin'),temp)
    # Prototype distance supplements class confidence for unknown detection.
    mean=X.mean(0); scale=X.std(0)+1e-3
    Z=(X-mean)/scale; Z/=np.linalg.norm(Z,axis=1,keepdims=True)
    centers=np.stack([Z[y==i].mean(0) for i in range(35)]); centers/=np.linalg.norm(centers,axis=1,keepdims=True)
    def novelty(A):
        A=(A-mean)/scale; A/=np.linalg.norm(A,axis=1,keepdims=True); return np.clip((A@centers.T).max(1),0,1)
    vs=P.max(1)*novelty(V); os=OP.max(1)*novelty(O)
    thresholds=np.unique(np.r_[0,vs,os,1])
    threshold=float(max(thresholds,key=lambda t:((vs>=t).mean()+(os<t).mean())/2))
    np.savez(out/'novelty.npz',mean=mean,scale=scale,centers=centers)
    report=dict(classes=CLASSES,temperature=temp,rejection_threshold=threshold,rejection_rule='max temperature-scaled class probability * max cosine similarity to training-class prototypes in standardized feature space',validation=metrics(vy,P),validation_known_coverage=float((vs>=threshold).mean()),validation_ood_recall=float((os<threshold).mean()),training_images=len(train),validation_images=len(val),ood_validation_images=len(ood),training_seconds=time.perf_counter()-start,parameters=params)
    (out/'training.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
