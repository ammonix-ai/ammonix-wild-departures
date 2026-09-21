"""Freeze reviewed labels and photographer-separated splits before evaluation."""
import runtime
from runtime import ROOT, CLASSES
import json, random, collections
import imagehash
REJECT = {
 'dog': [146846013,40910895,194951068,49858627,186658434,122282597,175183512,163514868,2898522,95087937,110416777,'inat-87498061'],
 'cat': [100151966,100152205,92201445],
 'horse': [22789658,146129084,38362130,117517756,50970683,4874351],
 'cow': [115727983],
 'goat': [141831760,194097888,123652417,80131861,82317624],
 'sheep': [114770848,122780458,121299817,121299818,38464488],
 'pig': [36053657,7541647,85577962,60403947,840096,34908726,146454347],
 'lion': [2196131,32042755],
 'elephant': [127209663,39820824,16151948,154005751,125948563,126285117,126633429,42677634,90089327,180969306,5923385,44470098,127525562,49042006,104528029,143516669,'inat-607144400','inat-154160065',67817174],
 'tiger': [178967624,178967619],
 'bear': [89412529,98796708,40984665],
 'rabbit': [103280824,186714927,21710709,119139366,65860656,195838418],
 'zebra': [175278240,4018080,145246154,25855508,25855505,105363230,12774773,186070902],
 'giraffe': ['inat-510220275'],
 'monkey': ['inat-219658643','inat-339428082','inat-111186','inat-254297764'],
 'eagle': ['inat-375464911','inat-370866345','inat-172940570','inat-166065348','inat-31950023','inat-610251716','inat-26992578','inat-54552552'],
 'chicken': ['inat-518956260','inat-451780056','inat-452604562','inat-325923865','inat-179358341','inat-111522587','inat-381673932','inat-246215398'],
 'duck': ['inat-252376727','inat-495113315','inat-271487982'],
 'parrot': ['inat-323193769','inat-607449666','inat-27184451','inat-332105560','inat-714280843'],
 'penguin': ['inat-617204755','inat-72265375','inat-62087174'],
 'flamingo': ['inat-278480100','inat-234874692'],
 'ostrich': ['inat-55224884','inat-522224034','inat-101039094','inat-15786722'],
 'crocodile': ['inat-368766310','inat-226458066','inat-28419686','inat-502792906'],
 'alligator': ['inat-206939231','inat-133357272','inat-19680043','inat-249518484'],
 'snake': ['inat-188344513','inat-102416341','inat-120026763','inat-29541401','inat-25171938','inat-81998339','inat-448291888'],
 'turtle': ['inat-5518581','inat-158956388','inat-589922520','inat-615588922','inat-536943771','inat-160796108'],
 'lizard': ['inat-514537641'],
 'bee': ['inat-17767241','inat-103047151','inat-580690822','inat-9460918','inat-657066690','inat-52274835'],
 'butterfly': ['inat-60419411','inat-29069005','inat-229887014','inat-606313286','inat-22178273','inat-440740793','inat-116808410','inat-230764123','inat-148992204','inat-1181829','inat-92809657','inat-360149314','inat-11288937','inat-211305988'],
 'ant': ['inat-341560540','inat-332556594','inat-32691260','inat-203915242','inat-308310558','inat-113079786'],
 'beetle': ['inat-12071973','inat-14939076','inat-4123012','inat-469878980','inat-56190066','inat-302090929'],
 'grasshopper': ['inat-32532025'],
 'dragonfly': ['inat-111133501','inat-383770391','inat-8707975','inat-64997090'],
 'ladybug': ['inat-82618589','inat-73842105','inat-6398640','inat-446662957'],
}
REJECT['dog'] += ['inat-603256966']
REJECT['elephant'] += ['inat-1119']
REJECT['lion'] += [146526140,76171361]
REJECT['tiger'] += [150306500]
REJECT['bear'] += ['inat-30787454','inat-30787317','inat-2129360']
REJECT['rabbit'] += ['inat-202919305','inat-1182695','inat-158875874','inat-179314619']
REJECT['zebra'] += ['inat-15574512']
REJECT['chicken'] += ['inat-170037296']
REJECT['butterfly'] += ['inat-9318766','inat-410243686']
REJECT['snail'] = ['inat-11343240']
REJECT['moose'] = ['inat-23946339','inat-64561076']
def freeze():
    out=[]; seen=[]; summary={}
    for species in CLASSES+['dolphin','snail','moose','kangaroo']:
        rows=json.loads((ROOT/'data'/(species+'.json')).read_text('utf-8'))
        rows=[r for r in rows if r['id'] not in {str(x) for x in REJECT.get(species,[])}]
        unique=[]
        for r in rows:
            h=imagehash.hex_to_hash(r['phash'])
            if any(h-x<=8 for x in seen): continue
            seen.append(h); unique.append(r)
        rng=random.Random(27035); rng.shuffle(unique)
        if species in CLASSES:
            groups=collections.defaultdict(list)
            for r in unique: groups[r['author'] or r['id']].append(r)
            ordered=sorted(groups.values(),key=len)
            assert len(ordered)>=8,species
            for i,group in enumerate(ordered):
                split='test' if i<2 else 'validation' if i<7 else 'train'
                for j,r in enumerate(group):
                    r['split']=split if split=='train' or j==0 else 'excluded_same_photographer'
        else:
            for r in unique: r['split']='ood_test' if species=='kangaroo' else 'ood_validation'
        for r in unique: r['label_review']='visual contact-sheet audit'
        counts=dict(collections.Counter(r['split'] for r in unique)); summary[species]=counts
        if species in CLASSES: assert counts.get('train',0)>=20,(species,counts)
        out+=unique
    (ROOT/'manifest.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    (ROOT/'split-counts.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': freeze()
