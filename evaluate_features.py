"""Re-derive the held-out result from the shipped states: no GPU, no photos, no language model.

Scores the 70 test states and the one out-of-distribution state with a decision
layer and compares every prediction with the recorded benchmark. By default the
released classifier is used; pass --classifier retrained to check your own fit.
"""
import runtime
from runtime import ROOT, CLASSES, load_features
import argparse, json
import numpy as np
import xgboost as xgb
from train import probs, metrics

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--classifier',default=str(ROOT),help='Folder with ammonix-xgboost.ubj, novelty.npz and training.json')
args=parser.parse_args()
from pathlib import Path
folder=Path(args.classifier)
model=xgb.Booster();model.load_model(folder/'ammonix-xgboost.ubj');model.set_param({'nthread':1,'device':'cpu'})
config=json.loads((folder/'training.json').read_text());novel=dict(np.load(folder/'novelty.npz'))

rows=json.loads((ROOT/'manifest.json').read_text('utf-8'))
test=[r for r in rows if r['split']=='test'];ood=[r for r in rows if r['split']=='ood_test']
X=load_features(test+ood)
P=probs(model.inplace_predict(X,predict_type='margin'),config['temperature'])
Z=(X-novel['mean'])/novel['scale'];Z/=np.linalg.norm(Z,axis=1,keepdims=True)
score=P.max(1)*np.clip((Z@novel['centers'].T).max(1),0,1)
rejected=score<config['rejection_threshold']
labels=[CLASSES[i] for i in P.argmax(1)]

y=np.array([CLASSES.index(r['species']) for r in test])
report=metrics(y,P[:70])
correct=int(sum(label==row['species'] for label,row in zip(labels,test)))
print(f"held-out test images : {correct}/70 correct ({report['top1_accuracy']:.1%}), Wilson 95% {report['accuracy_wilson95'][0]:.1%} to {report['accuracy_wilson95'][1]:.1%}")
print(f"macro-F1 {report['macro_f1']:.3f} | one-vs-rest macro AUROC {report['macro_ovr_auroc']:.3f}")
print(f"known images escalated to System Two: {int(rejected[:70].sum())}/70")
for row,flag,label in zip(ood,rejected[70:],labels[70:]):
    print(f"unknown image ({row['species']}): {'rejected and escalated' if flag else 'NOT rejected, classified as '+label}")

if folder.resolve()==ROOT.resolve():
    recorded={r['id']:r for r in json.loads((ROOT/'results.json').read_text('utf-8'))['rows']}
    same=sum(recorded[row['id']]['prediction']==label and bool(recorded[row['id']]['escalated'])==bool(flag)
             for row,label,flag in zip(test+ood,labels,rejected))
    print(f"agreement with the recorded benchmark (prediction and escalation): {same}/{len(test)+len(ood)}")
