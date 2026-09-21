import runtime
from runtime import ROOT,WORK,CLASSES
from PIL import Image,ImageDraw,ImageFont,ImageOps
import json
reviewed={s:40 for s in CLASSES}
reviewed.update(dog=46,horse=41,elephant=51,tiger=27,bear=27,rabbit=27,zebra=27,pig=40,lion=27,ladybug=28,butterfly=38)
rows=[]
for s in CLASSES+['dolphin','snail','moose','kangaroo']:
    data=json.loads((ROOT/'data'/(s+'.json')).read_text('utf-8'))
    rows += [(s,i,r) for i,r in enumerate(data) if i>=reviewed.get(s,0)]
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12)
for start in range(0,len(rows),49):
    batch=rows[start:start+49];sheet=Image.new('RGB',(1260,((len(batch)+6)//7)*168),'#eff3ed');d=ImageDraw.Draw(sheet)
    for j,(s,i,r) in enumerate(batch):
        x=j%7*180;y=j//7*168;im=ImageOps.contain(Image.open(ROOT/r['image']),(174,128));sheet.paste(im,(x+(174-im.width)//2,y));d.text((x+3,y+129),f'{s} {i}\n{r["id"]}',font=font,fill='#102a2e')
    sheet.save(WORK/'audit'/f'additions-{start//49}.jpg')
print('additional rows',len(rows))
