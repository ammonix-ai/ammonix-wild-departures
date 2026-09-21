import runtime
from runtime import ROOT, WORK, CLASSES
import json
from PIL import Image, ImageDraw, ImageFont, ImageOps
out=WORK/'audit'; out.mkdir(exist_ok=True)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12)
for species in CLASSES+['dolphin','snail','moose','kangaroo']:
    path=ROOT/'data'/(species+'.json')
    if not path.exists(): continue
    rows=json.loads(path.read_text('utf-8'))
    sheet=Image.new('RGB',(1260,((len(rows)+6)//7)*155),(240,242,245)); d=ImageDraw.Draw(sheet)
    for i,row in enumerate(rows):
        x=(i%7)*180; y=(i//7)*155
        im=Image.open(ROOT/row['image']); im=ImageOps.contain(im,(174,126))
        sheet.paste(im,(x+(174-im.width)//2,y))
        d.text((x+3,y+129),f'{i}: {row["id"]}',font=font,fill=(20,30,40))
    sheet.save(out/(species+'.jpg'))
print('audit sheets',len(list(out.glob('*.jpg'))))
