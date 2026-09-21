"""Fill candidate pools from licensed iNaturalist observations, at <=1 API request/s."""
import runtime
from runtime import ROOT, CLASSES
from curate import REJECT
import requests,json,time,io,hashlib,re
from PIL import Image,ImageOps
import imagehash
TAXA=dict(zip(CLASSES, ['Canis familiaris','Felis catus','Equus caballus','Bos taurus','Ovis aries','Capra hircus','Sus scrofa domesticus','Elephantidae','Panthera leo','Panthera tigris','Ursidae','Oryctolagus cuniculus','Giraffa','Equus quagga','Cercopithecidae','Aquila','Strigiformes','Gallus gallus','Anas','Psittaciformes','Spheniscidae','Phoenicopteridae','Struthio','Crocodylidae','Alligator','Serpentes','Testudines','Lacertidae','Anthophila','Papilionoidea','Formicidae','Coleoptera','Caelifera','Anisoptera','Coccinellidae']))
TAXA.update(dolphin='Delphinidae',snail='Helix',moose='Alces alces',kangaroo='Macropus')
session=requests.Session();session.headers['User-Agent']='AmmonixAnimalDemo/1.0 (small licensed-image research dataset)'
last=0
def api(path,params):
    global last
    time.sleep(max(0,1.1-(time.monotonic()-last)));last=time.monotonic()
    r=session.get('https://api.inaturalist.org/v1/'+path,params=params,timeout=40)
    if r.status_code==429: time.sleep(float(r.headers.get('Retry-After',60)));return api(path,params)
    r.raise_for_status();return r.json()

def run():
    for species,scientific in TAXA.items():
        p=ROOT/'data'/(species+'.json');rows=json.loads(p.read_text('utf-8')) if p.exists() else []
        wanted=(max(40,35+len(REJECT.get(species,[]))) if species in CLASSES else 1 if species=='kangaroo' else 10)
        if len(rows)>=wanted:continue
        result=api('taxa',dict(q=scientific,per_page=30))['results']
        exact=[t for t in result if t['name'].lower()==scientific.lower()]
        if not exact: print('NO EXACT TAXON',species,scientific,[(t['id'],t['name']) for t in result[:5]],flush=True);continue
        taxon=exact[0];hashes=[imagehash.hex_to_hash(r['phash']) for r in rows]
        for page in range(1,5):
            obs=api('observations',dict(taxon_id=taxon['id'],photos='true',photo_license='cc0,cc-by,cc-by-sa',per_page=100,order_by='votes',page=page))['results']
            for o in obs:
                # Exclude ladybirds from the mutually-exclusive beetle label.
                if species=='beetle' and 48486 in o.get('taxon',{}).get('ancestor_ids',[]):continue
                author=o['user'].get('name') or o['user']['login']
                if sum(r['author']==author for r in rows)>=2:continue
                photos=[ph for ph in o.get('photos',[]) if ph.get('license_code') in ['cc0','cc-by','cc-by-sa'] and not ph.get('hidden')]
                if not photos:continue
                ph=photos[0];id='inat-'+str(ph['id'])
                if any(r['id']==id for r in rows):continue
                url=re.sub(r'/square\.', '/medium.', ph['url'])
                try:
                    r=session.get(url,timeout=25);r.raise_for_status();raw=r.content
                    im=ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert('RGB')
                    if min(im.size)<180:continue
                    h=imagehash.phash(im)
                    if any(h-x<=8 for x in hashes):continue
                    name=species+'-'+id+'.jpg';im.save(ROOT/'data'/name,quality=93)
                    license=ph['license_code'];license_url='https://creativecommons.org/publicdomain/zero/1.0/' if license=='cc0' else 'https://creativecommons.org/licenses/'+license[3:]+'/4.0/'
                    rows.append(dict(id=id,species=species,image='data/'+name,source='https://www.inaturalist.org/observations/'+str(o['id']),original=url,license=license.upper(),license_url=license_url,author=author,title=o['taxon']['name'],description=o['taxon'].get('preferred_common_name',''),phash=str(h),sha256=hashlib.sha256(raw).hexdigest(),label_review='taxon-filtered source; requires visual audit',taxon_id=o['taxon']['id']))
                    hashes.append(h)
                    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(rows,indent=2),encoding='utf-8');tmp.replace(p)
                except Exception as e: print('skip',species,str(e)[:100],flush=True)
                if len(rows)>=wanted:break
            if len(rows)>=wanted:break
        print(species,len(rows),'target',wanted,flush=True)
    print('INAT_DONE',flush=True)
if __name__=='__main__':run()
