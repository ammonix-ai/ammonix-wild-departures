"""Download traceable Commons photographs; split only after global deduplication."""
import runtime
from runtime import ROOT, WORK, CLASSES
import requests, json, time, re, hashlib, random, io, html
from PIL import Image, ImageOps
import imagehash
from concurrent.futures import ThreadPoolExecutor

DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)
UA = {'User-Agent': 'AmmonixAnimalDemo/1.0 (local visual classification research)'}
QUERIES = {
 'dog': 'dog animal', 'cat':'domestic cat', 'cow':'cow cattle', 'pig':'domestic pig',
 'bear':'bear animal', 'monkey':'monkey primate', 'eagle':'eagle bird', 'owl':'owl bird',
 'chicken':'chicken bird', 'duck':'duck bird', 'parrot':'parrot bird', 'ant':'ant insect',
 'beetle':'beetle -ladybug -ladybird -Coccinellidae', 'ladybug':'ladybird Coccinellidae',
 'lizard':'lizard -crocodile -alligator -snake', 'bee':'bee insect', 'turtle':'turtle animal',
}
def clean(s): return html.unescape(re.sub('<[^>]+>', '', s or '')).strip()
def get(url, **kw):
 for attempt in range(8 if '/w/api.php' in url else 2):
  r = requests.get(url, headers=UA, timeout=45, **kw)
  if r.status_code in (429, 503): time.sleep(float(r.headers.get('Retry-After',60))); continue
  r.raise_for_status(); return r
 raise RuntimeError('rate limited')

def collect(species):
 cache = DATA / (species+'.json')
 rows=json.loads(cache.read_text('utf-8')) if cache.exists() else []
 hashes=[imagehash.hex_to_hash(r['phash']) for r in rows]
 desired=({'elephant':52,'dog':46,'goat':46,'sheep':46}.get(species,40)) if species in CLASSES else (1 if species=='kangaroo' else 10)
 if len(rows)>=desired: return rows
 for offset in range(0, 400, 40):
  query=QUERIES.get(species, species+' animal')+' filetype:bitmap -skeleton -drawing -illustration -logo -statue -toy -painting'
  data=get('https://commons.wikimedia.org/w/api.php',params=dict(action='query',generator='search',gsrsearch=query,gsrnamespace=6,gsrlimit=40,gsroffset=offset,prop='imageinfo',iiprop='url|extmetadata|size',iiurlwidth=640,format='json')).json()
  pages=sorted(data.get('query',{}).get('pages',{}).values(),key=lambda p:p.get('index',0))
  for page in pages:
   try:
    if str(page['pageid']) in {r['id'] for r in rows}: continue
    inf=page['imageinfo'][0]; meta=inf.get('extmetadata',{})
    val=lambda k:clean(meta.get(k,{}).get('value',''))
    license=val('LicenseShortName')
    author=val('Artist')
    if author and sum(r['author']==author for r in rows)>=2: continue
    if not any(k in license.lower() for k in ['cc by','cc0','public domain']): continue
    if min(inf.get('width',0),inf.get('height',0))<300: continue
    if inf.get('size',0)>15000000: continue
    url=inf['url']
    raw=get(url).content
    im=ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert('RGB')
    ph=imagehash.phash(im)
    if any(ph-h<=8 for h in hashes): continue
    name=f'{species}-{page["pageid"]}.jpg'; im.thumbnail((960,960)); im.save(DATA/name,quality=91)
    hashes.append(ph)
    rows.append(dict(id=str(page['pageid']),species=species,image='data/'+name,source=inf['descriptionurl'],original=inf['url'],license=license,license_url=val('LicenseUrl'),author=author,title=page['title'],description=val('ImageDescription'),phash=str(ph),sha256=hashlib.sha256(raw).hexdigest(),label_review='source-search; requires visual audit'))
    cache.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    if len(rows)>=desired: break
    time.sleep(.1)
   except Exception as e: print('skip',species,str(e)[:100],flush=True)
  if len(rows)>=desired: break
 cache.write_text(json.dumps(rows,indent=2),encoding='utf-8')
 print(species,len(rows),flush=True)
 return rows

if __name__=='__main__':
 species=CLASSES+['dolphin','snail','moose','kangaroo']
 with ThreadPoolExecutor(max_workers=2) as pool: groups=list(pool.map(collect,species))
 allrows=[]; seen=[]
 for name,rows in zip(species,groups):
  unique=[]
  for row in rows:
   h=imagehash.hex_to_hash(row['phash'])
   if any(h-x<=8 for x in seen): continue
   seen.append(h); unique.append(row)
  random.Random(27035).shuffle(unique)
  for i,row in enumerate(unique):
   row['split']=('test' if i<2 else 'validation' if i<7 else 'train') if name in CLASSES else ('ood_test' if name=='kangaroo' else 'ood_validation')
  allrows+=unique
 (ROOT/'manifest.json').write_text(json.dumps(allrows,indent=2),encoding='utf-8')
 print('TOTAL',len(allrows),'train',sum(r['split']=='train' for r in allrows),flush=True)
