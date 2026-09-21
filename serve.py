"""Local interactive replay, with optional fresh GPU inference via --live."""
import runtime
from runtime import ROOT
import argparse,json,threading
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from benchmark import Classifier, fallback, traditional

parser=argparse.ArgumentParser();parser.add_argument('--live',action='store_true');parser.add_argument('--port',type=int,default=8766)
args=parser.parse_args();lock=threading.Lock();engine=None;classifier=None
if args.live:
    from engine import Engine
    engine=Engine();classifier=Classifier()
    from PIL import Image
    for _ in range(3):
        warmup_image=Image.new('RGB',(384,384))
        x,_=engine.feature(warmup_image);classifier.predict(x);traditional(engine,warmup_image)

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*a,**kw):super().__init__(*a,directory=str(ROOT),**kw)
    def do_GET(self):
        if self.path=='/api/status':
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(json.dumps({'live':engine is not None}).encode());return
        super().do_GET()
    def do_POST(self):
        if self.path!='/api/classify' or engine is None:self.send_error(404);return
        try:
            length=int(self.headers.get('Content-Length',0))
            if length>1024:raise ValueError('Request too large')
            requested=json.loads(self.rfile.read(length))['id']
            manifest=json.loads((ROOT/'manifest.json').read_text('utf-8'))
            row=next(r for r in manifest if r['id']==requested and r['split'] in ['test','ood_test'])
            with lock:
                x,t=engine.feature(ROOT/row['image']);rec,p=classifier.predict(x);answer,gms,completion=traditional(engine,ROOT/row['image'])
                rec.update(t);rec.update(ammonix_ms=t['feature_ms']+rec['classifier_ms'],traditional_ms=gms,traditional_answer=answer,traditional_completion=completion,qwen_ms=gms,qwen_answer=answer,final_label=rec['prediction'],final_gate=rec['gate'],fallback_ms=0)
                if rec['escalated']:rec.update(fallback(engine,ROOT/row['image']))
            rec.update(id=row['id'],species=row['species'],image=row['image'],ood=row['split']=='ood_test',correct=rec['prediction']==row['species'])
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(json.dumps(rec).encode())
        except Exception as e:self.send_error(400,str(e))
print(f'Airport ready: http://127.0.0.1:{args.port} live={args.live}',flush=True)
ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
