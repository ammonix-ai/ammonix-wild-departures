"""Frozen local Qwen; image-conditioned first-answer-token hidden state."""
import runtime
runtime.configure_transformers()
from runtime import MODEL, PROMPT
import json, time
import numpy as np
import torch
from safetensors import safe_open
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoTokenizer, Qwen2VLImageProcessorPil, Qwen3VLProcessor, Qwen3_5ForConditionalGeneration
from PIL import Image

class ImageOnlyProcessor(Qwen3VLProcessor):
    @classmethod
    def get_attributes(cls): return ['image_processor', 'tokenizer']

def decode_nf4(f, name):
    """Decode the checkpoint's published NF4 lookup tables using ordinary Torch ops.

    No native bitsandbytes kernels are needed. The quantized checkpoint is retained;
    reconstructed weights run in FP16. Both comparison paths use these same weights.
    """
    state=json.loads(bytes(f.get_tensor(name+'.quant_state.bitsandbytes__nf4').tolist()))
    assert state['quant_type']=='nf4' and state['dtype']=='float16'
    codes=f.get_tensor(name).to('cuda').flatten()
    qmap=f.get_tensor(name+'.quant_map').to('cuda')
    absmax=f.get_tensor(name+'.absmax').to('cuda')
    nested_map=f.get_tensor(name+'.nested_quant_map').to('cuda')
    nested_abs=f.get_tensor(name+'.nested_absmax').to('cuda')
    indices=torch.arange(absmax.numel(),device='cuda') // state['nested_blocksize']
    scales=nested_map[absmax.long()]*nested_abs[indices]+state['nested_offset']
    n=int(np.prod(state['shape'])); result=torch.empty(n,device='cuda',dtype=torch.float16)
    chunk=4*1024*1024
    for lo in range(0,n,chunk):
        hi=min(lo+chunk,n); packed=codes[lo//2:(hi+1)//2]
        values=torch.stack((packed>>4,packed&15),dim=1).flatten()[:hi-lo].long()
        scale=scales[torch.arange(lo,hi,device='cuda')//state['blocksize']]
        result[lo:hi]=(qmap[values]*scale).half()
    return result.reshape(state['shape'])

class Engine:
    def __init__(self):
        torch.set_num_threads(8)
        cfg=json.loads((MODEL/'processor_config.json').read_text())['image_processor']
        cfg.pop('image_processor_type',None)
        cfg['size']={'shortest_edge':65536,'longest_edge':147456}
        self.processor=ImageOnlyProcessor(image_processor=Qwen2VLImageProcessorPil(**cfg),tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True),chat_template=(MODEL/'chat_template.jinja').read_text())
        config=AutoConfig.from_pretrained(str(MODEL),local_files_only=True)
        if hasattr(config,'quantization_config'): del config.quantization_config
        config._attn_implementation='sdpa'
        with init_empty_weights():
            self.model=Qwen3_5ForConditionalGeneration(config).half()
        with safe_open(str(MODEL/'model.safetensors'),framework='pt',device='cpu') as f:
            keys=set(f.keys())
            params=list(self.model.named_parameters())
            for i,(name,param) in enumerate(params):
                if name not in keys: raise RuntimeError('Missing checkpoint tensor: '+name)
                weight=decode_nf4(f,name) if name+'.quant_state.bitsandbytes__nf4' in keys else f.get_tensor(name).to(device='cuda',dtype=torch.float16)
                assert weight.shape==param.shape,(name,weight.shape,param.shape)
                parent,leaf=name.rsplit('.',1)
                setattr(self.model.get_submodule(parent),leaf,torch.nn.Parameter(weight,requires_grad=False))
                if i%100==0: print('weights',i,'/',len(params),flush=True)
        for module in self.model.modules():
            for name,buffer in module.named_buffers(recurse=False):
                if buffer is not None: setattr(module,name,buffer.to('cuda'))
        self.model.eval()
        torch.cuda.empty_cache()
        print('MODEL_READY',flush=True)

    def inputs(self,image,prompt=PROMPT):
        if not isinstance(image,Image.Image): image=Image.open(image).convert('RGB')
        messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
        return self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt').to('cuda')

    @torch.inference_mode()
    def feature(self,image):
        torch.cuda.synchronize(); start=time.perf_counter()
        inputs=self.inputs(image)
        torch.cuda.synchronize(); processed=time.perf_counter()
        out=self.model.model(**inputs,use_cache=False)
        feature=out.last_hidden_state[0,-1].float().cpu().numpy()
        torch.cuda.synchronize(); done=time.perf_counter()
        assert feature.shape==(5120,) and np.isfinite(feature).all()
        return feature,{'preprocess_ms':(processed-start)*1000,'qwen_ms':(done-processed)*1000,'feature_ms':(done-start)*1000}

    @torch.inference_mode()
    def generate(self,image,prompt=PROMPT,max_new_tokens=24):
        torch.cuda.synchronize(); start=time.perf_counter()
        inputs=self.inputs(image,prompt)
        result=self.model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,use_cache=True)
        generated=result[0,inputs['input_ids'].shape[1]:]
        answer=self.processor.tokenizer.decode(generated,skip_special_tokens=True).strip()
        torch.cuda.synchronize()
        elapsed=(time.perf_counter()-start)*1000
        eos=self.model.generation_config.eos_token_id
        if eos is None:eos=self.processor.tokenizer.eos_token_id
        eos_ids={eos} if isinstance(eos,int) else set(eos or [])
        ended=bool(generated.numel() and int(generated[-1].item()) in eos_ids)
        self.last_generation={'generated_tokens':int(generated.numel()),'finish_reason':'eos' if ended else 'length','complete':ended}
        return answer,elapsed

if __name__=='__main__':
    e=Engine()
    for color in [(20,130,80),(130,20,80)]:
        x,t=e.feature(Image.new('RGB',(384,384),color)); print(x.shape,t,flush=True)
    print(e.generate(Image.new('RGB',(384,384),(20,130,80))),flush=True)
