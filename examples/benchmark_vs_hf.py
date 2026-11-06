"""Compare PagedTransformerEngine with HF transformers on a real checkpoint.

Usage: python examples/benchmark_vs_hf.py <model_dir> {hf|hm_default|hm_lowram} [new_tokens]
Run each mode in a separate process so peak RSS is measured independently.
"""
import sys, time, json, resource, torch
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))
torch.manual_seed(0); torch.set_num_threads(2)
M=sys.argv[1]; mode=sys.argv[2]; N=int(sys.argv[3]) if len(sys.argv)>3 else 16
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(M)
prompt="The capital of Egypt is"
ids=tok(prompt,return_tensors="pt").input_ids
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
base=rss()
t0=time.time()
if mode=="hf":
    from transformers import LlamaForCausalLM
    m=LlamaForCausalLM.from_pretrained(M,dtype=torch.bfloat16).eval(); tl=time.time()-t0
    with torch.no_grad():
        logits=m(ids).logits[0,-1].float()
        t1=time.time(); out=m.generate(ids,max_new_tokens=N,do_sample=False)[0,ids.size(1):].tolist(); tg=time.time()-t1
else:
    from hivemind.core.file_pager import FileBackedPager
    from hivemind.core.transformer_engine import PagedTransformerEngine, TransformerConfig
    if mode=="hm_default": pager=FileBackedPager(M+"/model.safetensors"); cache=100
    else: pager=FileBackedPager(M+"/model.safetensors",page_size_mb=100,max_cached_pages=2); cache=0
    e=PagedTransformerEngine(pager,tok,TransformerConfig.from_hf_config(M+"/config.json"),max_cached_tensors=cache); tl=time.time()-t0
    with torch.no_grad():
        logits=e.forward(ids)[0,-1].float()
        t1=time.time(); e.generate(prompt,max_new_tokens=N,temperature=0); tg=time.time()-t1
        # re-run deterministically to get token ids
        e.reset_cache(); x=ids.clone()
        for i in range(N):
            l=e.forward(x if i==0 else x[:,-1:],use_cache=True); nt=l[0,-1].argmax().view(1,1); x=torch.cat([x,nt],1)
        out=x[0,ids.size(1):].tolist()
torch.save(logits,f"{mode}_logits.pt")
print(json.dumps(dict(mode=mode,load_s=round(tl,2),gen_s=round(tg,2),tok_per_s=round(N/tg,3),peak_rss_mb=round(rss()),baseline_rss_mb=round(base),tokens=out,text=tok.decode(out))))
