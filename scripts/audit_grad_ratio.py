"""Size of the contrastive gradient against cross-entropy, on a few real batches.

  python scripts/audit_grad_ratio.py <RGB PACS root> <6-tap M2-CL Sketch checkpoint>
"""
import os, sys, time
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
sys.path.insert(0,R+'/src')
import torch, torch.nn as nn
from m2cl.data import make_lodo_loaders
from m2cl.models import build_model
from m2cl.losses import m2cl_contrastive_loss
torch.manual_seed(0)
data,ckpt=sys.argv[1],sys.argv[2]
loaders,info=make_lodo_loaders(data,'sketch','baseline',224,128,0.2,0,0)
batches=[]
for i,(x,y,_) in enumerate(loaders['train']):
    batches.append((x,y))
    if i==2: break
print('batches',len(batches),'classes in batch',[len(y.unique()) for _,y in batches],flush=True)
def groups(m):
    bb,eb=[],[]
    for n,p in m.named_parameters():
        if n.startswith('layer') : bb.append(p)
        elif n.startswith('eb'): eb.append(p)
    return bb,eb
def gnorm(loss,ps):
    g=torch.autograd.grad(loss,ps,retain_graph=True,allow_unused=True)
    return float(sum((x**2).sum() for x in g if x is not None)**.5)
ckpts={'ImageNet start':None,'trained 6-tap M2-CL (sketch seed0)':ckpt}
ce=nn.CrossEntropyLoss()
for name,ck in ckpts.items():
    m=build_model('m2cl',7,6,True)
    if ck:
        sd=torch.load(ck,map_location='cpu',weights_only=True); sd=sd.get('model_state',sd)
        m.load_state_dict(sd)
    m.train()
    bb,eb=groups(m)
    print('\n==',name,flush=True)
    for x,y in batches[:2]:
        t=time.time(); logits,reps=m(x); lce=ce(logits,y)
        gce_b,gce_e=gnorm(lce,bb),gnorm(lce,eb)
        out=[]
        for tau in (1.0,0.5,0.07):
            cl=m2cl_contrastive_loss(reps,y,tau,'sum')
            gb,ge=gnorm(cl,bb),gnorm(cl,eb)
            out.append((tau,round(cl.item(),2),gb,ge))
        print(f' CE={lce.item():.3f} |gradCE| backbone={gce_b:.3f} blocks={gce_e:.3f} ({time.time()-t:.0f}s)',flush=True)
        for tau,v,gb,ge in out:
            print(f'  tau={tau}: CL value={v}  |gradCL| backbone={gb:.3f} blocks={ge:.3f}  |  alpha*gradCL/gradCE backbone: a=.003 {0.003*gb/gce_b:.3f}, a=.01 {0.01*gb/gce_b:.3f}, a=.03 {0.03*gb/gce_b:.3f}; blocks a=.003 {0.003*ge/gce_e:.3f}',flush=True)
