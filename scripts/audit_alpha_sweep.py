"""Validation scores and backbone distances across the alpha and tau screening runs.

  python scripts/audit_alpha_sweep.py <m2cl_improvement_checkpoints zip>
"""
import io, sys, zipfile, torch
p=sys.argv[1]
runs={}
with zipfile.ZipFile(p) as z:
    for n in z.namelist():
        if n.endswith('best.pt'):
            try: ck=torch.load(io.BytesIO(z.read(n)),map_location='cpu',weights_only=True)
            except Exception: continue
            runs[n.split('/')[-2]]=ck
def bb(ck): return {k:v.float() for k,v in ck['model_state'].items() if k.startswith(('layer0','layer1','layer2','layer3','layer4')) and 'running' not in k and 'num_batches' not in k}
def dist(a,b):
    A,B=bb(a),bb(b); return float(sum(((A[k]-B[k])**2).sum() for k in A if k in B)**.5)
order=sorted(runs,key=lambda r:(runs[r]['config']['arch_taps'],runs[r]['config']['alpha'],runs[r]['config'].get('tau'),str(runs[r]['config']['augmentation'])))
print('run alpha taps tau aug best_val best_ep last_loss last_val cl_keys')
for r in order:
    c=runs[r]['config']; h=runs[r]['history']; last=h[-1]
    be=max(h,key=lambda e:e['val_acc'])['epoch']
    print(r,c['alpha'],c['arch_taps'],c.get('tau'),c['augmentation'],round(runs[r]['best_val'],3),be,round(last['loss'],5),round(last['val_acc'],3),sorted(k for k in last if 'cl' in k or 'con' in k))
base=[r for r in order if runs[r]['config']['alpha']==0.0][0]
print('\nBackbone distance from alpha=0 run (best checkpoints):')
for r in order:
    c=runs[r]['config']
    if c['arch_taps']==4: print(' alpha',c['alpha'],round(dist(runs[base],runs[r]),4),'val per epoch equal to alpha0:',sum(abs(a['val_acc']-b['val_acc'])<1e-9 for a,b in zip(runs[base]['history'],runs[r]['history'])),'/',len(runs[r]['history']))
six=[r for r in order if runs[r]['config']['arch_taps']==6 and runs[r]['config']['augmentation']=='baseline']
print('\n6-tap baseline runs:',[(r,runs[r]['config'].get('tau'),round(runs[r]['history'][-1]['loss'],5)) for r in six])
for a in six[1:]: print(' distance',six[0],a,round(dist(runs[six[0]],runs[a]),4))
