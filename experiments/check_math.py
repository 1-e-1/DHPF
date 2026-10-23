"""Checks that target the V3 structural claims and actual OT/integrator gradients."""
import json
import math
import torch
from torch.autograd.functional import jacobian
from dhpf_experiment import Sources, integrate, ot_loss

torch.set_num_threads(2)
torch.manual_seed(41)
checks={}
for kind in ['radial','tangential']:
    model=Sources(kind,n=5,h=16).double()
    prep=model.prepare(torch.tensor([.37],dtype=torch.double))
    errors=[]
    for x in torch.randn(8,2,dtype=torch.double):
        j=jacobian(lambda z:model.field(z[None],prep,0)[0],x)
        errors.append(float((j-j.T).abs().max() if kind=='radial' else j.trace().abs()))
    checks[kind+'_structure_max_error']=max(errors)
    assert max(errors)<1e-10
    z=torch.randn(4,dtype=torch.double)
    j=jacobian(lambda s:torch.cat([s[2:],model.field(s[:2][None],prep,0)[0]]),z)
    assert abs(float(j.trace()))<1e-10

model=Sources('full',n=4,h=16).double()
x=torch.randn(5,2,dtype=torch.double);v=torch.randn(5,2,dtype=torch.double)
y=torch.randn(5,2,dtype=torch.double)
loss=ot_loss(integrate(model,x,v,4)[0],y)
loss.backward()
analytic=float(model.head.bias.grad[2]); eps=1e-6
with torch.no_grad():
    model.head.bias[2]+=eps
    plus=float(ot_loss(integrate(model,x,v,4)[0],y))
    model.head.bias[2]-=2*eps
    minus=float(ot_loss(integrate(model,x,v,4)[0],y))
    model.head.bias[2]+=eps
numeric=(plus-minus)/(2*eps)
checks['ot_integrator_gradient_relative_error']=abs(numeric-analytic)/max(1,abs(numeric),abs(analytic))
assert checks['ot_integrator_gradient_relative_error']<1e-5
with torch.no_grad():
    prep=model.prepare(torch.tensor([.3],dtype=torch.double))
    together=model.field(x,prep,0)
    separate=torch.cat([model.field(q[None],prep,0) for q in x])
    checks['batch_independence_error']=float((together-separate).abs().max())
    assert checks['batch_independence_error']<1e-12
    ref=integrate(model,x,v,256)[0]
    errors=[float((integrate(model,x,v,k)[0]-ref).norm()) for k in [16,32,64]]
    checks['midpoint_errors_16_32_64']=errors
    assert errors[2]<errors[1]<errors[0]

class Oscillator:
    def prepare(self, times):
        return times
    def field(self, x, prepared, index):
        return -x * (math.pi/2)**2

with torch.no_grad():
    omega=math.pi/2
    exact_x=v/omega
    oscillator_errors=[float((integrate(Oscillator(),x,v,k)[0]-exact_x).norm()) for k in [8,16,32]]
    checks['oscillator_errors_8_16_32']=oscillator_errors
    checks['oscillator_observed_orders']=[math.log(oscillator_errors[i]/oscillator_errors[i+1],2) for i in range(2)]
    assert all(1.8<order<2.2 for order in checks['oscillator_observed_orders'])
print(json.dumps({'passed':True,'checks':checks},indent=2))
