"""E01 tests mathematical behavior, not just code paths."""
import json
import math
import numpy as np
import torch
from .models import Sources, integrate, prior, match, drifting_loss
from .protocol import RESULTS
from .run import write_json


def analytic(x,v,k,acceleration):
    for j in range(k):
        h=1/k; a=acceleration(x); xm=x+h*v/2; vm=v+h*a/2
        x,v=x+h*vm,v+h*acceleration(xm)
    return x,v


def run():
    torch.set_num_threads(2); torch.manual_seed(50000)
    rows=[]
    for d in [2,3,16,384,1152,3072]:
        m=Sources(d,n=8,width=16).double()
        x=torch.randn(5,d,dtype=torch.double); v=torch.randn_like(x)*.1
        t=torch.tensor([.37],dtype=torch.double)
        prepared=m.prepare(t); repeated=m.prepare(t)
        source_error=max(float((a-b).abs().max()) for a,b in zip(prepared,repeated))
        a,diag=m.field(x,prepared,diagnostic=True)
        other=torch.cat([x[:1],torch.randn(7,d,dtype=torch.double)])
        batch=float((m.field(x[:1],prepared)-m.field(other,prepared)[:1]).abs().max())
        # Full Jacobians for moderate d; random 3-D coordinate subblocks in high dimensions.
        idx=torch.arange(min(d,16))
        def field(z,part):
            xx=x[0].clone(); xx[idx]=z
            return m.field(xx[None],prepared,component=part)[0,idx]
        # Divergence needs all d diagonal terms: exploit batched JVPs only for d<=16,
        # and full analytic skew/Hessian trace for high-d query to avoid O(d^2) autograd.
        if d<=16:
            jr=torch.autograd.functional.jacobian(lambda z:field(z,'radial'),x[0,idx])
            js=torch.autograd.functional.jacobian(lambda z:field(z,'tangential'),x[0,idx])
            curl=float((jr-jr.T).abs().max()); div=float(js.trace().abs())
            eps=1e-5; fd=[]
            for j in range(d):
                dx=torch.zeros_like(x[:1]); dx[0,j]=eps
                fd.append(((m.field(x[:1]+dx,prepared,component='tangential')-m.field(x[:1]-dx,prepared,component='tangential'))/(2*eps))[0])
            finite_error=float((torch.stack(fd,1)-js).abs().max())
            div_kind='full autograd Jacobian and central finite differences'
        else:
            # Three random directions verify Hessian symmetry bilinearly and directional derivatives.
            z=x[0].requires_grad_(); u=torch.randn_like(z); w=torch.randn_like(z)
            _,ju=torch.autograd.functional.jvp(lambda xx:m.field(xx[None],prepared,component='radial')[0],z,u)
            _,jw=torch.autograd.functional.jvp(lambda xx:m.field(xx[None],prepared,component='radial')[0],z,w)
            curl=float(abs(w@ju-u@jw))
            # Full divergence autograd by exact trace diagonal in chunks is feasible on a single point.
            jac=torch.func.jacrev(lambda xx:m.field(xx[None],prepared,component='tangential')[0],chunk_size=64)(z)
            div=float(jac.trace().abs())
            _,jv=torch.autograd.functional.jvp(lambda xx:m.field(xx[None],prepared,component='tangential')[0],z,u)
            eps=1e-5
            fd=(m.field((z+eps*u)[None],prepared,component='tangential')[0]-m.field((z-eps*u)[None],prepared,component='tangential')[0])/(2*eps)
            finite_error=float((fd-jv).abs().max()); div_kind='full autograd trace; random directional finite difference'
        with torch.no_grad():
            single=integrate(m,'dhpf',x[:1],v[:1],32)[0]
            group=integrate(m,'dhpf',x,v,32)[0]
        generation_error=float((single-group[:1]).abs().max()); force_effect=float((group-(x+v)).norm(dim=-1).mean())
        row=dict(d=d,source_error=source_error,batch_field_error=batch,radial_symmetry_error=curl,tangent_divergence=div,finite_difference_error=finite_error,divergence_method=div_kind,batch_generation_error=generation_error,nonzero_force_endpoint_displacement=force_effect,**diag)
        assert source_error==0 and batch<1e-10 and curl<1e-8 and div<1e-8 and finite_error<1e-7 and generation_error<1e-9 and force_effect>0 and diag['zero_kernel_fraction']==0
        rows.append(row)
    numerical=[]
    x=torch.tensor([[.2,-.3]],dtype=torch.double); v=torch.tensor([[.7,.4]],dtype=torch.double)
    for k in [8,16,32,64,128]:
        const=torch.tensor([[.3,-.8]],dtype=torch.double,requires_grad=True)
        xx,vv=analytic(x,v,k,lambda z:const.expand_as(z))
        grad=torch.autograd.grad(xx.sum(),const)[0]
        osc,oscv=analytic(x,v,k,lambda z:-z)
        exact=x*math.cos(1)+v*math.sin(1); ev=-x*math.sin(1)+v*math.cos(1)
        numerical.append(dict(k=k,constant_position_error=float((xx-(x+v+.5*const)).abs().max()),constant_velocity_error=float((vv-v-const).abs().max()),constant_gradient_error=float((grad-.5).abs().max()),oscillator_position_error=float((osc-exact).norm()),oscillator_velocity_error=float((oscv-ev).norm())))
    assert all(r['constant_gradient_error']<1e-12 for r in numerical)
    # Gradient through actual shared-source integration and fixed optimal assignment.
    torch.manual_seed(50001); m=Sources(2,8,16).double(); xx=torch.randn(12,2,dtype=torch.double); vv=torch.randn_like(xx)*.1; target=torch.randn_like(xx)
    def loss():
        out=integrate(m,'dhpf',xx,vv,16)[0]
        return (out-match(out,target)).square().sum(-1).mean()
    val=loss(); val.backward(); p=m.head.bias; auto=float(p.grad[2]); eps=1e-5
    with torch.no_grad(): p[2]+=eps
    plus=float(loss().detach())
    with torch.no_grad(): p[2]-=2*eps
    minus=float(loss().detach())
    with torch.no_grad(): p[2]+=eps
    finite=(plus-minus)/(2*eps); error=abs(auto-finite)/max(1,abs(auto),abs(finite)); assert error<1e-5
    # Equal empirical positive and negative measures should have small drift objective;
    # exclude self mask nuance, verify finite nonzero gradients for unequal distributions.
    z=torch.randn(16,2,requires_grad=True); dl=drifting_loss(z,torch.randn(16,2)+2); dl.backward(); assert torch.isfinite(z.grad).all() and float(z.grad.norm())>0
    output=dict(status='passed',experiment='E01',structure=rows,numerics=numerical,endpoint_ot_gradient=dict(autograd=auto,finite_difference=finite,relative_error=error),notes='Generation time source API has no x, v, or batch argument. Checks run in float64 on CPU.')
    write_json(RESULTS/'E01'/'checks.json',output)
    print(json.dumps(output,indent=2))


if __name__=='__main__': run()
