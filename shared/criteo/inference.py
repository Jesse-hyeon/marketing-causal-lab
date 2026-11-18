"""Paired, arm-stratified bootstrap of fixed-score batch targeting policies.

Sort once per score. Bootstrap multiplicities re-create the top-k ranking of
resampled customers, including a partially selected last repeated row. This is
exactly the row bootstrap without sorting 300k observations on every draw.
"""
import numpy as np


def verdict(lo, hi):
    return '지지' if lo > 0 else ('반대 근거' if hi < 0 else '판단 보류')


def policy_stats(scores, t, y, e, mu0, mu1, weights=None, k=.3, seed=20261111):
    t,y,e,mu0,mu1=[np.asarray(a,dtype=float) for a in (t,y,e,mu0,mu1)]
    n=len(t)
    weights=np.ones(n) if weights is None else np.asarray(weights,dtype=float)
    e=np.clip(e,.03,.97)
    z=t*y/e-(1-t)*y/(1-e)
    dr0=mu0+(1-t)*(y-mu0)/(1-e)
    dr=mu1-mu0+t*(y-mu1)/e-(1-t)*(y-mu0)/(1-e)
    rng=np.random.default_rng(seed)
    tie=rng.random(n)
    order=np.lexsort((tie,-np.asarray(scores)))
    return _stats(order,weights,t,y,z,dr0,dr,k)


def _stats(order,weights,t,y,z,dr0,dr,k):
    n=weights.sum()
    w=weights[order]
    z0=z[order]
    cum=np.cumsum(w)
    take=np.clip(np.ceil(k*n)-(cum-w),0,w)
    q=np.ceil(k*n)/n
    inc=np.dot(take,dr[order])/n
    random_inc=q*np.dot(weights,dr)/n
    base=np.dot(weights,dr0)/n
    cumz=np.cumsum(w*z0)
    auuc=np.sum(w*(cumz-w*z0/2))/n**2
    selected_t=np.dot(take,t[order])
    selected_c=np.dot(take,1-t[order])
    uplift=(np.dot(take,t[order]*y[order])/selected_t-
            np.dot(take,(1-t[order])*y[order])/selected_c) if min(selected_t,selected_c)>0 else np.nan
    return dict(PolicyValue=base+inc,RandomValue=base+random_inc,
                PolicyGain=inc-random_inc,AUUC=auuc,
                Qini=auuc-np.dot(weights,z)/(2*n),
                IncrementalRate=np.dot(take,z0)/n,
                UpliftAt30=uplift,TargetFraction=q)


def bootstrap(scores, t, y, e, mu0, mu1, contrasts, B=399,k=.3,seed=20261111):
    """contrasts={name: (score_key, comparator_key_or_None)}.

    A None comparator means the analytical random policy with equal batch size.
    Returns point metrics and nominal 95% percentile intervals. Conditional on
    fitted models; does not include their training/selection uncertainty.
    """
    t,y,e,mu0,mu1=[np.asarray(a,dtype=float) for a in (t,y,e,mu0,mu1)]
    n=len(t)
    if not all(len(a)==n for a in (y,e,mu0,mu1)):
        raise ValueError('equal-length evaluation arrays required')
    e=np.clip(e,.03,.97)
    z=t*y/e-(1-t)*y/(1-e)
    dr0=mu0+(1-t)*(y-mu0)/(1-e)
    dr=mu1-mu0+t*(y-mu1)/e-(1-t)*(y-mu0)/(1-e)
    rng=np.random.default_rng(seed)
    tie=np.random.default_rng(seed).random(n)
    orders={key:np.lexsort((tie,-np.asarray(s))) for key,s in scores.items()}
    point={key:_stats(order,np.ones(n),t,y,z,dr0,dr,k) for key,order in orders.items()}
    draws={name:[] for name in contrasts}
    arms=[np.flatnonzero(t==arm) for arm in [0,1]]
    if any(len(a)==0 for a in arms):
        raise ValueError('both treatment arms required')
    for _ in range(B):
        ids=np.concatenate([rng.choice(arm,len(arm),replace=True) for arm in arms])
        weights=np.bincount(ids,minlength=n)
        stats={key:_stats(order,weights,t,y,z,dr0,dr,k) for key,order in orders.items()}
        for name,(a,b) in contrasts.items():
            draws[name].append(stats[a]['PolicyGain'] if b is None else stats[a]['PolicyValue']-stats[b]['PolicyValue'])
    result={}
    for name,(a,b) in contrasts.items():
        point_value=point[a]['PolicyGain'] if b is None else point[a]['PolicyValue']-point[b]['PolicyValue']
        lo,hi=np.quantile(draws[name],[.025,.975])
        result[name]=dict(estimate=float(point_value),lo=float(lo),hi=float(hi),verdict=verdict(lo,hi))
    return point,result
