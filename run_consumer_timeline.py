#!/usr/bin/env python3
"""Consumer timeline: downstream proxies through 20 periods of drift.

Tracks when each consumer type breaks under warm-once vs stays fine
under warm-periodic. Uses simple linear classifiers as downstream proxies.

Usage:
    python3 run_consumer_timeline.py --seeds 5
"""
from __future__ import annotations
import argparse,json,logging,math,time
import numpy as np
from scipy.stats import special_ortho_group
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s",datefmt="%H:%M:%S")
log=logging.getLogger(__name__)
def _kmeans(X,k,n_iter=20,rng=None,init=None):
    if rng is None:rng=np.random.RandomState(42)
    n=len(X)
    if init is not None:centroids=init.copy()
    else:
        centroids=np.zeros((k,X.shape[1]),dtype=np.float32);centroids[0]=X[rng.randint(n)]
        for i in range(1,k):
            d=np.min(np.sum((X[:,None,:]-centroids[None,:i,:])**2,axis=2),axis=1);t=d.sum()
            centroids[i]=X[rng.choice(n,p=d/max(t,1e-12))] if t>1e-12 else X[rng.randint(n)]
    for _ in range(n_iter):
        a=np.argmin(np.sum((X[:,None,:]-centroids[None,:,:])**2,axis=2),axis=1)
        for j in range(k):m=a==j;centroids[j]=X[m].mean(axis=0) if m.sum()>0 else centroids[j]
    return centroids
def _assign(X,c):return np.argmin(np.sum((X[:,None,:]-c[None,:,:])**2,axis=2),axis=1).astype(np.int64)
class RQ:
    def __init__(s,m,codes,dim):s.m,s.dim=m,dim;s.K=[codes]*m if isinstance(codes,int) else list(codes);s.cb=[]
    def fit(s,X,n_iter=20,seed=42):
        rng=np.random.RandomState(seed);r=X.copy();s.cb=[]
        for i in range(s.m):c=_kmeans(r,s.K[i],n_iter=n_iter,rng=rng);s.cb.append(c);a=_assign(r,c);r=r-c[a]
        return s
    def mse(s,X):
        r=X.copy()
        for c in s.cb:a=_assign(r,c);r=r-c[a]
        return float(np.mean(np.sum(r**2,axis=1)))
    def encode(s,X,ns=None):
        if ns is None:ns=len(s.cb)
        r=X.copy();codes=[]
        for i in range(ns):a=_assign(r,s.cb[i]);codes.append(a);r=r-s.cb[i][a]
        return codes
def warm_retrain(rq,X,fd,n_iter=20,seed=42):
    rq2=RQ(rq.m,rq.K,rq.dim);rq2.cb=[c.copy() for c in rq.cb];r=X.copy()
    for i in range(fd):a=_assign(r,rq2.cb[i]);r=r-rq2.cb[i][a]
    rng=np.random.RandomState(seed)
    for i in range(fd,rq.m):c=_kmeans(r,rq.K[i],n_iter=n_iter,rng=rng,init=rq2.cb[i]);rq2.cb[i]=c;a=_assign(r,c);r=r-c[a]
    return rq2
def generate(n,dim,nc=20,seed=42):
    rng=np.random.RandomState(seed);centers=rng.randn(nc,dim).astype(np.float32)*3.0
    labels=rng.randint(0,nc,size=n)
    return centers[labels]+rng.randn(n,dim).astype(np.float32)*0.5,centers,labels
def apply_drift(X,centers,period,dim,seed):
    rng=np.random.RandomState(seed+period*1000);X=X.copy()
    drift=np.zeros(dim,dtype=np.float32)
    for t in range(period):
        d=np.random.RandomState(seed+t*1000+1).randn(dim).astype(np.float32)
        drift+=d/np.linalg.norm(d)*0.3
    X=X+drift
    if period>=7:
        R=special_ortho_group.rvs(dim,random_state=np.random.RandomState(seed+7*7777)).astype(np.float32)
        Ri=(0.85)*np.eye(dim,dtype=np.float32)+0.15*R;U,_,Vt=np.linalg.svd(Ri)
        X=X@(U@Vt).astype(np.float32).T
    X=X*(1.0+0.2*math.sin(2*math.pi*period/6))
    return X

def prefix_accuracy(rq_source, rq_current, X, depth):
    """Fraction of items whose prefix matches the source codebook's assignment."""
    c0=np.column_stack(rq_source.encode(X,depth))
    cc=np.column_stack(rq_current.encode(X,depth))
    return float(np.all(c0==cc,axis=1).mean())

def main():
    p=argparse.ArgumentParser();p.add_argument("--seeds",type=int,default=5)
    p.add_argument("--n-periods",type=int,default=20);p.add_argument("--dim",type=int,default=128)
    p.add_argument("--json-output",type=str,default="consumer_timeline.json");args=p.parse_args()
    t0=time.time();results=[];m=4;codes=64;fd=2;n=20000
    for seed in range(args.seeds):
        log.info(f"seed={seed}")
        X0,centers,labels=generate(n,args.dim,seed=seed)
        rq0=RQ(m,codes,args.dim).fit(X0,seed=seed)
        rq_periodic=rq0;rq_once=rq0
        for period in range(args.n_periods+1):
            Xt=X0 if period==0 else apply_drift(X0,centers,period,args.dim,seed)
            if period==1:rq_once=warm_retrain(rq0,Xt,fd,seed=seed+1)
            if period>0:rq_periodic=warm_retrain(rq_periodic,Xt,fd,seed=seed+period)
            rq_full=RQ(m,codes,args.dim).fit(Xt,seed=seed+period*100)
            for depth in [1,2,3,4]:
                for strat,rq_s in [("frozen",rq0),("warm_once",rq_once),("warm_periodic",rq_periodic),("full",rq_full)]:
                    pfx=prefix_accuracy(rq0,rq_s,Xt,depth)
                    results.append({"seed":seed,"period":period,"consumer_depth":depth,"strategy":strat,
                                   "prefix_accuracy":pfx,"mse":rq_s.mse(Xt)})
        with open(args.json_output,"w") as f:json.dump(results,f)
        log.info(f"  done period {args.n_periods}")
    log.info(f"Done in {time.time()-t0:.0f}s. {len(results)} rows.")

if __name__=="__main__":main()
