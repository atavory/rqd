#!/usr/bin/env python3
"""Full grid: the comprehensive operating envelope.

dims × archs × alphas × freeze_depths × seeds × periods.
~50K RQ fits. Takes hours.

Usage:
    python3 run_full_grid.py --seeds 10
"""
from __future__ import annotations
import argparse,json,logging,time
import numpy as np
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

def main():
    p=argparse.ArgumentParser();p.add_argument("--seeds",type=int,default=10)
    p.add_argument("--n-samples",type=int,default=30000)
    p.add_argument("--json-output",type=str,default="full_grid.json");args=p.parse_args()
    t0=time.time();results=[];count=0
    dims=[32,64,128,256,512]
    alphas=[0.5,1.0,2.0,3.0,5.0,7.0,10.0]
    archs={
        "uniform_64_4":(64,4),"funnel_4":([16,16,256,256],4),
        "uniform_64_6":(64,6),"funnel_6":([16,16,16,64,256,256],6),
        "steep_funnel_4":([8,8,512,512],4),
    }
    total=len(dims)*len(alphas)*len(archs)*args.seeds
    log.info(f"Full grid: {total} combos")
    for dim in dims:
        for alpha in alphas:
            for aname,(codes,m) in archs.items():
                for seed in range(args.seeds):
                    count+=1
                    rng=np.random.RandomState(seed)
                    nc=20;centers=rng.randn(nc,dim).astype(np.float32)*3.0
                    labels=rng.randint(0,nc,size=args.n_samples)
                    X0=centers[labels]+rng.randn(args.n_samples,dim).astype(np.float32)*0.5
                    d=rng.randn(dim).astype(np.float32);d/=np.linalg.norm(d)
                    X1=(X0+d*alpha).astype(np.float32)
                    rq0=RQ(m,codes,dim).fit(X0,seed=seed)
                    rq_full=RQ(m,codes,dim).fit(X1,seed=seed+500)
                    mse_full=rq_full.mse(X1);mse_frz=rq0.mse(X1)
                    for fd in range(m+1):
                        if fd==0:
                            mse_s=mse_full;rho=1.0
                        elif fd==m:
                            mse_s=mse_frz;rho=0.0
                        else:
                            rq_w=warm_retrain(rq0,X1,fd,seed=seed)
                            mse_s=rq_w.mse(X1)
                            den=mse_frz-mse_full
                            rho=1.0-(mse_s-mse_full)/den if abs(den)>1e-12 else 1.0
                        # prefix consistency
                        if fd>0 and fd<m:
                            c0=np.column_stack(rq0.encode(X1,fd))
                            cw=np.column_stack(rq_w.encode(X1,fd))
                            pfx=float(np.all(c0==cw,axis=1).mean())
                        elif fd==0:pfx=0.0
                        else:pfx=1.0
                        results.append({"dim":dim,"alpha":alpha,"arch":aname,"m":m,
                            "freeze_depth":fd,"seed":seed,"mse":mse_s,"mse_frozen":mse_frz,
                            "mse_full":mse_full,"recovery":rho,"prefix_consistency":pfx})
                    if count%50==0:
                        with open(args.json_output,"w") as f:json.dump(results,f)
                        log.info(f"  {count}/{total} ({100*count/total:.0f}%) d={dim} a={alpha} {aname}")
    with open(args.json_output,"w") as f:json.dump(results,f)
    log.info(f"Done in {time.time()-t0:.0f}s. {len(results)} rows from {count} combos.")

if __name__=="__main__":main()
