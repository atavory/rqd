#!/usr/bin/env python3
"""Retrain cost sweep: how many k-means iters are needed?

Sweep n_iter from 1 to 50 for suffix retraining. 5 iters might get 95% of the gain.
Also sweep source/target data sizes.

Usage:
    python3 run_retrain_cost_sweep.py --seeds 5
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
def warm_retrain(rq,X,fd,n_iter=20,seed=42):
    rq2=RQ(rq.m,rq.K,rq.dim);rq2.cb=[c.copy() for c in rq.cb];r=X.copy()
    for i in range(fd):a=_assign(r,rq2.cb[i]);r=r-rq2.cb[i][a]
    rng=np.random.RandomState(seed)
    for i in range(fd,rq.m):c=_kmeans(r,rq.K[i],n_iter=n_iter,rng=rng,init=rq2.cb[i]);rq2.cb[i]=c;a=_assign(r,c);r=r-c[a]
    return rq2

def main():
    p=argparse.ArgumentParser();p.add_argument("--seeds",type=int,default=5)
    p.add_argument("--json-output",type=str,default="retrain_cost_sweep.json");args=p.parse_args()
    t0=time.time();results=[];dim=128;alpha=3.0;m=4;codes=64;fd=2
    n_iters=[1,2,3,5,7,10,15,20,30,50]
    target_sizes=[500,1000,2000,5000,10000,20000,50000]
    # Part 1: n_iter sweep
    log.info("=== n_iter sweep ===")
    for ni in n_iters:
        for seed in range(args.seeds):
            rng=np.random.RandomState(seed);X0=rng.randn(20000,dim).astype(np.float32)*3.0
            d=rng.randn(dim).astype(np.float32);d/=np.linalg.norm(d);X1=(X0+d*alpha).astype(np.float32)
            rq0=RQ(m,codes,dim).fit(X0,seed=seed);rq_warm=warm_retrain(rq0,X1,fd,n_iter=ni,seed=seed)
            rq_full=RQ(m,codes,dim).fit(X1,seed=seed+500)
            mf,mw,mfull=rq0.mse(X1),rq_warm.mse(X1),rq_full.mse(X1)
            den=mf-mfull;rho=1.0-(mw-mfull)/den if abs(den)>1e-12 else 1.0
            results.append({"sweep":"n_iter","n_iter":ni,"target_n":20000,"seed":seed,"recovery":rho,"mse_warm":mw})
        with open(args.json_output,"w") as f:json.dump(results,f)
        rhos=[r["recovery"] for r in results if r["sweep"]=="n_iter" and r["n_iter"]==ni]
        log.info(f"  n_iter={ni}: rho={np.mean(rhos):.3f}")
    # Part 2: target data size sweep
    log.info("=== target size sweep ===")
    for tn in target_sizes:
        for seed in range(args.seeds):
            rng=np.random.RandomState(seed);X0=rng.randn(20000,dim).astype(np.float32)*3.0
            d=rng.randn(dim).astype(np.float32);d/=np.linalg.norm(d);X1_full=(X0+d*alpha).astype(np.float32)
            X1=X1_full[rng.choice(len(X1_full),min(tn,len(X1_full)),replace=False)]
            rq0=RQ(m,codes,dim).fit(X0,seed=seed);rq_warm=warm_retrain(rq0,X1,fd,seed=seed)
            rq_full=RQ(m,codes,dim).fit(X1,seed=seed+500)
            mf,mw,mfull=rq0.mse(X1),rq_warm.mse(X1),rq_full.mse(X1)
            den=mf-mfull;rho=1.0-(mw-mfull)/den if abs(den)>1e-12 else 1.0
            results.append({"sweep":"target_n","n_iter":20,"target_n":tn,"seed":seed,"recovery":rho,"mse_warm":mw})
        with open(args.json_output,"w") as f:json.dump(results,f)
        rhos=[r["recovery"] for r in results if r["sweep"]=="target_n" and r["target_n"]==tn]
        log.info(f"  target_n={tn}: rho={np.mean(rhos):.3f}")
    log.info(f"Done in {time.time()-t0:.0f}s")

if __name__=="__main__":main()
