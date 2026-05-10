#!/usr/bin/env python3
"""Scale test: 1M items at d=128. Proves the method scales.

1M items, d=128, uniform and funnel, multiple drift magnitudes, 3 seeds.
Each k-means fit takes ~10 min. Total ~2-3 hours.

Usage:
    python3 run_scale_1m.py --seeds 3
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
    def fit(s,X,n_iter=15,seed=42):
        rng=np.random.RandomState(seed);r=X.copy();s.cb=[]
        for i in range(s.m):c=_kmeans(r,s.K[i],n_iter=n_iter,rng=rng);s.cb.append(c);a=_assign(r,c);r=r-c[a]
        return s
    def mse(s,X):
        r=X.copy()
        for c in s.cb:a=_assign(r,c);r=r-c[a]
        return float(np.mean(np.sum(r**2,axis=1)))
def warm_retrain(rq,X,fd,n_iter=15,seed=42):
    rq2=RQ(rq.m,rq.K,rq.dim);rq2.cb=[c.copy() for c in rq.cb];r=X.copy()
    for i in range(fd):a=_assign(r,rq2.cb[i]);r=r-rq2.cb[i][a]
    rng=np.random.RandomState(seed)
    for i in range(fd,rq.m):c=_kmeans(r,rq.K[i],n_iter=n_iter,rng=rng,init=rq2.cb[i]);rq2.cb[i]=c;a=_assign(r,c);r=r-c[a]
    return rq2

def main():
    p=argparse.ArgumentParser();p.add_argument("--seeds",type=int,default=3)
    p.add_argument("--n",type=int,default=1000000);p.add_argument("--dim",type=int,default=128)
    p.add_argument("--json-output",type=str,default="scale_1m.json");args=p.parse_args()
    t0=time.time();results=[]
    alphas=[1.0,3.0,5.0]
    archs=[("uniform_64_4",64,4),("funnel_4",[16,16,256,256],4)]
    for alpha in alphas:
        for aname,codes,m in archs:
            fd=m//2
            for seed in range(args.seeds):
                log.info(f"n={args.n} alpha={alpha} {aname} seed={seed}")
                rng=np.random.RandomState(seed);nc=50
                centers=rng.randn(nc,args.dim).astype(np.float32)*3.0
                labels=rng.randint(0,nc,size=args.n)
                X0=centers[labels]+rng.randn(args.n,args.dim).astype(np.float32)*0.5
                d=rng.randn(args.dim).astype(np.float32);d/=np.linalg.norm(d)
                X1=(X0+d*alpha).astype(np.float32)
                log.info(f"  fitting source...");st=time.time()
                rq0=RQ(m,codes,args.dim).fit(X0,seed=seed)
                log.info(f"  source fit in {time.time()-st:.0f}s")
                log.info(f"  fitting full...");st=time.time()
                rq_full=RQ(m,codes,args.dim).fit(X1,seed=seed+500)
                log.info(f"  full fit in {time.time()-st:.0f}s")
                log.info(f"  warm retrain...");st=time.time()
                rq_warm=warm_retrain(rq0,X1,fd,seed=seed)
                log.info(f"  warm retrain in {time.time()-st:.0f}s")
                mf,mw,mfull=rq0.mse(X1),rq_warm.mse(X1),rq_full.mse(X1)
                den=mf-mfull;rho=1.0-(mw-mfull)/den if abs(den)>1e-12 else 1.0
                results.append({"n":args.n,"dim":args.dim,"alpha":alpha,"arch":aname,"seed":seed,
                    "mse_frozen":mf,"mse_warm":mw,"mse_full":mfull,"recovery":rho,
                    "time_source_fit":time.time()-t0})
                with open(args.json_output,"w") as f:json.dump(results,f)
                log.info(f"  rho={rho:.3f}")
    log.info(f"Done in {time.time()-t0:.0f}s. {len(results)} rows.")

if __name__=="__main__":main()
