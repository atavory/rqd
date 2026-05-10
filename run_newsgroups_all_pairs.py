#!/usr/bin/env python3
"""20newsgroups: all pairwise domain shifts between 20 categories.

20 source categories × 19 target categories = 380 pairs.
TF-IDF + SVD to d=128. Uniform and funnel. 5 seeds.

Usage:
    python3 run_newsgroups_all_pairs.py --seeds 5
"""
from __future__ import annotations
import argparse,json,logging,time,os
import numpy as np
os.environ['SCIKIT_LEARN_DATA']='/tmp/sklearn_data'
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
    p.add_argument("--json-output",type=str,default="newsgroups_all_pairs.json");args=p.parse_args()
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    t0=time.time();results=[]
    cats=fetch_20newsgroups(subset='all').target_names
    log.info(f"{len(cats)} categories: {cats}")
    # Embed each category separately
    cat_data={}
    for cat in cats:
        data=fetch_20newsgroups(subset='all',categories=[cat],remove=('headers','footers','quotes'))
        tfidf=TfidfVectorizer(max_features=5000)
        X_tfidf=tfidf.fit_transform(data.data)
        svd=TruncatedSVD(n_components=128,random_state=42)
        X_svd=svd.fit_transform(X_tfidf).astype(np.float32)
        norms=np.linalg.norm(X_svd,axis=1,keepdims=True)+1e-8
        cat_data[cat]=X_svd/norms
        log.info(f"  {cat}: {cat_data[cat].shape}")
    # All pairs
    count=0;total=len(cats)*(len(cats)-1)*2*args.seeds
    log.info(f"Running {total} combos")
    for src_cat in cats:
        for tgt_cat in cats:
            if src_cat==tgt_cat:continue
            X0=cat_data[src_cat];X1=cat_data[tgt_cat]
            for aname,codes,m in [("uniform_64_4",64,4),("funnel_4",[16,16,256,256],4)]:
                fd=m//2
                for seed in range(args.seeds):
                    count+=1
                    rq0=RQ(m,codes,128).fit(X0,seed=seed)
                    rq_full=RQ(m,codes,128).fit(X1,seed=seed+500)
                    rq_warm=warm_retrain(rq0,X1,fd,seed=seed)
                    mf,mw,mfull=rq0.mse(X1),rq_warm.mse(X1),rq_full.mse(X1)
                    den=mf-mfull;rho=1.0-(mw-mfull)/den if abs(den)>1e-12 else 1.0
                    results.append({"source":src_cat,"target":tgt_cat,"arch":aname,"seed":seed,"recovery":rho,
                                   "mse_frozen":mf,"mse_warm":mw,"mse_full":mfull})
                    if count%100==0:
                        with open(args.json_output,"w") as f:json.dump(results,f)
                        log.info(f"  {count}/{total} {src_cat}->{tgt_cat} {aname}: rho={rho:.3f}")
    with open(args.json_output,"w") as f:json.dump(results,f)
    log.info(f"Done in {time.time()-t0:.0f}s. {len(results)} rows.")

if __name__=="__main__":main()
