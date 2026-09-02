import numpy as np, time
from split import sample, KEYS

def build(D,M,Th,W,prm,N,chunk,rs):          # unbiased estimate of E[J1J2^T], per (z,z') and total
    acc={k:np.zeros((M,D,M,D)) for k in KEYS}; done=0
    while done<N:
        n=min(chunk,N-done); g1,g2,gram,GG=sample(n,D,M,Th,W,prm,rs)
        for k in KEYS:
            coef=GG[k]*gram[k][:,None,None]
            for a in range(M):
                for b in range(M):
                    acc[k][a,:,b,:]+=(g1[k[0]][:,a,:]*coef[:,a,b,None]).T@g2[k[1]][:,b,:]
        done+=n
    return {k:acc[k].reshape(M*D,M*D)/N for k in KEYS}

def fro2(D,M,Th,W,prm,N,chunk,rs):           # <M1,M2>_F  is unbiased for ||M||_F^2
    A=build(D,M,Th,W,prm,N,chunk,rs); B=build(D,M,Th,W,prm,N,chunk,rs)
    out={k:float((A[k]*B[k]).sum()) for k in KEYS}
    TA,TB=sum(A.values()),sum(B.values())
    out['TOTAL']=float((TA*TB).sum())
    out['_op']=float(np.linalg.norm((TA+TB)/2,2)); out['_noise']=float(np.linalg.norm((TA-TB)/2,2))
    return out

M,Mplus,N,REP=1,2,20_000,5
print("||M||_F / sqrt(D)   -- if E[J1 J2^T] has bounded op norm this is bounded by sqrt(M)=1")
print(f"{'D':>5} | {'TOTAL':>16} | {'xx':>16} | {'xy':>16} |   op(noisy)  noise")
for D in (20,40,80,160,320):
    Th=np.linalg.qr(np.random.randn(D,Mplus))[0]; W=np.random.randn(M,Mplus)/np.sqrt(Mplus)
    c=np.zeros(D); prm=(0.5,0.5,0.5,0.5,c,c,1.0)
    rs=np.random.RandomState(np.random.randint(1<<30))
    t=time.time(); reps=[fro2(D,M,Th,W,prm,N,4000,rs) for _ in range(REP)]
    cells=[]
    for k in ('TOTAL','xx','xy'):
        v=np.array([max(r[k],0)**0.5/np.sqrt(D) for r in reps])
        cells.append(f"{v.mean():8.4f} +/-{v.std(ddof=1)/np.sqrt(REP):5.4f}")
    op=np.mean([r['_op'] for r in reps]); nz=np.mean([r['_noise'] for r in reps])
    print(f"{D:>5} | " + " | ".join(cells) + f" |  {op:8.4f}  {nz:7.4f}   ({time.time()-t:.0f}s)", flush=True)
