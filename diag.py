import numpy as np, time
from split import sample, stageA, KEYS

def diag_est(D,M,Th,W,prm,N,chunk,rs):
    s={k:np.zeros(M*D) for k in KEYS}; s2={k:np.zeros(M*D) for k in KEYS}
    st=np.zeros(M*D); st2=np.zeros(M*D); done=0
    while done<N:
        n=min(chunk,N-done); g1,g2,gram,GG=sample(n,D,M,Th,W,prm,rs)
        tot=np.zeros((n,M,D))
        for k in KEYS:
            caa=np.einsum('naa,n->na',GG[k],gram[k])                 # (n,M)
            v=caa[:,:,None]*g1[k[0]]*g2[k[1]]                        # (n,M,D)
            s[k]+=v.sum(0).ravel(); s2[k]+=(v**2).sum(0).ravel(); tot+=v
        st+=tot.sum(0).ravel(); st2+=(tot**2).sum(0).ravel(); done+=n
    out={}
    for k in KEYS:
        m=s[k]/N; out[k]=(m, np.sqrt(np.maximum(s2[k]/N-m*m,0)/N))
    m=st/N; out['TOTAL']=(m, np.sqrt(np.maximum(st2/N-m*m,0)/N))
    return out

def model(D,M=2,Mplus=3):
    return (np.linalg.qr(np.random.randn(D,Mplus))[0],
            np.random.randn(M,Mplus)/np.sqrt(Mplus))

# --- 1. is the mean matrix diagonal-dominant? ---
print("diagonal-dominance check  (op norm vs max|diag| of the SAME estimate)")
for D in (20,40,80):
    M=2; Th,W=model(D); c=np.zeros(D)
    prm=(0.5,0.5,0.5,0.5,c,c,1.0); rs=np.random.RandomState(np.random.randint(1<<30))
    Ms=stageA(D,M,Th,W,prm,120_000,4000,rs); T=sum(Ms.values())
    print(f"  D={D:4d}  op={np.linalg.norm(T,2):7.4f}  max|diag|={np.abs(np.diag(T)).max():7.4f}"
          f"  offdiag_frac={np.linalg.norm(T-np.diag(np.diag(T)))/np.linalg.norm(T):5.3f}", flush=True)

# --- 2. diagonal-only scaling, cheap and unbiased ---
print("\nmax|diag| of E[J1 J2^T]   (+/- = max over i of the per-entry stderr)")
print(f"{'D':>6} | {'TOTAL':>18} | {'xx':>18} | {'xy':>18}")
for D in (20,40,80,160,320,640,1280):
    M=2; Th,W=model(D); c=np.zeros(D)
    prm=(0.5,0.5,0.5,0.5,c,c,1.0); rs=np.random.RandomState(np.random.randint(1<<30))
    t=time.time(); r=diag_est(D,M,Th,W,prm,300_000,3000,rs)
    row=" | ".join(f"{np.abs(r[k][0]).max():9.4f} +/-{r[k][1].max():6.4f}" for k in ('TOTAL','xx','xy'))
    print(f"{D:>6} | {row}   ({time.time()-t:.0f}s)", flush=True)
