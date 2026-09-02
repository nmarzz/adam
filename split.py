import numpy as np, time

def fvals(q, W):
    z = q @ W.T; f = np.tanh(z)
    return f, (1.0 - f**2)[..., :, None] * W

KEYS = ['xx','xy','yx','yy']

def sample(n, D, M, Th, W, prm, rs):
    b1,b2,b1p,b2p,c,cp,eps = prm
    X = rs.randn(n,D); Y = rs.randn(n,D)
    fx,Gx = fvals(X@Th, W); fy,Gy = fvals(Y@Th, W)
    Gx,Gy = Gx[:,:,:M], Gy[:,:,:M]
    A = (fx**2)[:,:,None]*(X**2)[:,None,:]; B = (fy**2)[:,:,None]*(Y**2)[:,None,:]
    S1 = b1*A+b2*B+c+eps; S2 = b1p*A+b2p*B+cp+eps
    g1 = {'x': X[:,None,:]*((b2*B+c+eps)/S1**1.5),
          'y': -X[:,None,:]*(b2*(fx*fy)[:,:,None]*(Y**2)[:,None,:]/S1**1.5)}
    g2 = {'y': Y[:,None,:]*((b1p*A+cp+eps)/S2**1.5),
          'x': -Y[:,None,:]*(b1p*(fx*fy)[:,:,None]*(X**2)[:,None,:]/S2**1.5)}
    xy = np.einsum('ni,ni->n',X,Y)
    gram = {'xx':np.einsum('ni,ni->n',X,X),'xy':xy,'yx':xy,'yy':np.einsum('ni,ni->n',Y,Y)}
    GG = {k: np.einsum('nas,nbs->nab', {'x':Gx,'y':Gy}[k[0]], {'x':Gx,'y':Gy}[k[1]]) for k in KEYS}
    return g1, g2, gram, GG

def stageA(D,M,Th,W,prm,N,chunk,rs):                    # build mean matrix -> directions
    acc = {k: np.zeros((M,D,M,D)) for k in KEYS}; done=0
    while done < N:
        n=min(chunk,N-done); g1,g2,gram,GG = sample(n,D,M,Th,W,prm,rs)
        for k in KEYS:
            coef = GG[k]*gram[k][:,None,None]
            for a in range(M):
                for b in range(M):
                    acc[k][a,:,b,:] += (g1[k[0]][:,a,:]*coef[:,a,b,None]).T @ g2[k[1]][:,b,:]
        done+=n
    return {k: acc[k].reshape(M*D,M*D)/N for k in KEYS}

def stageB(D,M,Th,W,prm,N,chunk,rs,dirs):               # unbiased u^T . v on fresh samples
    sums={k:0.0 for k in dirs}; sqs={k:0.0 for k in dirs}; done=0
    while done < N:
        n=min(chunk,N-done); g1,g2,gram,GG = sample(n,D,M,Th,W,prm,rs)
        for name,(u,v) in dirs.items():
            val=np.zeros(n)
            for k in KEYS:
                p=np.einsum('ai,nai->na',u,g1[k[0]]); q=np.einsum('bk,nbk->nb',v,g2[k[1]])
                val += np.einsum('nab,na,nb->n', GG[k]*gram[k][:,None,None], p, q)
            sums[name]+=val.sum(); sqs[name]+=(val**2).sum()
        done+=n
    out={}
    for name in dirs:
        m=sums[name]/N; sd=np.sqrt(max(sqs[name]/N-m*m,0)/N)
        out[name]=(m,sd)
    return out

def run(D, NA=25_000, NB=400_000, M=2, Mplus=3, chunk=4000):
    Th=np.linalg.qr(np.random.randn(D,Mplus))[0]; W=np.random.randn(M,Mplus)/np.sqrt(Mplus)
    c=np.zeros(D); cp=np.zeros(D); eps=1.0
    prm=(0.5,0.5,0.5,0.5,c,cp,eps)
    rs=np.random.RandomState(np.random.randint(1<<30))
    Ms=stageA(D,M,Th,W,prm,NA,chunk,rs)
    dirs={}
    for name,mat in [('TOTAL',sum(Ms.values())),('xx',Ms['xx']),('xy',Ms['xy'])]:
        U,S,Vt=np.linalg.svd(mat); dirs[name]=(U[:,0].reshape(M,D),Vt[0].reshape(M,D))
    return stageB(D,M,Th,W,prm,NB,chunk,rs,dirs)

if __name__=='__main__':
    print(f"{'D':>5} | " + " | ".join(f"{n:>22}" for n in ('TOTAL','xx','xy')))
    import sys
    for D in (20,40,80,160,320):
        t=time.time(); r=run(D)
        row=" | ".join(f"{r[n][0]:9.4f} +/- {r[n][1]:6.4f}" for n in ('TOTAL','xx','xy'))
        print(f"{D:>5} | {row}   ({time.time()-t:.0f}s)", flush=True)
