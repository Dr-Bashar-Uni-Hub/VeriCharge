from common import *
rng=np.random.default_rng(42); B=1000; L=672
N=len(test['y']); nb=int(np.ceil(N/L))
def bidx():
    s=rng.integers(0,N-L+1,nb); return (s[:,None]+np.arange(L)).ravel()[:N]
IDX=[bidx() for _ in range(B)]
pairs=[('URS (VeriCharge)','Predicted load ŷ (rule)'),('URS (VeriCharge)','Load context (LR)'),
       ('URS + load context (LR)','Load context (LR)'),('URS + load context (LR)','Predicted load ŷ (rule)')]
rows=[]
for j,h in enumerate(H):
    e=np.abs(test['y'][:,j]-test['yhat'][:,j]); fail=(e>eps[j]).astype(float)
    S=build_scores(j,(np.abs(dev['y'][:,j]-dev['yhat'][:,j])<=eps[j]).astype(int))
    for a,b in pairs:
        sa,sb=S[a][1],S[b][1]
        # AURC on failure; and fail-rate at fixed 70% coverage (matched coverage)
        d0=aurc(sa,fail)-aurc(sb,fail); r0=risk_at(sa,fail,.7)-risk_at(sb,fail,.7)
        ds_=[];rs_=[]
        for ix in IDX:
            ds_.append(aurc(sa[ix],fail[ix])-aurc(sb[ix],fail[ix])); rs_.append(risk_at(sa[ix],fail[ix],.7)-risk_at(sb[ix],fail[ix],.7))
        rows.append(dict(horizon=h,A=a,B=b,dAURC_fail=d0,CI_lo=np.percentile(ds_,2.5),CI_hi=np.percentile(ds_,97.5),
                         dFail_at70cov=r0,CI70_lo=np.percentile(rs_,2.5),CI70_hi=np.percentile(rs_,97.5)))
df=pd.DataFrame(rows); df.to_csv(os.path.join(OUT,'A3_paired_bootstrap.csv'),index=False)
pd.set_option('display.width',250); print(df.round(4).to_string(index=False))
