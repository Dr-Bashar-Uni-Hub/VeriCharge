from common import *
from scipy.stats import spearmanr
z=np.load(R+'revision_B1_ensemble_predictions.npz'); SE=list(z['seeds'])
assert (z['dev_idx']==dev['idx']).all() and (z['test_idx']==test['idx']).all()
E={sp:np.stack([z[f'{sp}_seed_{s}'] for s in SE]).std(0,ddof=1) for sp in ['mv','dev','test']}
# MV signal validation (Table 3 analogue)
mve=np.abs(mvp['seed_42']-mvp['y_true'])
print('MV Spearman(E,|e|):',[round(spearmanr(E['mv'][:,j],mve[:,j])[0],4) for j in range(3)],
      ' MV Spearman(U,|e|):',[round(spearmanr(mvf[f'X_{h}'][:,0],mve[:,j])[0],4) for j,h in enumerate(H)])
def q5ratio(s,e):
    q=np.quantile(s,[.2,.8]); return e[s>=q[1]].mean()/e[s<=q[0]].mean()
print('MV top/bottom quintile MAE ratio E:',[round(q5ratio(E['mv'][:,j],mve[:,j]),3) for j in range(3)])
rows=[];rng=np.random.default_rng(42);N=len(test['y']);L=672;nb=int(np.ceil(N/L))
IDX=[(lambda s:(s[:,None]+np.arange(L)).ravel()[:N])(rng.integers(0,N-L+1,nb)) for _ in range(1000)]
strata_rows=[]
for j,h in enumerate(H):
    yh=test['yhat'][:,j]; e=np.abs(test['y'][:,j]-yh); fail=(e>eps[j]).astype(float); ok=1-fail
    lab=(np.abs(dev['y'][:,j]-dev['yhat'][:,j])<=eps[j]).astype(int); f=dev['fit']
    S=build_scores(j,lab)
    Ud=dev['X'][j]; Ut=np.c_[test['U'][:,j],test['R'][:,j],test['S']]
    Cd=ctx(dev['idx'],dev['yhat'][:,j]); Ct=ctx(test['idx'],yh)
    Ed,Et=E['dev'][:,[j]],E['test'][:,[j]]
    def lr(Xd,Xt):
        sc=StandardScaler().fit(Xd[f]); m=LogisticRegression(C=1.0,max_iter=1000).fit(sc.transform(Xd[f]),lab[f])
        return m.predict_proba(sc.transform(Xd))[:,1], m.predict_proba(sc.transform(Xt))[:,1]
    S['E only (deep ensemble)']=lr(Ed,Et)
    S['ERS (ensemble replaces MC)']=lr(np.c_[Ed,Ud[:,1:]],np.c_[Et,Ut[:,1:]])
    S['URSE (LR)']=lr(np.c_[Ud,Ed],np.c_[Ut,Et])
    S['Load context + E (LR)']=lr(np.c_[Cd,Ed],np.c_[Ct,Et])
    S['URSE + load context (LR)']=lr(np.c_[Ud,Ed,Cd],np.c_[Ut,Et,Ct])
    nbm=None
    hl=test['y'][:,j]>q95[j]; miss=hl&(yh<=q95[j]); nbm=miss.sum()
    for name,(sd,st) in S.items():
        acc=st>=thr70(sd); unsafe=(miss&acc).sum()
        rows.append(dict(horizon=h,gater=name,AUROC=roc_auc_score(ok,st),AURC_fail=aurc(st,fail),
            fail_at70=risk_at(st,fail,.7),MAE_at70=risk_at(st,e,.7),frozen_cov=acc.mean(),
            unsafe_miss_red=1-unsafe/nbm,breakeven=(nbm-unsafe)/(~acc).sum()))
    # within-load-strata AUROC
    qs=np.quantile(yh,[0,.2,.4,.6,.8,1]); b=np.clip(np.searchsorted(qs,yh,side='right')-1,0,4)
    def strat(v):
        a=[roc_auc_score(ok[b==s],v[b==s]) for s in range(5)]; w=[(b==s).sum() for s in range(5)]; return np.average(a,weights=w)
    strata_rows.append(dict(horizon=h,E=strat(-Et[:,0]),U=strat(-test['U'][:,j]),R=strat(-test['R'][:,j]),S=strat(-test['S']),
        E_relative=strat(-(Et[:,0]/(np.maximum(yh,0)+22.08))),spearman_E_yhat=spearmanr(Et[:,0],yh)[0]))
    # paired bootstrap: does E add to load context? does best E-model beat yhat rule?
    for a,bb in [('Load context + E (LR)','Load context (LR)'),('URSE + load context (LR)','Predicted load ŷ (rule)'),('E only (deep ensemble)','U only (MC-dropout)')]:
        sa,sb=S[a][1],S[bb][1]; d0=risk_at(sa,fail,.7)-risk_at(sb,fail,.7)
        bs=[risk_at(sa[ix],fail[ix],.7)-risk_at(sb[ix],fail[ix],.7) for ix in IDX]
        rows.append(dict(horizon=h,gater=f'Δfail@70: {a} − {bb}',fail_at70=d0,CI_lo=np.percentile(bs,2.5),CI_hi=np.percentile(bs,97.5)))
df=pd.DataFrame(rows); df.to_csv(os.path.join(OUT,'B1_ensemble_results.csv'),index=False)
st=pd.DataFrame(strata_rows); st.to_csv(os.path.join(OUT,'B1_within_strata.csv'),index=False)
pd.set_option('display.width',250); pd.set_option('display.max_columns',20)
main=df[~df.gater.str.startswith('Δ')]
print(main.pivot_table(index='gater',columns='horizon',values='AUROC').round(3).to_string())
print(main.pivot_table(index='gater',columns='horizon',values='fail_at70').round(3).to_string())
print(main.pivot_table(index='gater',columns='horizon',values='unsafe_miss_red').round(3).to_string())
print(df[df.gater.str.startswith('Δ')][['horizon','gater','fail_at70','CI_lo','CI_hi']].round(4).to_string(index=False))
print(st.round(3).to_string(index=False))
