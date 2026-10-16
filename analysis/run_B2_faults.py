from common import *
z=np.load(R+'revision_B2_fault_injection.npz'); SC=list(z['scenarios'])
b1=np.load(R+'revision_B1_ensemble_predictions.npz'); SE=list(b1['seeds'])
Edev=np.stack([b1[f'dev_seed_{s}'] for s in SE]).std(0,ddof=1)
m=meta.iloc[test['idx']]; hr=m.hour_local.values+pd.to_datetime(m.origin_timestamp_local.str[:19]).dt.minute.values/60
def ctx_t(yh,F): return np.c_[yh,F[:,0],np.sin(2*np.pi*hr/24),np.cos(2*np.pi*hr/24),m.is_weekend.values,F[:,8]]
f=dev['fit']; ytrue=test['y']
rows=[]
for j,h in enumerate(H):
    lab=(np.abs(dev['y'][:,j]-dev['yhat'][:,j])<=eps[j]).astype(int)
    Ud=dev['X'][j]; Cd=ctx(dev['idx'],dev['yhat'][:,j]); Ed=Edev[:,[j]]
    fitted={}
    def fit(name,Xd):
        sc=StandardScaler().fit(Xd[f]); mm=LogisticRegression(max_iter=1000).fit(sc.transform(Xd[f]),lab[f])
        sd=mm.predict_proba(sc.transform(Xd))[:,1]; fitted[name]=(sc,mm,np.quantile(sd[dev['cal']],.3))
    fit('U only',Ud[:,[0]]); fit('E only',Ed); fit('Load context',Cd); fit('URSE + context',np.c_[Ud,Ed,Cd])
    thr_yhat=np.quantile(-dev['yhat'][dev['cal'],j],.3)
    def scores(sc_):
        yh=z[f'{sc_}_yhat'][:,j].astype(float); U=z[f'{sc_}_U'][:,j]; Rr=z[f'{sc_}_R'][:,j]; S=z[f'{sc_}_S']; F=z[f'{sc_}_shiftfeat']
        E=z[f'{sc_}_ens'][:,:,j].std(0,ddof=1)
        out={}
        out['URS (VeriCharge, frozen)']=(models[j].predict_proba(scalers[j].transform(np.c_[U,Rr,S]))[:,1],tau[j])
        out['Predicted load ŷ (rule)']=(-yh,thr_yhat)
        feats={'U only':U[:,None],'E only':E[:,None],'Load context':ctx_t(yh,F),'URSE + context':np.c_[U,Rr,S,E,ctx_t(yh,F)]}
        for k,(scl,mm,t) in fitted.items(): out[k]=(mm.predict_proba(scl.transform(feats[k]))[:,1],t)
        return out,yh
    clean,_=scores('clean')
    for sc_ in SC:
        S_,yh=scores(sc_); e=np.abs(ytrue[:,j]-yh); bad=e>eps[j]
        for g,(s,t) in S_.items():
            acc=s>=t
            r=dict(horizon=h,scenario=sc_,gater=g,MAE_all=e.mean(),fail_all=bad.mean(),
                   accept_rate=acc.mean(),harmful_accept=(acc&bad).mean(),accepted_MAE=e[acc].mean() if acc.any() else np.nan)
            if sc_!='clean':  # can the score separate corrupted from clean copies of the same origins?
                cs=clean[g][0]; r['fault_detect_AUROC']=roc_auc_score(np.r_[np.ones(len(cs)),np.zeros(len(s))],np.r_[cs,s])
            rows.append(r)
df=pd.DataFrame(rows); df.to_csv(os.path.join(OUT,'B2_fault_results.csv'),index=False)
pd.set_option('display.width',250); pd.set_option('display.max_columns',20)
d=df[df.horizon=='30min']
for v in ['accept_rate','harmful_accept','accepted_MAE','fault_detect_AUROC']:
    print('\n== 30min',v); print(d.pivot_table(index='gater',columns='scenario',values=v).round(3)[SC if v!='fault_detect_AUROC' else SC[1:]].to_string())
print(df[df.gater=='URS (VeriCharge, frozen)'].pivot_table(index='scenario',columns='horizon',values='fail_all').round(3).to_string())
