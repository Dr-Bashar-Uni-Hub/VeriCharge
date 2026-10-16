from validator import *
b1=np.load(R+'revision_B1_ensemble_predictions.npz'); SE=list(b1['seeds'])
Edev=np.stack([b1[f'dev_seed_{s}'] for s in SE]).std(0,ddof=1)
m=meta.iloc[test['idx']]; hr=m.hour_local.values+pd.to_datetime(m.origin_timestamp_local.str[:19]).dt.minute.values/60
def ctx_t(yh,F): return np.c_[yh,F[:,0],np.sin(2*np.pi*hr/24),np.cos(2*np.pi*hr/24),m.is_weekend.values,F[:,8]]
f=dev['fit']; FL={sc:flags(vfeats(*fault(sc)))['any'] for sc in SC}
rows=[]
for j,h in enumerate(H):
    lab=(np.abs(dev['y'][:,j]-dev['yhat'][:,j])<=eps[j]).astype(int)
    Ud=dev['X'][j]; Cd=ctx(dev['idx'],dev['yhat'][:,j]); Ed=Edev[:,[j]]
    X=np.c_[Ud,Ed,Cd]; sc_=StandardScaler().fit(X[f]); mm=LogisticRegression(max_iter=1000).fit(sc_.transform(X[f]),lab[f])
    t_urse=np.quantile(mm.predict_proba(sc_.transform(X))[dev['cal'],1],.3); t_y=np.quantile(-dev['yhat'][dev['cal'],j],.3)
    for sc in SC:
        yh=z[f'{sc}_yhat'][:,j].astype(float); U=z[f'{sc}_U'][:,j]; Rr=z[f'{sc}_R'][:,j]; S=z[f'{sc}_S']; F=z[f'{sc}_shiftfeat']
        E=z[f'{sc}_ens'][:,:,j].std(0,ddof=1); e=np.abs(test['y'][:,j]-yh); bad=e>eps[j]; v=FL[sc]
        acc={'URS (VeriCharge, frozen)':models[j].predict_proba(scalers[j].transform(np.c_[U,Rr,S]))[:,1]>=tau[j],
             'Predicted load ŷ (rule)':-yh>=t_y,
             'URSE + context':mm.predict_proba(sc_.transform(np.c_[U,Rr,S,E,ctx_t(yh,F)]))[:,1]>=t_urse}
        for g in list(acc): acc['Validator + '+g]=acc[g]&~v
        for g,a in acc.items():
            rows.append(dict(horizon=h,scenario=sc,policy=g,accept_rate=a.mean(),harmful_accept=(a&bad).mean(),
                fail_all=bad.mean(),accepted_MAE=e[a].mean(),accepted_fail=bad[a].mean(),validator_flag_rate=v.mean(),
                harmful_flagged_by_validator=(v&bad).sum()/max(bad.sum(),1)))
df=pd.DataFrame(rows); df.to_csv(os.path.join(OUT,'V_validator_results.csv'),index=False)
pd.set_option('display.width',250)
for v in ['accept_rate','harmful_accept','accepted_MAE']:
    print('\n== 30min',v); print(df[df.horizon=='30min'].pivot_table(index='policy',columns='scenario',values=v).round(3)[SC].to_string())
print('\nvalidator recall on harmful corrupted forecasts (30min):'); print(df[(df.horizon=='30min')&(df.policy=='Predicted load ŷ (rule)')].set_index('scenario')[['validator_flag_rate','harmful_flagged_by_validator','fail_all']].round(3))
print('\nclean, all horizons:'); print(df[df.scenario=='clean'].pivot_table(index='policy',columns='horizon',values=['accept_rate','accepted_fail']).round(3))
