from common import *
L0=22.0802  # ModelTraining mean load (kW)
mv_e=np.abs(mvp['seed_42']-mvp['y_true']); mv_yh=mvp['seed_42']
rows=[]
variants=['abs_P50 (current)','abs_P60','abs_P75','normalized_P50']
for j,h in enumerate(H):
  for v in variants:
    if v.startswith('abs'):
      q={'abs_P50 (current)':.5,'abs_P60':.6,'abs_P75':.75}[v]
      tol=np.quantile(mv_e[:,j],q)
      ldev=np.abs(dev['y'][:,j]-dev['yhat'][:,j]); ltest=np.abs(test['y'][:,j]-test['yhat'][:,j])
    else:
      nz=lambda e,yh: e/(np.maximum(yh,0)+L0)
      tol=np.quantile(nz(mv_e[:,j],mv_yh[:,j]),.5)
      ldev=nz(np.abs(dev['y'][:,j]-dev['yhat'][:,j]),dev['yhat'][:,j]); ltest=nz(np.abs(test['y'][:,j]-test['yhat'][:,j]),test['yhat'][:,j])
    lab=(ldev<=tol).astype(int); fail=(ltest>tol).astype(float)
    S=build_scores(j,lab)
    if v.startswith('normalized'):  # hetero GBM was fit on abs error; refit on normalized loss
      f=dev['f'] if 'f' in dev else dev['fit']
      Cd=ctx(dev['idx'],dev['yhat'][:,j]); Ct=ctx(test['idx'],test['yhat'][:,j])
      g=HistGradientBoostingRegressor(max_iter=200,learning_rate=0.05,max_depth=3,random_state=0).fit(Cd[f],ldev[f])
      S['Hetero. error model (GBM)']=(-g.predict(Cd),-g.predict(Ct))
    for name,(sd,st) in S.items():
      acc=st>=thr70(sd)
      rows.append(dict(horizon=h,label=v,tol=tol,test_fail_prev=fail.mean(),gater=name,AUROC=roc_auc_score(1-fail,st),
        AURC_fail=aurc(st,fail),oracle_gain_fail=(fail.mean()-aurc(st,fail))/(fail.mean()-aurc_oracle(fail)),
        frozen_cov=acc.mean(),frozen_fail_red=1-fail[acc].mean()/fail.mean()))
df=pd.DataFrame(rows); df.to_csv(os.path.join(OUT,'A2_tolerance.csv'),index=False)
pd.set_option('display.width',250)
keep=['URS (VeriCharge)','U only (MC-dropout)','Predicted load ŷ (rule)','Load context (LR)','Hetero. error model (GBM)','URS + load context (LR)']
print(df[df.gater.isin(keep)].pivot_table(index=['label','gater'],columns='horizon',values='AUROC').round(3).to_string())
print(df[df.gater.isin(keep)].pivot_table(index=['label','gater'],columns='horizon',values='frozen_fail_red').round(3).to_string())
print(df.groupby(['label','horizon']).tol.first().round(4))
