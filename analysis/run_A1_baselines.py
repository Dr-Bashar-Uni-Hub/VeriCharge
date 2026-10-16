from common import *
rows=[];curves={}
for j,h in enumerate(H):
    lab_dev=(np.abs(dev['y'][:,j]-dev['yhat'][:,j])<=eps[j]).astype(int)
    S=build_scores(j,lab_dev)
    e=np.abs(test['y'][:,j]-test['yhat'][:,j]); fail=(e>eps[j]).astype(float)
    hl=test['y'][:,j]>q95[j]; miss=hl&(test['yhat'][:,j]<=q95[j]); nb=miss.sum()
    a_or_mae,a_or_f=aurc_oracle(e),aurc_oracle(fail)
    rnd_mae=e.mean(); rnd_f=fail.mean()
    for name,(sd,st) in S.items():
        t=thr70(sd); acc=st>=t
        unsafe=(miss&acc).sum(); nab=(~acc).sum()
        A=aurc(st,e); Af=aurc(st,fail)
        rows.append(dict(horizon=h,gater=name,
          AUROC=roc_auc_score(1-fail,st),
          AURC_MAE=A, EAURC_MAE=A-a_or_mae, oracle_gain_captured=(rnd_mae-A)/(rnd_mae-a_or_mae),
          AURC_fail=Af, MAE_at70=risk_at(st,e,0.70), fail_at70=risk_at(st,fail,0.70),
          frozen_cov=acc.mean(), frozen_accMAE=e[acc].mean(), frozen_accFail=fail[acc].mean(),
          unsafe_miss_red=1-unsafe/nb, breakeven=(nb-unsafe)/nab))
    rows.append(dict(horizon=h,gater='Oracle',AURC_MAE=a_or_mae,EAURC_MAE=0,oracle_gain_captured=1,AURC_fail=a_or_f))
    rows.append(dict(horizon=h,gater='Random / no selection',AURC_MAE=rnd_mae,oracle_gain_captured=0,AURC_fail=rnd_f,AUROC=0.5,MAE_at70=rnd_mae,fail_at70=rnd_f))
df=pd.DataFrame(rows); df.to_csv(os.path.join(OUT,'A1_baselines.csv'),index=False)
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(df.round(4).to_string(index=False))
