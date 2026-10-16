"""Shared loaders and gating utilities for the VeriCharge revision analyses.
Set VERICHARGE_ARTIFACTS to the folder holding data/forecasting and results (default: ./artifacts).
Outputs go to VERICHARGE_OUTPUTS (default: ./outputs)."""
import numpy as np, pandas as pd, joblib, json
import os
A=os.environ.get('VERICHARGE_ARTIFACTS','artifacts')
OUT=os.environ.get('VERICHARGE_OUTPUTS','outputs'); os.makedirs(OUT,exist_ok=True); os.makedirs(os.path.join(OUT,'figures'),exist_ok=True)
R=os.path.join(A,'results')+os.sep
H=['15min','30min','60min']
meta=pd.read_csv(os.path.join(A,'data/forecasting/caltech_forecasting_metadata_roles.csv.gz'))
ds=np.load(os.path.join(A,'data/forecasting/caltech_forecasting_15m.npz'))
pol=json.load(open(R+'vericharge_frozen_deployment_policy.json'))
eps=np.array([pol['forecast_acceptability_tolerances_kW'][h] for h in H])
tau=np.array([pol['trust_cutoffs'][h] for h in H])
# trust dev
lab=np.load(R+'trustcalibration_failure_labels.npz')
sp=np.load(R+'trustcalibration_internal_split.npz')
tf=np.load(R+'trust_features_trustcalibration.npz',allow_pickle=True)
dev=dict(idx=lab['sample_indices'],y=lab['y_true'],yhat=lab['y_pred'],fit=sp['trustfit_mask'],cal=sp['trustcal_mask'],
  X=[tf['X_%s'%h] for h in H],fn=list(tf['feature_names']))
# MV
mvf=np.load(R+'trust_features_modelvalidation.npz',allow_pickle=True)
mvp=np.load(R+'tcn_five_seed_modelvalidation_predictions.npz')
# test
fu=np.load(R+'finaltest_forecast_uncertainty.npz'); fr=np.load(R+'finaltest_robustness_shift.npz'); ft=np.load(R+'finaltest_trust_predictions.npz')
test=dict(idx=fu['sample_indices'],y=fu['y_true'].astype(float),yhat=fu['deterministic'].astype(float),
  U=ft['U'].astype(float),R=ft['R'].astype(float),S=ft['S'],T=np.c_[ft['trust_15min'],ft['trust_30min'],ft['trust_60min']])
models=[joblib.load(R+f'trust_models/logistic_trust_{h}.joblib') for h in H]
scalers=[joblib.load(R+f'trust_models/trust_scaler_{h}.joblib') for h in H]

import warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
XL=ds['X_load']; XA=ds['X_active']
grid=pd.read_csv(R+'grid_operating_thresholds_training.csv')
q95=grid[grid.threshold_label=='P95'].threshold_kw.values

def ctx(idx,yhat_j):
    m=meta.iloc[idx]; hr=m.hour_local.values+ pd.to_datetime(m.origin_timestamp_local.str[:19]).dt.minute.values/60
    return np.c_[yhat_j, XL[idx,-1], np.sin(2*np.pi*hr/24), np.cos(2*np.pi*hr/24), m.is_weekend.values, XA[idx,-1]]
def hourdum(idx):
    m=meta.iloc[idx]; D=np.zeros((len(idx),25)); D[np.arange(len(idx)),m.hour_local.values]=1; D[:,24]=m.is_weekend.values; return D

def aurc(score,loss):
    o=np.argsort(-score,kind='stable'); c=np.cumsum(loss[o])/np.arange(1,len(loss)+1); return c.mean()
def aurc_oracle(loss): return aurc(-loss,loss)
def risk_at(score,loss,cov):
    k=int(round(cov*len(loss))); o=np.argsort(-score,kind='stable'); return loss[o[:k]].mean()

def build_scores(j, label_dev, label_name='base'):
    """fit all gaters on TrustFit with given acceptable-label, return dict name->(dev_score,test_score)"""
    f,c=dev['fit'],dev['cal']
    out={}
    Udev=np.c_[dev['X'][j]]; Utest=np.c_[test['U'][:,j],test['R'][:,j],test['S']]
    Cdev=ctx(dev['idx'],dev['yhat'][:,j]); Ctest=ctx(test['idx'],test['yhat'][:,j])
    def lr(Xd,Xt):
        sc=StandardScaler().fit(Xd[f]); m=LogisticRegression(C=1.0,max_iter=1000).fit(sc.transform(Xd[f]),label_dev[f])
        return m.predict_proba(sc.transform(Xd))[:,1], m.predict_proba(sc.transform(Xt))[:,1]
    out['URS (VeriCharge)']=lr(Udev,Utest)
    out['U only (MC-dropout)']=lr(Udev[:,[0]],Utest[:,[0]])
    out['Predicted load ŷ (rule)']=(-dev['yhat'][:,j],-test['yhat'][:,j])
    out['Time-of-day/weekday (LR)']=lr(hourdum(dev['idx']),hourdum(test['idx']))
    out['Load context (LR)']=lr(Cdev,Ctest)
    # heteroscedastic error regressor (nonlinear), fit |e| on TrustFit context
    edev=np.abs(dev['y'][:,j]-dev['yhat'][:,j])
    g=HistGradientBoostingRegressor(max_iter=200,learning_rate=0.05,max_depth=3,random_state=0).fit(Cdev[f],edev[f])
    out['Hetero. error model (GBM)']=(-g.predict(Cdev),-g.predict(Ctest))
    out['URS + ŷ (LR)']=lr(np.c_[Udev,Cdev[:,[0]]],np.c_[Utest,Ctest[:,[0]]])
    out['URS + load context (LR)']=lr(np.c_[Udev,Cdev],np.c_[Utest,Ctest])
    return out

def thr70(sdev): return np.quantile(sdev[dev['cal']],0.30)
