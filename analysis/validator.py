from common import *
z=np.load(R+'revision_B2_fault_injection.npz'); SC=list(z['scenarios']); spike_pos=z['spike_pos']
tr=np.where(meta.development_role.values=='ModelTraining')[0]
L0,A0=XL[test['idx']].astype(np.float32),XA[test['idx']].astype(np.float32)
def fault(name):
    L,A=L0.copy(),A0.copy()
    if name=='dropout_2h': L[:,-8:]=0
    elif name=='flatline_4h': L[:,-16:]=L[:,[-17]]
    elif name=='stale_1h': L[:,-4:]=L[:,[-5]]
    elif name=='spike': L[np.arange(len(L)),spike_pos]+=100
    elif name=='half_scale': L*=0.5
    elif name=='sessions_lost': A[:]=0
    return L,A
def vfeats(L,A):
    l8,a8=L[:,-8:],A[:,-8:]
    d=np.abs(np.diff(L[:,-9:],axis=1))
    on=A[:,-16:]>0; pps=np.where(on,L[:,-16:]/np.maximum(A[:,-16:],1),np.nan)
    with np.errstate(all='ignore'): med_pps=np.nanmedian(pps,axis=1)
    return dict(
      power_wo_sessions=((L[:,-8:]>0.5)&(A[:,-8:]==0)).sum(1),          # steps with load but no sessions
      sessions_wo_power=np.where((l8==0).all(1),a8.mean(1),0.0),         # mean sessions during an all-zero 2h load run
      flat_run=np.array([ (np.diff(r[::-1])==0).argmin() if (np.diff(r[::-1])!=0).any() else 95 for r in L])*(L[:,-1]>0),  # identical trailing steps
      max_step=d.max(1),
      per_session_kw=med_pps)
# --- training-derived thresholds ---
Ft=vfeats(XL[tr],XA[tr])
thr=dict(power_wo_sessions=np.quantile(Ft['power_wo_sessions'],.999),
         sessions_wo_power=np.quantile(Ft['sessions_wo_power'],.999),
         flat_run=np.quantile(Ft['flat_run'],.999),
         max_step=np.quantile(Ft['max_step'],.999)
)
def flags(F):
    fl=dict(power_wo_sessions=F['power_wo_sessions']>thr['power_wo_sessions'],
            sessions_wo_power=F['sessions_wo_power']>thr['sessions_wo_power'],
            flat_run=F['flat_run']>thr['flat_run'],
            max_step=F['max_step']>thr['max_step'])

    fl['any']=np.any(np.stack(list(fl.values())),axis=0); return fl
if __name__=='__main__':
    print('thresholds',{k:round(float(v),3) for k,v in thr.items()})
    print('train flag rate',{k:round(v.mean(),4) for k,v in flags(Ft).items()})
    for sc in SC:
        L,A=fault(sc); fl=flags(vfeats(L,A))
        print(f'{sc:>14}',{k:round(v.mean(),3) for k,v in fl.items()})
