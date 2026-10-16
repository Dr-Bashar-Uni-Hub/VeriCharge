from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
FIG=os.path.join(OUT,'figures')+os.sep
b1=np.load(R+'revision_B1_ensemble_predictions.npz'); SE=list(b1['seeds'])
E={sp:np.stack([b1[f'{sp}_seed_{s}'] for s in SE]).std(0,ddof=1) for sp in ['dev','test']}
f=dev['fit']
# ---------- Fig 4: benchmark risk-coverage (failure rate) ----------
cols={'URS (VeriCharge)':'#c0392b','U only (MC-dropout)':'#e67e22','E only (deep ensemble)':'#8e44ad',
      'Predicted load ŷ (rule)':'#1f4e79','Load context (LR)':'#2e86c1','URSE + load context (LR)':'#27ae60','Oracle':'#7f8c8d'}
fig,ax=plt.subplots(1,3,figsize=(11,3.8),sharey=False)
covs=np.linspace(0.2,1,81)
for j,h in enumerate(H):
    lab=(np.abs(dev['y'][:,j]-dev['yhat'][:,j])<=eps[j]).astype(int)
    S=build_scores(j,lab)
    def lr(Xd,Xt):
        sc=StandardScaler().fit(Xd[f]); m=LogisticRegression(max_iter=1000).fit(sc.transform(Xd[f]),lab[f]); return m.predict_proba(sc.transform(Xt))[:,1]
    Cd=ctx(dev['idx'],dev['yhat'][:,j]); Ct=ctx(test['idx'],test['yhat'][:,j])
    S2={'URS (VeriCharge)':S['URS (VeriCharge)'][1],'U only (MC-dropout)':S['U only (MC-dropout)'][1],
        'E only (deep ensemble)':lr(E['dev'][:,[j]],E['test'][:,[j]]),
        'Predicted load ŷ (rule)':S['Predicted load ŷ (rule)'][1],'Load context (LR)':S['Load context (LR)'][1],
        'URSE + load context (LR)':lr(np.c_[dev['X'][j],E['dev'][:,[j]],Cd],np.c_[test['U'][:,j],test['R'][:,j],test['S'],E['test'][:,[j]],Ct])}
    e=np.abs(test['y'][:,j]-test['yhat'][:,j]); fail=(e>eps[j]).astype(float); S2['Oracle']=-e
    for k,s in S2.items():
        r=[risk_at(s,fail,c)*100 for c in covs]
        ax[j].plot(covs*100,r,color=cols[k],lw=2.2 if k in('URS (VeriCharge)','Predicted load ŷ (rule)') else 1.4,ls='--' if k=='Oracle' else '-',label=k)
    ax[j].axvline(70,color='grey',lw=0.8,ls=':'); ax[j].set_title(f'({"abc"[j]}) {h[:-3]}-min horizon'); ax[j].set_xlabel('Coverage (%)')
    ax[j].set_ylim(0,55); ax[j].invert_xaxis()
ax[0].set_ylabel('Failure rate among accepted forecasts (%)')
h_,l_=ax[0].get_legend_handles_labels(); fig.legend(h_,[x.replace('ŷ','$\\hat{y}$') for x in l_],loc='lower center',ncol=4,frameon=False,fontsize=8)
plt.tight_layout(rect=(0,0.13,1,1)); plt.savefig(FIG+'fig4_benchmark.png',dpi=300); plt.close()
# ---------- Fig 5: load stratification ----------
fig,ax=plt.subplots(1,2,figsize=(10,3.4))
st=pd.read_csv(os.path.join(OUT,'B1_within_strata.csv'))
for j,h in enumerate(H):
    yh=test['yhat'][:,j]; ok=(np.abs(test['y'][:,j]-yh)<=eps[j])
    qs=np.quantile(yh,[0,.2,.4,.6,.8,1]); b=np.clip(np.searchsorted(qs,yh,side='right')-1,0,4)
    ax[0].plot(range(1,6),[ok[b==s].mean()*100 for s in range(5)],'o-',label=f'{h[:-3]} min')
ax[0].set_xticks(range(1,6)); ax[0].set_xlabel('Predicted-load quintile (1 = lowest)'); ax[0].set_ylabel('Acceptable forecasts (%)')
ax[0].set_title('(a) Acceptability falls with predicted load'); ax[0].legend(frameon=False)
w=0.2; sigs=['U','R','S','E']; names={'U':'U (MC-dropout)','R':'R (perturbation)','S':'S (shift)','E':'E (ensemble)'}
for i,s in enumerate(sigs): ax[1].bar(np.arange(3)+(i-1.5)*w,st[s],w,label=names[s])
ax[1].axhline(0.5,color='k',lw=0.8,ls=':'); ax[1].set_xticks(range(3)); ax[1].set_xticklabels(['15 min','30 min','60 min'])
ax[1].set_ylim(0.45,0.62); ax[1].set_ylabel('AUROC within load quintiles'); ax[1].set_title('(b) Signal discrimination at fixed load level'); ax[1].legend(frameon=False,fontsize=8,ncol=2)
plt.tight_layout(); plt.savefig(FIG+'fig5_strata.png',dpi=300); plt.close()
# ---------- Fig 6: fault-detection heatmap (30 min) ----------
b2=pd.read_csv(os.path.join(OUT,'B2_fault_results.csv')); d=b2[(b2.horizon=='30min')&(b2.scenario!='clean')]
order_g=['Predicted load ŷ (rule)','Load context','U only','E only','URS (VeriCharge, frozen)','URSE + context']
order_s=['dropout_2h','half_scale','flatline_4h','stale_1h','sessions_lost','spike']
M=d.pivot_table(index='gater',columns='scenario',values='fault_detect_AUROC').loc[order_g,order_s]
fig,ax=plt.subplots(figsize=(7.5,3.4)); im=ax.imshow(M.values,cmap='RdBu',vmin=0,vmax=1,aspect='auto')
ax.set_xticks(range(6)); ax.set_xticklabels(['2-h dropout','Half scale','4-h flatline','1-h stale','Sessions lost','Spike'],rotation=0,fontsize=8)
ax.set_yticks(range(6)); ax.set_yticklabels(['Predicted load $\\hat{y}$ (rule)','Load context (LR)','U only (MC-dropout)','E only (ensemble)','URS (VeriCharge)','URSE + context'],fontsize=8)
for a in range(6):
    for c in range(6): ax.text(c,a,f'{M.values[a,c]:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(M.values[a,c]-.5)>.3 else 'black')
cb=plt.colorbar(im,ax=ax,fraction=0.03); cb.set_label('Fault-detection AUROC',fontsize=8)
for s in ax.spines.values(): s.set_visible(False)
plt.tight_layout(); plt.savefig(FIG+'fig6_faults.png',dpi=300); plt.close()
# ---------- Fig 7: validator ----------
v=pd.read_csv(os.path.join(OUT,'V_validator_results.csv')); v=v[v.horizon=='30min']
sc=['clean','dropout_2h','flatline_4h','stale_1h','spike','sessions_lost','half_scale']
lbl=['Clean','2-h dropout','4-h flatline','1-h stale','Spike','Sessions lost','Half scale']
P=[('URS (VeriCharge, frozen)','URS (VeriCharge)','#c0392b'),('Validator + URS (VeriCharge, frozen)','Validator + URS','#f1948a'),
   ('Predicted load ŷ (rule)','$\\hat{y}$ rule','#1f4e79'),('Validator + Predicted load ŷ (rule)','Validator + $\\hat{y}$ rule','#85c1e9')]
fig,ax=plt.subplots(figsize=(10,3.3)); w=0.2
for i,(p,l,c) in enumerate(P):
    vals=[v[(v.policy==p)&(v.scenario==s)].harmful_accept.values[0]*100 for s in sc]
    ax.bar(np.arange(len(sc))+(i-1.5)*w,vals,w,label=l,color=c)
ax.set_xticks(range(len(sc))); ax.set_xticklabels(lbl); ax.set_ylabel('Harmful automatic acceptance (%)')
ax.legend(frameon=False,ncol=4,fontsize=8,loc='upper right'); plt.tight_layout(); plt.savefig(FIG+'fig7_validator.png',dpi=300); plt.close()
print('ok')
