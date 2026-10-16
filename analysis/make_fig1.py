import os
OUT=os.environ.get('VERICHARGE_OUTPUTS','outputs'); os.makedirs(os.path.join(OUT,'figures'),exist_ok=True)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'DejaVu Sans'})
fig,ax=plt.subplots(figsize=(11,4.2)); ax.set_xlim(0,110); ax.set_ylim(0,42); ax.axis('off')
def box(x,y,w,h,title,lines,fc,ec='#34495e'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.4,rounding_size=1.2',fc=fc,ec=ec,lw=1.2))
    ax.text(x+w/2,y+h-2.6,title,ha='center',va='top',fontsize=9.5,weight='bold')
    for i,l in enumerate(lines): ax.text(x+w/2,y+h-7-i*3.1,l,ha='center',va='top',fontsize=8)
def arr(x1,y1,x2,y2): ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',color='#34495e',lw=1.3))
box(1,17,15,23,'Charging telemetry',['ACN-Data, Caltech','~4-s session records','→ 15-min site load','+ active sessions'],'#eaf2f8')
box(20,17,15,23,'Telemetry validator',['training-calibrated','physical-consistency','checks (zero runs,','frozen values, steps,','session mismatch)'],'#fdedec')
box(39,17,15,23,'Frozen forecaster',['multi-horizon TCN','(seed 42; 5-seed','ensemble for E)','15 / 30 / 60 min'],'#f4ecf7')
box(58,17,22,23,'Reliability evidence',['Load context: $\\hat{y}$, last load,','hour, weekday, sessions','Model-internal: U (MC-dropout),','R (perturbation), S (shift),','E (deep-ensemble disagreement)'],'#fef9e7')
box(84,17,24,23,'Selective gate',['trust or rule score ≥ τ','(τ frozen on TrustCal)','ACCEPT → automatic use','ABSTAIN → fallback /','reserve / operator'],'#e9f7ef')
for a,b in [(16.8,28.5),(35.8,28.5),(54.8,28.5),(80.8,28.5)]: arr(a,b,a+3,b)
ax.add_patch(FancyBboxPatch((1,1.5),107,13,boxstyle='round,pad=0.4,rounding_size=1.2',fc='#f8f9f9',ec='#7f8c8d',lw=1))
ax.text(54.5,12.2,'Leakage-safe evaluation on untouched FinalTest (2020)',ha='center',fontsize=9.5,weight='bold')
items=['Risk–coverage &\nmatched-coverage benchmark','Load-stratified\ndiscrimination','Tolerance\nsensitivity','Telemetry-fault\ninjection','Grid high-load triage\n& break-even cost']
for i,t in enumerate(items): ax.text(11+i*21.5,6.2,t,ha='center',va='center',fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT,'figures','fig1_overview.png'),dpi=300)
