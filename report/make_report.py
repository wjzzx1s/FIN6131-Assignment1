"""Deterministic LaTeX tables and TikZ/PGFPlots from notebook outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'results'
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
s = json.loads((RESULTS/'summary.json').read_text())
a = pd.read_csv(RESULTS/'assets.csv').set_index('ticker')
p = pd.read_csv(RESULTS/'full_portfolios.csv').set_index('strategy')
w = pd.read_csv(RESULTS/'full_weights.csv').set_index('strategy')
perf = pd.read_csv(RESULTS/'performance.csv')
sr = pd.read_csv(RESULTS/'sharpe_comparison.csv').set_index('strategy')
tickers = list(a.index)
names = ['MVP unrestricted','MVP long-only','Tangency unrestricted','Tangency long-only']
labels = {'MVP unrestricted':'MVP U','MVP long-only':'MVP L','Tangency unrestricted':'Tangency U','Tangency long-only':'Tangency L','Equal weight 1/N':'Equal weight','Bounded Sharpe diagnostic +/-3':'Bounded diagnostic'}
colors = ['blue!70!black','cyan!70!black','orange!90!black','red!75!black','black']
def put(name, text):
    (HERE/name).write_text(text+'\n',encoding='utf-8')
def pct(v): return f'{100*v:.2f}\\%'
def macro(k,v): return '\\newcommand{\\'+k+'}{'+str(v)+'}'
m = s['data']; h=s['holdout']; d=s['full_sample']['diagnostic']
nums = {'StartDate':m['first_date'],'EndDate':m['last_date'],'PriceDays':m['price_rows'],'ReturnDays':m['return_rows'],'SplitDate':h['split_base_date'],'InDays':h['n_is'],'OutDays':h['n_oos'],'FullDen':f"{d['D']:.3f}"}
fu=p.loc['Tangency unrestricted']; ml=p.loc['MVP long-only']; tl=p.loc['Tangency long-only']
nums['FullInterpretation'] = 'The unrestricted tangency has '+pct(fu.expected_return_log)+' fitted log-proxy return, '+pct(fu.volatility_log)+' volatility and gross exposure '+f'{fu.gross_exposure:.2f}'+' times equity. Its largest position is '+pct(w.loc['Tangency unrestricted','SPY'])+' in SPY. These weights require substantial shorting and are not an investable recommendation. The long-only tangency instead combines '+pct(w.loc['Tangency long-only','SPY'])+' SPY and '+pct(w.loc['Tangency long-only','GLD'])+' GLD, with proxy Sharpe '+f'{tl.sharpe_log_model:.3f}'+'. The long-only MVP has volatility '+pct(ml.volatility_log)+', versus '+pct(p.loc['MVP unrestricted','volatility_log'])+' unrestricted.'
train=h['training_diagnostic']
nums['ISDiagnostic'] = 'On the training sample, $d='+f"{train['D']:.3f}"+'<0$. Thus there is no finite fully invested unrestricted Sharpe maximum. Its supremum is $\\sqrt{C-B^2/A}='+f"{train['supremum']:.3f}"+'$, approached only with unbounded leverage; the normalized closed form is a Sharpe minimum, not a tangency maximum. The extra bounded diagnostic imposes $|w_i|\\le3$ and solves a different problem. It is labeled separately and does not replace the unavailable unrestricted optimum.'
o=perf[perf.period=='OOS'].set_index('strategy')
lo=o.loc['Tangency long-only']; eq=o.loc['Equal weight 1/N']; bd=o.loc['Bounded Sharpe diagnostic +/-3']
nums['OOSInterpretation'] = 'The long-only tangency earns Sharpe '+f'{lo.sharpe_simple:.3f}'+' versus '+f'{eq.sharpe_simple:.3f}'+' for equal weight, with total gains of '+pct(lo.total_return)+' and '+pct(eq.total_return)+', respectively. Its maximum drawdown is '+pct(lo.max_drawdown)+' versus '+pct(eq.max_drawdown)+'. The bounded diagnostic earns '+f'{bd.sharpe_simple:.3f}'+' Sharpe but draws down '+pct(bd.max_drawdown)+'. Both MVPs reduce volatility but underperform the benchmark in Sharpe. Higher realized return therefore does not imply safer implementation.'
put('numbers.tex','\n'.join(macro(k,v) for k,v in nums.items()))

# Row separator must be two literal backslashes and a newline.
sep=' \\\\\n'
put('table_assets.tex',sep.join(f'{t} & {100*r.mean_annual_log:.2f} & {100*r.vol_annual_log:.2f} & {100*r.se_mean_annual:.2f}' for t,r in a.iterrows()))
put('table_portfolios.tex',sep.join(f'{labels[n]} & {100*p.loc[n,"expected_return_log"]:.2f} & {100*p.loc[n,"volatility_log"]:.2f} & {p.loc[n,"sharpe_log_model"]:.3f} & {p.loc[n,"gross_exposure"]:.2f}' for n in names))
put('table_oos.tex',sep.join(f'{labels[n]} & {sr.loc[n,"IS"]:.3f} & {sr.loc[n,"OOS"]:.3f} & {100*o.loc[n,"cagr"]:.2f} & {100*o.loc[n,"volatility_simple"]:.2f} & {100*o.loc[n,"max_drawdown"]:.2f}' for n in sr.index))
def coords(x,y):
    return ' '.join(f'({float(xx):.6f},{float(yy):.6f})' for xx,yy in zip(x,y))
def plot(x,y,opt,legend=None):
    r=r'\addplot['+opt+'] coordinates {'+coords(x,y)+'};\n'
    if legend: r+=r'\addlegendentry{'+legend+'}\n'
    return r
style=r'width=15cm,height=5.1cm,tick label style={font=\scriptsize},label style={font=\small},legend style={font=\scriptsize,draw=none},scaled ticks=false,grid=major,grid style={gray!15},'
def axis(body,opts):
    return r'\begin{tikzpicture}\begin{axis}['+style+opts+']\n'+body+r'\end{axis}\end{tikzpicture}'
# Correlations: every colored cell and value is drawn in TikZ.
c=pd.read_csv(RESULTS/'correlation.csv').set_index('ticker')
heat=[r'\begin{tikzpicture}[x=0.40cm,y=0.40cm,every node/.style={font=\fontsize{6}{7}\selectfont}]']
for i,t in enumerate(tickers):
    heat.append(f'\\node[anchor=east] at (-.2,{-i-.5}) {{{t}}};')
    heat.append(f'\\node[rotate=45,anchor=west] at ({i+.5},.15) {{{t}}};')
    for j,u in enumerate(tickers):
        v=c.loc[t,u]; color=f'red!{int(abs(v)*75)}!white' if v>=0 else f'blue!{int(abs(v)*75)}!white'
        heat.append(f'\\fill[{color}] ({j},{-i}) rectangle ++(1,-1);')
        txt=f'{v:.2f}'.replace('0.','.').replace('-0.','-.')
        heat.append(f'\\node at ({j+.5},{-i-.5}) {{{txt}}};')
heat.append(r'\end{tikzpicture}')
put('figures/correlation.tex','\n'.join(heat))
f=pd.read_csv(RESULTS/'frontier_unrestricted.csv'); l=pd.read_csv(RESULTS/'frontier_long_only.csv')
body=plot(f.volatility_log*100,f.expected_return_log*100,'blue!70!black,thick','Unrestricted')+plot(l.volatility_log*100,l.expected_return_log*100,'red!75!black,thick','Long-only')
cal=pd.read_csv(RESULTS/'cal.csv')
for n,color in [('Tangency unrestricted','blue!70!black'),('Tangency long-only','red!75!black')]:
    z=cal[cal.strategy==n]; body+=plot(z.volatility_log*100,z.expected_return_log*100,color+',dashed','CAL '+('U' if 'unrestricted' in n else 'L'))
body+=plot(a.vol_annual_log*100,a.mean_annual_log*100,'only marks,mark=*,mark size=1.3pt,gray','ETFs')
for i,t in enumerate(a.sort_values('mean_annual_log').index):
    rr=a.loc[t]; yy=.07+i*.055
    body+=f'\\draw[gray!45,thin] (axis cs:{100*rr.vol_annual_log:.6f},{100*rr.mean_annual_log:.6f}) -- (axis description cs:.42,{yy});\n'
    body+=f'\\node[font=\\fontsize{{5.5}}{{6}}\\selectfont,anchor=west,xshift=2pt] at (axis description cs:.42,{yy}) {{{t}}};\n'
for n in names:
    r=p.loc[n]; body+=plot([r.volatility_log*100],[r.expected_return_log*100],('blue!70!black' if 'unrestricted' in n else 'red!75!black')+',only marks,mark='+('star' if 'Tangency' in n else 'square*')+',mark size=2.6pt',labels[n])
put('figures/frontier.tex',axis(body,'xlabel={Annual volatility (\\%)},ylabel={Annual log-proxy return (\\%)},legend columns=4,legend pos=north west,height=6cm'))
body=''
for i,n in enumerate(names):
    body+=plot(np.arange(10)+i*.18-.27,w.loc[n].to_numpy()*100,'ybar,area legend,bar shift=0pt,bar width=2pt,fill='+colors[i]+',draw='+colors[i],labels[n])
put('figures/weights.tex',axis(body,'height=4.6cm,xmin=-.7,xmax=9.7,xtick={0,1,2,3,4,5,6,7,8,9},xticklabels={'+','.join(tickers)+'},ylabel={Weight (\\%)},legend columns=4,legend pos=north west'))
wealth=pd.read_csv(RESULTS/'oos_wealth.csv'); body=''
for i,n in enumerate(wealth.columns[1:]):
    body+=plot(np.arange(len(wealth)),wealth[n],colors[i]+',thin',labels[n])
put('figures/wealth.tex',axis(body,'height=4.5cm,xlabel={Held-out trading sessions from split},ylabel={Wealth per initial dollar},legend columns=3,legend style={at={(.5,1.04)},anchor=south,font=\\scriptsize,draw=none},legend cell align=left'))
body=plot(np.arange(len(sr))-.15,sr.IS,'ybar,area legend,bar shift=0pt,bar width=10pt,fill=gray!50,draw=gray','IS')+plot(np.arange(len(sr))+.15,sr.OOS,'ybar,area legend,bar shift=0pt,bar width=10pt,fill=teal!65,draw=teal','OOS')
put('figures/sharpe.tex',axis(body,'height=3.1cm,ylabel={Simple-excess Sharpe},xtick={'+','.join(str(i) for i in range(len(sr)))+'},xticklabels={'+','.join(labels[n] for n in sr.index)+'},x tick label style={font=\\scriptsize,rotate=15,anchor=east},legend columns=2,legend pos=north west'))
print('Generated numbers, 3 tables and 5 TikZ figures from verified CSV inputs.')
