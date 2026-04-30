#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
FIG = ROOT / 'recreated_figures'
FIG.mkdir(exist_ok=True)
rows = json.loads((DATA / 'all_confirm128_rows.json').read_text())['rows']
KEY = ['B1 NoLeague baseline','B3 HeuristicPublicAggro','B4 HeuristicPublicControl']
SHORT = {'B1 NoLeague baseline':'B1','B3 HeuristicPublicAggro':'B3','B4 HeuristicPublicControl':'B4'}
COLORS = {'Anchor':'#2364aa','Architecture':'#5b8e7d','League':'#f18f01','Reward':'#8f5fbf','Auxiliary':'#c73e1d','Algorithm':'#6c757d'}

def save(name):
    plt.savefig(FIG / f'{name}.png', bbox_inches='tight', dpi=220)
    plt.close()
ranked=sorted(rows,key=lambda r:(r['mean_5_anchor'],r['mean_b1_b3_b4']),reverse=True)
fig,ax=plt.subplots(figsize=(9,4.8)); y=np.arange(len(ranked))
ax.barh(y,[r['mean_5_anchor'] for r in ranked],color=[COLORS[r['family']] for r in ranked])
ax.set_yticks(y,[r['label'] for r in ranked]); ax.invert_yaxis(); ax.set_xlabel('Mean win rate'); ax.set_title('Recreated overall ranking')
save('overall_ranking')
fig,ax=plt.subplots(figsize=(9,4.8)); mat=np.array([[r[a] for a in KEY] for r in rows])
im=ax.imshow(mat,vmin=.2,vmax=1,cmap='magma',aspect='auto')
ax.set_xticks(range(len(KEY)),[SHORT[a] for a in KEY]); ax.set_yticks(range(len(rows)),[r['label'] for r in rows])
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]): ax.text(j,i,f'{mat[i,j]:.3f}',ha='center',va='center',fontsize=8,color='white' if mat[i,j]<.68 else 'black')
fig.colorbar(im,ax=ax,fraction=.025,pad=.02); ax.set_title('Recreated key-anchor matrix')
save('key_anchor_matrix')
print(f'wrote recreated figures to {FIG}')
