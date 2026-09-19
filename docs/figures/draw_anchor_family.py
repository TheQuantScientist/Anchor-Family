#!/usr/bin/env python3
"""Minimal graph-style figure for the Anchor family."""
from __future__ import annotations
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/chronolm-figure-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np
HERE=Path(__file__).resolve().parent
W,H=720,440
INK="#17212B"; MUTED="#66737D"; GRID="#D7DEE3"; BLUE="#136F9A"; BLUE_LIGHT="#EAF4F8"; TEAL="#16827A"; TEAL_LIGHT="#EAF6F3"; ORANGE="#C56827"; ORANGE_LIGHT="#FCF1E8"; PURPLE="#7654A4"; PURPLE_LIGHT="#F2EFF8"; RED="#B44B45"; RED_LIGHT="#F9EEEE"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","svg.hashsalt":"chronolm-anchor-v3"})
class G:
    def __init__(self):
        self.fig=plt.figure(figsize=(W/72,H/72),facecolor="white"); self.ax=self.fig.add_axes((0,0,1,1)); self.ax.set(xlim=(0,W),ylim=(H,0)); self.ax.axis("off")
    def text(self,x,y,s,size=10,color=INK,weight="normal",ha="left"):
        return self.ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va="center",zorder=5)
    def box(self,x,y,w,h,fill="white",edge=GRID,lw=1,radius=5):
        self.ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={radius}",facecolor=fill,edgecolor=edge,linewidth=lw,zorder=1))
    def line(self,pts,color=GRID,lw=1,dashed=False,z=2):
        xs,ys=zip(*pts); self.ax.add_line(Line2D(xs,ys,color=color,lw=lw,linestyle=(0,(4,3)) if dashed else "-",zorder=z))
    def arrow(self,a,b,color=INK,lw=1.1,dashed=False):
        self.ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=9,linewidth=lw,color=color,shrinkA=0,shrinkB=0,linestyle=(0,(4,3)) if dashed else "-",zorder=3))
    def save(self):
        self.fig.canvas.draw(); stem=HERE/"anchor_family_architecture"; meta={"Title":"Anchor family computational graph","Author":"","Creator":"ChronoLM","CreationDate":None,"ModDate":None}
        self.fig.savefig(stem.with_suffix(".pdf"),metadata=meta); self.fig.savefig(stem.with_suffix(".svg"),metadata={"Date":None}); self.fig.savefig(stem.with_suffix(".png"),dpi=600); self.fig.savefig(HERE/"anchor_family_architecture_preview.png",dpi=180); self.fig.savefig(HERE/"anchor_family_architecture_print_proof.png",dpi=180); print("Wrote vector PDF/SVG and 600 dpi PNG."); plt.close(self.fig)
def history_graph(g,x,y,w,h,accent=BLUE,labels=True):
    tt=np.array([0,.8,1.7,2.3,3.4,4.7,5.1,6.2,7.1,8.0]); vv=np.array([.40,.54,.84,.68,.28,.37,.88,.66,.25,.52]); q=np.array([8.7,9.5,10.4]); xp=lambda t:x+np.asarray(t)/11*w; yp=lambda v:y+h-(np.asarray(v)+.05)/1.05*h
    g.line([(x,y+h),(x+w,y+h)],color=GRID,lw=.8); cut=float(xp(8.35)); g.line([(cut,y-3),(cut,y+h+3)],color=MUTED,lw=.8,dashed=True); g.ax.plot(xp(tt),yp(vv),color="#AAB6BE",lw=1.1,zorder=2); g.ax.scatter(xp(tt),yp(vv),s=18,color=INK,zorder=4)
    pred=np.array([.53,.53,.53]) if accent==TEAL else np.array([.54,.66,.58]); g.ax.plot(xp(q),yp(pred),color=accent,lw=2.0,zorder=4); g.ax.scatter(xp(q),yp(pred),s=35,facecolor="white",edgecolor=accent,lw=1.5,zorder=5)
    if labels:
        g.text((x+cut)/2,y+h+14,"observed",8.5,MUTED,ha="center"); g.text((cut+x+w)/2,y+h+14,"forecast queries",8.5,MUTED,ha="center")
def draw():
    g=G(); g.text(20,18,"Anchor Family",17,INK,"bold"); g.text(20,39,"One shared candidate space; five alternative selection policies",10.5,MUTED); g.text(700,28,"deterministic • no backpropagation",9,MUTED,ha="right")
    g.box(20,63,138,120,fill=BLUE_LIGHT,edge=BLUE,lw=1.2); g.text(89,79,"Irregular history",11,BLUE,"bold","center"); history_graph(g,34,93,110,61,BLUE,False); g.text(89,171,r"$\mathcal{H}_v=\{(t_j,x_j)\}$",10,INK,ha="center")
    g.box(190,63,190,120,fill=TEAL_LIGHT,edge=TEAL,lw=1.2); g.text(285,79,"Shared candidate space",11,TEAL,"bold","center")
    labels=[("last","persistence"),("EMA","smoothing"),("mode","repetition"),("mean / trim","robust level"),("trend","local slope"),("phase","recurrence")]
    for i,(a,b) in enumerate(labels):
        xx=203+(i%2)*88; yy=94+(i//2)*24; g.box(xx,yy,80,17,fill="white",edge="#9CCBC5",lw=.7,radius=2); g.text(xx+40,yy+9,a,8.5,INK,"bold","center")
    g.text(285,171,"+ fixed convex mixtures (85 total)",8.5,MUTED,ha="center"); g.arrow((158,123),(190,123),BLUE)
    g.text(405,57,"Selection policy",9.3,MUTED,"bold")
    rows=[("NaiveAnchor","last",MUTED),("ExpoAnchor","EMA",TEAL),("SparseAnchor","sparse",ORANGE),("ERMAnchor","risk",PURPLE),("AutoAnchor","ERM + prior",RED)]
    for i,(name,desc,color) in enumerate(rows):
        yy=74+i*25; g.box(404,yy,137,20,fill=RED_LIGHT if name=="AutoAnchor" else "white",edge=color,lw=1.3 if name=="AutoAnchor" else .8,radius=3); g.text(411,yy+10,name,8.1,color,"bold"); g.text(534,yy+10,desc,7.0,MUTED,ha="right"); g.arrow((380,123),(404,yy+10),TEAL if name=="AutoAnchor" else GRID,.75)
    g.box(404,203,137,89,fill=ORANGE_LIGHT,edge=ORANGE,lw=1.0); g.text(472,217,"Auto prior",10,ORANGE,"bold","center"); g.text(413,236,"1  repeated values → sparse",8.4); g.text(413,253,"2  long window → EMA",8.4); g.text(413,270,"3  phase / trend → mixture",8.4); g.text(472,285,"first matching rule",8.2,MUTED,ha="center"); g.arrow((472,199),(472,203),ORANGE,.9); g.arrow((472,292),(472,318),ORANGE,.9)
    g.box(565,63,135,120,fill=PURPLE_LIGHT,edge=PURPLE,lw=1.2); g.text(632,78,"Chosen specification",10,PURPLE,"bold","center"); g.text(632,108,r"$\theta_v=(a_v,\beta_v)$",14,INK,ha="center"); g.text(632,132,"one choice per variable",8.6,MUTED,ha="center"); g.text(632,150,"frozen before test",8.6,MUTED,ha="center")
    for i in range(5): g.arrow((541,84+i*25),(565,123),PURPLE if i in (3,4) else GRID,.75)
    g.line([(20,319),(700,319)],color=GRID,lw=.8); g.text(20,338,"Forecast operator",13,INK,"bold"); g.text(700,338,"same operator for every variant",9,MUTED,ha="right")
    history_graph(g,32,353,180,54,TEAL); g.box(253,350,136,62,fill=TEAL_LIGHT,edge=TEAL,lw=1.1); g.text(321,366,"Selected anchor",10,TEAL,"bold","center"); g.text(321,388,r"$f_{a_v}(\mathcal{H}_{b,v},t^*)$",12,INK,ha="center"); g.arrow((212,380),(253,380),INK)
    g.box(430,350,105,62,fill=PURPLE_LIGHT,edge=PURPLE,lw=1.1); g.text(482,366,"Postprocess",10,PURPLE,"bold","center"); g.text(482,388,r"$\beta_v$ + clip",12,INK,ha="center"); g.arrow((389,380),(430,380),INK); g.arrow((535,380),(566,380),BLUE); g.text(633,362,"Forecast",10,BLUE,"bold","center"); history_graph(g,568,353,120,54,BLUE); g.text(25,430,r"$\hat y_{b,k,v}=\mathcal{P}_{b,v}[\,\beta_v f_{a_v}(\mathcal{H}_{b,v},t^*_{b,k,v})\,]$",12,INK); g.text(700,430,"dots = observed; open dots = predictions",8.3,MUTED,ha="right"); g.save()
if __name__=="__main__": draw()
