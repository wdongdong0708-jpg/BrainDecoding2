"""An illustrated method overview; keeps the accepted figure suite unchanged.

Protocol schematic only. All waveforms, embedding glyphs and word tokens are
illustrative, not fabricated measurements or model outputs. No results are read
or altered. The two synchronized lower panels share a single fixed target.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, PathPatch
from matplotlib.path import Path as MPath
from scripts.paper.publication_suite_style import style, INK, MUTED, BLUE, TEAL

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures/publication_suite/method_redesign"
RULE = "#D5DFE3"
PALE = "#F1F5F6"
PURPLE = "#776892"
LILAC = "#EFECF5"
SIGNAL = "#778B96"
WARM = "#F7EEDB"


def text(ax, x, y, value, size=6.5, color=INK, weight="normal", ha="left", **kw):
    return ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                   ha=ha, va="center", **kw)


def rect(ax, x, y, w, h, fc="white", ec="none", lw=.6, r=1.1, z=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                      facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    ax.add_patch(p)
    return p


def line(ax, xs, ys, color=RULE, lw=.65, **kw):
    ax.plot(xs, ys, color=color, linewidth=lw, solid_capstyle="round", **kw)


def arrow(ax, p, q, color=MUTED, lw=.8, rad=0, head=5.0, **kw):
    patch = FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=head,
                            connectionstyle=f"arc3,rad={rad}", linewidth=lw,
                            color=color, shrinkA=0, shrinkB=0, **kw)
    ax.add_patch(patch)
    return patch


def path(ax, vertices, codes, fc="none", ec=TEAL, lw=.8, **kw):
    p = PathPatch(MPath(vertices, codes), facecolor=fc, edgecolor=ec,
                  linewidth=lw, capstyle="round", joinstyle="round", **kw)
    ax.add_patch(p)
    return p


def wave(ax, x, y, w, h, phase=0, color=SIGNAL, lw=.9):
    # Deterministic schematic strokes, not recorded neural measurements.
    t = np.linspace(0, 1, 81)
    s = (.48*np.sin(2*np.pi*(2.4*t+.08*phase)) +
         .21*np.sin(2*np.pi*(5.2*t+.19*phase)))
    envelope = .65+.35*np.sin(np.pi*t)
    line(ax, x+t*w, y+s*envelope*h, color=color, lw=lw, zorder=5)


def panel(ax, x, y, letter, title):
    text(ax, x, y, letter, 8.5, weight="bold")
    text(ax, x+4.3, y, title, 7.5, weight="bold")


def neural_portrait(ax):
    """A quiet line illustration with electrode cap, not an anatomical claim."""
    # Head and neck in profile. Coordinates are millimetres on the page.
    v=[(9.1,37.3),(9.9,34.8),(11,34.3),(10.8,31.4),
       (6.0,27.8),(7.2,18.3),(14.2,16.5),(21.0,14.7),(26.0,18.5),(25.2,24.4),
       (25.4,25.8),(27.0,27.3),(26.7,27.9),(25.0,28.4),
       (25.5,31.7),(23.2,33.0),(21.2,33.0),(20.0,33.0),(20.4,35.5),(21.3,37.3)]
    c=[MPath.MOVETO]+[MPath.CURVE4]*9+[MPath.LINETO]*4+[MPath.CURVE4]*6
    path(ax,v,c,fc="#F1F5F4",ec="#7D9298",lw=.9,zorder=2)
    # Cap arcs and electrode sites.
    for vv in [([(9.4,24.0),(9.8,16.8),(19.5,15.2),(23.8,21.0)]),
               ([(12.0,29.3),(8.0,21.0),(17.4,16.0),(20.4,17.4)]),
               ([(10.3,21.5),(14.4,22.6),(18.3,19.7),(23.5,22.0)])]:
        path(ax,vv,[MPath.MOVETO]+[MPath.CURVE4]*3,ec="#AAC5C1",lw=.65,zorder=3)
    for x,y in [(10.2,22.0),(13.3,18.5),(18.2,17.5),(22.4,20.2),(12.3,27.5),(18.4,22.2)]:
        ax.add_patch(Circle((x,y),.75,facecolor=TEAL,edgecolor="white",linewidth=.45,zorder=4))
    # The ear is a single open contour; facial details remain restrained.
    path(ax,[(19.0,27.0),(16.2,24.3),(16.7,31.6),(19.1,29.0)],
         [MPath.MOVETO]+[MPath.CURVE4]*3,ec="#7D9298",lw=.65,zorder=4)
    line(ax,[23.0,23.7],[24.2,24.2],color=INK,lw=.65,zorder=4)


def overview(ax):
    panel(ax,5,6,"a","A neural stream becomes revisable text")
    neural_portrait(ax)
    # Signals are directly attached to the acquisition illustration.
    for k,y in enumerate([21,27,33]):
        path(ax,[(25,y-1),(29,y-1),(29,y),(32,y)],
             [MPath.MOVETO]+[MPath.CURVE4]*3,ec="#C5D7D6",lw=.65)
        wave(ax,32,y,23,2.3,phase=k,color=TEAL,lw=.8)
    text(ax,30.8,42.0,"Word-aligned EEG / MEG",6.5,weight="bold",ha="center")
    text(ax,30.8,47.0,"One neural window per word",6,color=MUTED,ha="center")
    arrow(ax,(58.5,27),(66,27),color="#9BAFB6",lw=.9)

    # A small stack of feature strips replaces an encoder rectangle.
    x0,y0=70,18
    for j in range(5):
        x=x0+4.15*j
        for k in range(7):
            color = ["#DFEAE9","#AECFCA","#6EA39C","#DAE7E5"][(j*3+k*2)%4]
            rect(ax,x,y0+2.4*k,3.3,1.6,fc=color,r=.45)
    # Local interactions are bundled above the representations.
    path(ax,[(73,15.7),(77,12.6),(84,12.6),(88,15.7)],
         [MPath.MOVETO]+[MPath.CURVE4]*3,ec=TEAL,lw=.7)
    ax.add_patch(Circle((80.1,14.0),.7,facecolor=TEAL,edgecolor="white",linewidth=.4,zorder=5))
    text(ax,81,42,"Contextual representations",6.5,weight="bold",ha="center")
    text(ax,81,47,"Neural encoder + local Transformer",6,color=MUTED,ha="center")
    arrow(ax,(93,27),(101,27),color="#9BAFB6",lw=.9)

    # Neural representation is compared against frozen lexical prototypes.
    hub=(106,27)
    text(ax,117,11.5,"Frozen text embeddings",5.8,color=MUTED,ha="center")
    for x,y,col in [(119,17.5,PURPLE),(121,27,TEAL),(119,36.5,BLUE)]:
        line(ax,[hub[0],x-1.7],[hub[1],y],color="#C6D4D9",lw=.75)
        ax.add_patch(Circle((x,y),1.25,facecolor=col,edgecolor="white",linewidth=.4,zorder=4))
    ax.add_patch(Circle(hub,1.7,facecolor=TEAL,edgecolor="white",linewidth=.5,zorder=4))
    text(ax,111,42,"Word scores",6.5,weight="bold",ha="center")
    text(ax,111,47,"Cosine similarity",6,color=MUTED,ha="center")
    # Open readouts, not process boxes. The two vocabularies remain distinct.
    line(ax,[124,130,130],[27,27,19],color="#9BAFB6",lw=.75)
    arrow(ax,(130,19),(136,19),color="#9BAFB6",lw=.75)
    line(ax,[130,130],[27,35],color="#9BAFB6",lw=.75)
    arrow(ax,(130,35),(136,35),color="#9BAFB6",lw=.75)
    for k in range(5):
        ax.add_patch(Circle((139+2.8*k,14.7),.7,facecolor=TEAL if k<2 else "#CDDAD9",edgecolor="none"))
    text(ax,138,20,"Word retrieval",6.7,weight="bold")
    text(ax,138,24.3,"50 candidates · Macro Top-10",5.9,color=MUTED)
    for k,(w,s) in enumerate([("a",5),("young",9),("prince",11)]):
        xx=[138,145,156][k]
        rect(ax,xx,30.5,s,5.0,fc=LILAC,r=.8)
        text(ax,xx+s/2,33,w,5.9,ha="center",color=PURPLE)
    text(ax,138,40,"Rolling sequence output",6.7,weight="bold")
    text(ax,138,44.3,"250-word vocabulary",5.9,color=MUTED)
    text(ax,138,48.6,"Top 5 per word + causal LM",5.9,color=MUTED)
    line(ax,[6,177],[55,55],color=RULE,lw=.65)


def bracket(ax, x1, x2, y, label):
    path(ax,[(x1,y-1),(x1,y),(x2,y),(x2,y-1)],
         [MPath.MOVETO]+[MPath.LINETO]*3,ec="#9EAFB7",lw=.65)
    text(ax,(x1+x2)/2,y+2.8,label,5.9,color=MUTED,ha="center")


def revision_story(ax):
    panel(ax,5,63,"b","More evidence for one target")
    panel(ax,106,63,"c","The editable tail moves forward")
    # Two different visual idioms: open evidence bracket vs equal-width band.
    text(ax,44.5,73,"Earlier words",6,color=MUTED,ha="center")
    text(ax,61.25,73,"Target",6,weight="bold",ha="center")
    text(ax,79,73,"Later words",6,color=MUTED,ha="center")
    text(ax,109.2,73,"Displayed words",6,color=MUTED)
    text(ax,146.25,73,"same target",6,weight="bold",ha="center")
    # Subtle fixed-target guides create a continuous story across the rows.
    line(ax,[61.25,61.25],[78.5,133.5],color="#DDD3BA",lw=.7,ls=(0,(2,2)),zorder=0)
    line(ax,[146.25,146.25],[78.5,138.8],color="#DDD3BA",lw=.7,ls=(0,(2,2)),zorder=0)
    ys=[87,110,133]
    arrow(ax,(9,88),(9,137),color="#B3C1C6",lw=.8,head=5)
    sx=[36+10.5*j for j in range(5)]
    ox=[117+12*j for j in range(5)]
    outputs=[["a","young","child"],["a","young","prince","is"],["a","young","prince","is","here"]]
    labels=["Initial","+1 word","+2 words"]
    for r,y in enumerate(ys):
        n=r+3
        ax.add_patch(Circle((9,y),1.8,facecolor="white",edgecolor="#9CADB5",linewidth=.8,zorder=5))
        text(ax,9,y,str(r+1),5.8,ha="center",weight="bold",zorder=6)
        text(ax,14,y,labels[r],6.2,weight="bold")
        # Evidence is one soft ribbon. Only the fixed target has a frame;
        # only the most recently observed waveform receives the blue accent.
        rect(ax,sx[0]-1.3,y-5.7,sx[n-1]+8.5-sx[0]+2.6,11.4,
             fc="#F2F6F7",r=2,z=1)
        for j in range(n):
            fresh=j==n-1
            if fresh:
                rect(ax,sx[j]-.25,y-5.4,9,10.8,fc="#E2EDF5",r=1.1)
            if j==2:
                rect(ax,sx[j]-.35,y-5.6,9.2,11.2,fc="none",ec=INK,lw=1.1,r=1.25,z=6)
            wave(ax,sx[j]+.65,y,7.2,3.8,phase=j,color=BLUE if fresh else SIGNAL,
                 lw=1.15 if fresh else .8)
        bracket(ax,sx[0]-1.1,sx[n-1]+9.6,y+7.2,f"{n} available signals")
        # The update arrow specifically starts at the pooled target context.
        arrow(ax,(91,y),(105,y),color="#9BAFB6",lw=.8)
        text(ax,98,y-4.3,"target scores",5.8,color=MUTED,ha="center")
        # Jointly decoded suffix is always 3 output positions, moving right.
        start=r
        band_x=ox[start]-1.5
        rect(ax,band_x,y-6.4,37.5,12.8,fc=LILAC,ec=PURPLE,lw=.8,r=2,z=1)
        for j,word in enumerate(outputs[r]):
            frozen=j<start
            target=j==2
            rect(ax,ox[j],y-4.4,10.5,8.8,
                 fc="#EFF2F3" if frozen else (WARM if target else "#FFFFFF"),
                 ec=INK if target else "none",lw=1.05 if target else .5,r=1.05,z=4)
            text(ax,ox[j]+5.25,y,word,6.3,color=MUTED if frozen else INK,
                 weight="bold" if target else "normal",ha="center",zorder=7)
        # Two short cues outside the band show freezing and equal width.
        if r==0:
            text(ax,134.3,y+10,"3 words jointly decoded",5.9,color=PURPLE,ha="center")
        elif r==1:
            text(ax,122.2,y+10,"fixed",5.8,color=MUTED,ha="center")
            text(ax,146.3,y+10,"3 words jointly decoded",5.9,color=PURPLE,ha="center")
        else:
            text(ax,128.2,y+10,"committed prefix",5.8,color=MUTED,ha="center")
            text(ax,158.3,y+10,"3 words jointly decoded",5.9,color=PURPLE,ha="center")
    # A single motion cue links equal-sized decoding bands, without feeding
    # predictions back into neural features.
    arrow(ax,(153.2,95.3),(166.0,102.2),color=PURPLE,lw=.7,rad=.15,head=4.2)
    arrow(ax,(166.3,118.2),(177,126.0),color=PURPLE,lw=.7,rad=.15,head=4.2)

    # Compact shared key. Copy-editable meaning lives below, not in callouts.
    wave(ax,6,153,6,2.4,color=BLUE,lw=1.1)
    text(ax,14,153,"Newly available signal",5.9,color=MUTED)
    rect(ax,59,150.9,3.5,4.2,fc="none",ec=INK,lw=.9,r=.6)
    text(ax,65,153,"Fixed target",5.9,color=MUTED)
    rect(ax,99,150.8,6,4.4,fc=LILAC,ec=PURPLE,lw=.65,r=.7)
    text(ax,107,153,"Editable suffix",5.9,color=MUTED)
    rect(ax,146,151,4.0,4.0,fc="#EFF2F3",r=.5)
    text(ax,152,153,"Committed output",5.9,color=MUTED)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with style():
        fig=plt.figure(figsize=(183/25.4,169/25.4),facecolor="white")
        ax=fig.add_axes([0,0,1,1])
        ax.set(xlim=(0,183),ylim=(169,0));ax.axis("off")
        overview(ax)
        revision_story(ax)
        line(ax,[6,177],[159,159],color=RULE,lw=.6)
        text(ax,6,162.3,"Shared neural model · three-word beam search (width 5), conditioned on committed output",5.8,color=MUTED)
        text(ax,6,166,"Each window ends 1 s after word onset. Signals, feature glyphs and word tokens are schematic.",5.8,color=MUTED)
        for ext in ("pdf","svg","png"):
            p=OUT/f"method_overview_editorial.{ext}"
            fig.savefig(p,dpi=400,facecolor="white")
            print(p)
        plt.close(fig)


if __name__=="__main__":
    main()
