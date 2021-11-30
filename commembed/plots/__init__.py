
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.transforms import Bbox

def add_subplot_label(ax, label, x=-30, y=0): 
    ax.annotate(label, xy=(0, 1), xycoords='axes fraction',
      xytext=(x, y), textcoords='offset points',
      fontsize=16, fontweight='bold', va='top', ha='left')


def render_part(part_fn, fig_size, *args, **kwargs):
    fig = plt.figure(figsize=fig_size)
    part_fn(fig, gridspec.GridSpec(1, 1, figure=fig)[0,0], *args, **kwargs)
    plt.show()

def adjust_date_ticks(ax, months, do_y=True, include_extra_ticks=False):

    jan_label = lambda x: x.split("-")[0] if (x.endswith("-01") or '-' not in x) else "" # include full year ticks

    if include_extra_ticks:
      ticks = [(x, jan_label(label)) for x, label in enumerate(months)]
    else:
      ticks = [(x, jan_label(label)) for x, label in enumerate(months) if jan_label(label) != ""]


    #ax.minorticks_on()

    if do_y:
      ax.set_yticks([x[0] for x in ticks if x[1] != ""])
      ax.set_yticklabels([x[1] for x in ticks if x[1] != ""])
      ax.set_yticks([x[0] for x in ticks if x[1] == ""], minor=True)
      ax.set_yticklabels([x[1] for x in ticks if x[1] == ""], minor=True)

    ax.set_xticks([x[0] for x in ticks if x[1] != ""])
    ax.set_xticklabels([x[1] for x in ticks if x[1] != ""])
    ax.set_xticks([x[0] for x in ticks if x[1] == ""], minor=True)
    ax.set_xticklabels([x[1] for x in ticks if x[1] == ""], minor=True)


def nudge_axis(ax, x, y):
    sp_ax = ax.get_position()
    ax.set_position(Bbox(sp_ax.get_points() + [[x, y], [x, y]]))

    
def shrink_axis(ax, x, y):
    sp_ax = ax.get_position()
    ax.set_position(Bbox(sp_ax.get_points() + [[0, 0], [-x, -y]]))