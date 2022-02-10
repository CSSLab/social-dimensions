"""Plotting code for Figure 5."""
import sys
import os
sys.path.append("../..")
from commembed.jupyter import *
import commembed.linalg as linalg
import commembed.dimens as dimens
import commembed.plots as plots
import commembed.data as data
import numpy as np
import pandas as pd
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import matplotlib.gridspec as gridspec
#import commembed.data as data
#spark = data.spark_context()


matplotlib.rcParams['mathtext.fontset'] = 'custom'

embedding = load_embedding('reddit', 'master')
dimen_list = dimens.load_dimen_list('final')
scores = dimens.score_embedding(embedding, dimen_list)


selected_months = ["%04d-%02d" % (year, month) for year in range(2011, 2019) for month in range(1, 13)]

from datetime import datetime
import matplotlib
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.ticker as mtick


def render_cohort_plot(fig, gs, partisan_dimen, political_activity_category, sharey=None, legend_bottom_right=False, point_size=None, sd=None, sd_key=None):

    ax = fig.add_subplot(gs, sharey=sharey)
    
    path = os.path.join(data.DATA_PATH, "cohort_data_%s_%s.parquet" % (partisan_dimen, political_activity_category))
    
    to_plot = pd.read_parquet(path)
    
    to_plot = to_plot[to_plot["month"] > "2012"]
    to_plot = to_plot[to_plot["month"] < "2019"]
    
    avg_abs_z = to_plot.pivot(index="month", columns="first_year", values="avg_abs_z_score")
    avg_abs_z.plot.line(cmap='viridis_r', ax=ax, style='.-', markersize=point_size)

    if sd is not None:
        sd[sd_key] = avg_abs_z

    #for x in first_points["month"]:
    #    x = avg_abs_z.index.get_loc(x)
    #    ax.axvline(x, linestyle='--', color='#555555', linewidth=1)

    #ax.set_title(plot_type)

    legend_labels = list(avg_abs_z.columns)
    legend_labels[0] = "$\leq$" + legend_labels[0]

    if legend_bottom_right:
        ax.legend(loc='lower right', ncol=3, frameon=False, labels=legend_labels, fontsize=9, handlelength=1, 
            labelspacing=0.5, columnspacing=1)
    else:
        ax.legend(loc='upper left', ncol=2, frameon=False, labels=legend_labels)

    ax.set_ylabel("Polarization")
    ax.set_xlabel("")

    ax.margins(x=0.02)

    plots.adjust_date_ticks(ax, avg_abs_z.index, do_y=False, include_extra_ticks=True)

    return ax

import matplotlib.gridspec as gridspec

def render_left_right_polarization(partisan_dimen):

    figsize = (9,(9/11)*9)
    fig = plt.figure(figsize=figsize)
    gs0 = gridspec.GridSpec(3, 8, figure=fig)
    gs0.update(wspace=1.7, hspace=0.4)

    gs_a = gs0[1, 0:2]
    gs_b = gs0[1, 2:4]
    
    #gs_c = gs0[1, 0:4]
    gs_d = gs0[1, 4:6]
    gs_e = gs0[1, 6:]
    
    gs_f = gs0[2, 0:4]
    gs_g = gs0[2, 4:]
    
    source_data = {}
    
#     ax_line = pol.render_abs_z_line_plot(fig, gs_a, partisan_dimen)
#     plots.add_subplot_label(ax_line, "a")
#     ax_line2 = pol.render_abs_z_line_plot(fig, gs_b, partisan_dimen, metric="num_comments")
#     plots.add_subplot_label(ax_line2, "b")

    s = 4.5
    ax_cohort_l = render_cohort_plot(fig, gs_f, partisan_dimen, "left", point_size=s, sd=source_data, sd_key="a")
    plots.add_subplot_label(ax_cohort_l, "a")
    ax_cohort_r = render_cohort_plot(fig, gs_g, partisan_dimen, "right", sharey=ax_cohort_l, point_size=s, sd=source_data, sd_key="b")
    plots.add_subplot_label(ax_cohort_r, "b")
    
#     ax_pol_l = pol.render_explained_polarization(fig, gs_d, partisan_dimen, "left")
#     plots.add_subplot_label(ax_pol_l, "c", x=-40)
#     ax_pol_r = pol.render_explained_polarization(fig, gs_e, partisan_dimen, "right", sharey=ax_pol_l)
#     plots.add_subplot_label(ax_pol_r, "d", x=-40)
    
    ax_cohort_r.get_legend().remove()
#     ax_pol_r.get_legend().remove()

    ax_cohort_r.set_ylim(1,3.9)
    
    # keep ticks
    ax_cohort_r.yaxis.set_tick_params(labelleft=True)
#     ax_pol_l.yaxis.set_tick_params(labelleft=True)
#     ax_pol_r.yaxis.set_tick_params(labelleft=True)
#     ax_pol_l.text(-2.6, 0.4, "Change in polarization", va='center', rotation='vertical')
    
    l_col = [0.2298057 , 0.29871797, 0.75368315, 1.        ]
    r_col = [0.70567316, 0.01555616, 0.15023281, 1.        ]
    for l, r in [(ax_cohort_l, ax_cohort_r)]: # (ax_pol_l, ax_pol_r), 
        l.annotate("Left", (1, 1), xytext=(-10, -10), textcoords='offset points',
                   xycoords='axes fraction', ha='right', va='top', color=l_col,
                   weight='bold')
        r.annotate("Right", (0, 1), xytext=(10, -10), textcoords='offset points',
                   xycoords='axes fraction', ha='left', va='top', color=r_col,
                   weight='bold')

        
    return fig, [], source_data

render_left_right_polarization("partisan")