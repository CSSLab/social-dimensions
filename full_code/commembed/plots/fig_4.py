"""Plotting code for Figure 4."""
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


    
def render_confound_plot(fig, gs, partisan_dimen, name, political_activity_category='all', sd=None, sd_key=None):
    
    ax = gs if isinstance(gs, matplotlib.axes.Axes) else fig.add_subplot(gs)
    
    to_plot = pd.read_csv(os.path.join(data.DATA_PATH, "user_confound_%s_%s_%s.csv" % (name, partisan_dimen, political_activity_category)), index_col=0)

    to_plot.plot.line(y="avg_abs_z_score", ax=ax, legend=False)

    ax.set_ylabel("Polarization")

    xlabel = {
        "month_since_join_index": "Months",
        "active_month_index": "Active months"
    }
    ax.set_xlabel(xlabel[name])
    
    ax.set_ylim(0, 2)

    if sd is not None:
        sd[sd_key] = to_plot

    return ax


def render_polarization_heatmap(fig, gs, partisan_dimen, political_activity_category='all', kind='correlation', line=True, sd=None, sd_key=None):
    """kind should be correlation or polprob"""
    ax = fig.add_subplot(gs)
    
    fname = "user_dynamic_%s_%s_%s.csv" % (kind, partisan_dimen, political_activity_category)
    path = os.path.join(data.DATA_PATH, fname)
    mat = pd.read_csv(path, index_col=0)

    # filter time period to >= 2012 and <= 2018
    mat = mat[mat["m1"] > "2012"]
    mat = mat[mat["m2"] > "2012"]
    mat = mat[mat["m1"] < "2019"]
    mat = mat[mat["m2"] < "2019"]

    if kind == "polprob":
        
        # Print some interesting statistics
        to_print = mat.set_index(["m1", "m2"])
        months = sorted(mat["m1"].unique())
        indices = list(zip(months[:-12], months[12:]))
        to_print = to_print.loc[indices, :].reset_index()
        
        pre_2016 = to_print[to_print["m2"]<"2016"]
        print("Polprob pre 2016: min %.4f, max %.4f" % (np.amin(pre_2016["proportion_over"]), np.amax(pre_2016["proportion_over"])))

        peak = to_print["proportion_over"].idxmax()
        print("Polprob max: %s" % to_print.loc[peak].to_dict())

    mat = mat.pivot(index="m1", columns="m2", values="proportion_over" if kind == 'polprob' else "pearson_r")
    

    if kind == "polprob":
        mat = mat.fillna(0) # always 0 probability of polarizing in same month



    #melted = mat.reset_index().melt(id_vars=['month']).set_index(["month", "variable"])
    
    #idxmax = melted["value"].idxmax()
    #print("\tmax: %s %s" % (idxmax, melted.loc[idxmax, "value"]))
    
    label = {
        "correlation": 'Pearson\'s $r$ of scores at $t_x$ and $t_y$',
        "polprob": 'Frac. $|z_{x}|-|z_{y}| \\geq 1$'
    }
    
    cmap = {"correlation":"magma","polprob":"cividis"}
    vmax = {"correlation":1,"polprob":0.3}
    
    sns.heatmap(mat, ax=ax, cmap=cmap[kind], vmin=0, vmax=vmax[kind], rasterized=True,
                cbar_kws={'label': label[kind]})

    if sd is not None:
        sd[sd_key] = mat

    
    #ax.add_patch(Rectangle((x, y), 1, 1, alpha=1, fill=None, lw=2, ec='b', clip_on=False))
    
    #if line:
    #    ax.plot([12,len(mat)], [0,len(mat)-12], linestyle='--', c='black', linewidth=1)

    
    plots.adjust_date_ticks(ax, mat.index)
    #ax.set_xlabel("$t_2$")
    #ax.set_ylabel("$t_1$")
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        
    
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.tick_params(axis='x', which='major', rotation=45)

    return ax


def render_explained_polarization(fig, gs, partisan_dimen, political_activity_category, sharey=None, fontsize=9, sd=None, sd_key=None):
    
    ax = fig.add_subplot(gs, sharey=sharey)
    
    path = os.path.join(data.DATA_PATH, "explained_polarization_%s_%s.csv" % (partisan_dimen, political_activity_category))
    result = pd.read_csv(path)
    result = result.set_index("year")
    
    ax.axhline(0, linewidth=1, linestyle='--', color='#000000', label='_nolegend_')

    default_colors = ["#666666", "#D55E00", "#009E73"] # #56B4E9

    # secondary_alpha = 0.5
    party_colors = {
        'all': "#56B4E9",
        'left': [0.2298057 , 0.29871797, 0.75368315, 1],
        'right': [0.70567316, 0.01555616, 0.15023281, 1]
    }
    # colors = np.array([colors[political_activity_category]] * 3)
    # colors[0][3] = 1

    colors = list(default_colors)
    #colors[0] = party_colors[political_activity_category]
    
    result[['actual','static_new','static_old']].plot.bar(color=colors, ax=ax)

    if sd is not None:
        sd[sd_key] = result[['actual','static_new','static_old']]

    print("Explained polarization (%s, %s) in 2016: %s" % (partisan_dimen, political_activity_category, 
        result[['actual','static_new','static_old']].loc[2016].to_dict()))

    #for container, hatch in zip(ax.containers, ['', '////', '....']):
    #    for bar in container.patches:
    #        bar.set_hatch(hatch)

    ax.set_xlabel("")
    ax.set_ylabel("Change in polarization")
    ax.legend([
        "$\Delta$e + $\Delta$n",
        "$\Delta$e",
        "$\Delta$n"
    ], frameon=False, prop={'size': fontsize}, loc='upper left')
    
    plots.adjust_date_ticks(ax, result.index.astype(str), do_y=False)

    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='x', which='major', rotation=45)

    return ax

def render_polarization_2(partisan_dimen):
    
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    figsize = (8,3.6)
    fig = plt.figure(figsize=figsize)
    gs0 = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1,1,0.7])
    gs0.update(wspace=0.4, hspace=0.5)

    gs_a = gs0[:, 0:2]
    gs_b = gs0[0, 2]
    gs_c = gs0[1, 2]

    source_data = {}
    
    ax_cohort = render_cohort_plot(fig, gs_a, partisan_dimen, "all", legend_bottom_right=True)
    plots.add_subplot_label(ax_cohort, "a", x=-40)
    ax_cohort.set_ylabel("Polarization (avg. absolute z-score)")
    
    ax_r = render_polarization_heatmap(fig, gs_b, partisan_dimen, kind="polprob")
    plots.add_subplot_label(ax_r, "b", x=-50)
    
    ax_pol = render_explained_polarization(fig, gs_c, partisan_dimen, "all")
    plots.add_subplot_label(ax_pol, "c", x=-50)
    
    confound_ax_size = 0.6
    x_offset = 0.09
    
    ax_confound_1 = inset_axes(ax_cohort, width=confound_ax_size, height=confound_ax_size,
                       loc='upper left', borderpad=1,
                       bbox_to_anchor=(x_offset, 0, 1, 1),
                       bbox_transform=ax_cohort.transAxes)
    ax_confound_2 = inset_axes(ax_cohort, width=confound_ax_size, height=confound_ax_size,
                       loc='upper left', borderpad=1,
                       bbox_to_anchor=(x_offset + 0.22, 0, 1, 1),
                       bbox_transform=ax_cohort.transAxes)
    
    render_confound_plot(fig, ax_confound_1, partisan_dimen, "month_since_join_index")
    #plots.add_subplot_label(ax_confound_1, "c", x=-50)
    
    render_confound_plot(fig, ax_confound_2, partisan_dimen, "active_month_index")
    ax_confound_2.set_ylabel("")
    #plots.add_subplot_label(ax_confound_2, "d", x=-60)
    
    ymax = 2.2
    ymin = 1
    for ax in [ax_cohort, ax_confound_1, ax_confound_2]:
        ax.set_ylim(ymin, ymax)

    return fig, [], source_data

render_polarization_2("partisan")