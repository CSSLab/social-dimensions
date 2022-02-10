"""Plotting code for Figure 3."""
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


def render_echo_chamber_plot(fig, gs, dimen_to_plot, nullhyp=False, all_data=False, sd=None, sd_key=None):
    data_suffix = "" if all_data else "_filt_" + dimen_to_plot
    user_dists_table = "user_dists_bins=5_breadth=2_" + data_suffix
    nullhyp_user_dists_table = "user_dists_nullhyp_bins=5_breadth=2_" + data_suffix

    path = os.path.join(data.DATA_PATH, \
        "echo_chamber_results_%s.csv" % user_dists_table)    
    bin_sizes_path = os.path.join(data.DATA_PATH, \
        "echo_chamber_bin_sizes_%s.csv" % user_dists_table)

    if nullhyp:
        # render nullhyp as main result
        path = os.path.join(data.DATA_PATH, \
            "echo_chamber_results_%s.csv" % nullhyp_user_dists_table)
        bin_sizes_path = os.path.join(data.DATA_PATH, \
            "echo_chamber_bin_sizes_%s.csv" % nullhyp_user_dists_table)

    result = pd.read_csv(path)
    bin_sizes = pd.read_csv(bin_sizes_path)
    
    gs = gs.subgridspec(2, 1, height_ratios=[0.5,3])

    cmap = cm.ScalarMappable(cmap=axis_colormap("partisan"))
    cmap.set_clim(-2, 2)
    partisan_colors = cmap.to_rgba([-2,-1,0,1,2])
    partisan_colors[2] = [0.73,0.73,0.73,1]

    cmap = axis_colormap(dimen_to_plot)
    quantiles = axis_quantile_names(dimen_to_plot)

    ax = fig.add_subplot(gs[1])
    ax_hist = fig.add_subplot(gs[0], sharex=ax)

    # histogram on top
    dimen_to_target = "partisan_bin" if (nullhyp and all_data == False) else dimen_to_plot # TODO hack

    sizes = bin_sizes[bin_sizes["dimen"] == dimen_to_target]
    sizes = sizes.set_index("bin").reindex([-2,-1,0,1,2])
    sizes["pct"] = sizes["total_comments"] / np.sum(sizes["total_comments"])
    sizes["pct"].plot.bar(ax=ax_hist, width=0.4, color=partisan_colors, legend=False)
    ax_hist.set_frame_on(False)
    ax_hist.set_ylabel("Pct. comments")
    ax_hist.yaxis.set_major_formatter(mtick.PercentFormatter())
    for b, row in sizes.iterrows():
        #ax_hist.annotate(str(row["pct"]), (b+2, row["pct"]))
        ax_hist.annotate("%.0f%%" % (100*row["pct"]), (b+2,row["pct"]), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9)
    ax_hist.axes.get_yaxis().set_visible(False)
    plt.setp(ax_hist.get_xticklabels(), visible=False)


    value_to_use = "bin1_avg_frac_bin2"

    to_plot = result[result["dimen"] == dimen_to_target]

    result_pivoted = to_plot.pivot(index="bin1", columns="bin2", values=value_to_use)
    result_pivoted = result_pivoted.reindex(index=[-2, -1, 0, 1, 2], columns=[-2, -1, 0, 1, 2]).fillna(0)

    print("Echo chamber (left, left) = %.4f" % result_pivoted.loc[-2, -2])
    print("Echo chamber (right, right) = %.4f" % result_pivoted.loc[2, 2])

    if nullhyp:
        deviations = (result_pivoted - sizes["pct"])
        print("Echo chamber nullhyp all bin pct deviations (max %f):" % np.amax(deviations.values))
        display(deviations)

    result_pivoted.plot.bar(stacked=False, color=partisan_colors, label=quantiles, width=0.8, ax=ax, legend=False)
    if sd is not None:
        sd[sd_key] = result_pivoted

    ax.set_frame_on(False)
    ax.set_xticklabels([q.replace(' ','\n', 1) for q in quantiles], rotation='horizontal', fontsize=7)
    ax.set_xlabel("Community partisan score bin")
    ax.set_ylabel("Avg. pct. of author activity")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    #ax_hist.get_legend().remove()
    #ax.get_legend().remove()

    cmap = cm.ScalarMappable(cmap=cmap)
    cmap.set_clim(-2, 2)
    [t.set_color(i) for (i,t) in
    zip(partisan_colors,ax.xaxis.get_ticklabels())]

    #plt.legend(labels=quantiles, bbox_to_anchor=(1.05, 1), bbox_transform=ax.transAxes)

    return ax_hist


def render_partisan_activity_plot(f, gs, partisan_dimen, activity_frac_mode=False, just_first_part=False, sd=None, sd_key=None):

    
    activity_data = pd.read_parquet("/ada/data/reddit/parquet/%s_activity_data_12mo.parquet" % partisan_dimen)


    
    if sd is not None:
        df = activity_data.groupby(["subreddit", "month"]).agg({"num_comments":"sum"})
        df = df.join(scores[[partisan_dimen]], on='subreddit')
        df = df.rename(columns={partisan_dimen:"partisan"})
        sd[sd_key] = df
    
    score_col_name = partisan_dimen
    
    mean = np.mean(scores[score_col_name])
    std_dev = np.std(scores[score_col_name])

    # partisan_bins are left side edges of bins
    
    if activity_frac_mode:
        mean = np.mean(scores[score_col_name])
        std = np.std(scores[score_col_name])
        partisan_bins = np.array([mean - 2*std, mean + 2*std])
    else:
        n_partisan_bins = 30
        partisan_bins = np.linspace(np.amin(scores[score_col_name]), np.amax(scores[score_col_name]), n_partisan_bins+1)
        
        #partisan_bins = np.array([mean-3*std_dev,mean-2*std_dev, mean-std_dev, mean, mean+std_dev,mean+2*std_dev,mean+3*std_dev])
        #n_partisan_bins = len(partisan_bins)

    scores_to_plot = scores[[score_col_name]].copy().rename(columns={score_col_name:"partisan"})

    scores_to_plot["partisan"] = np.digitize(scores_to_plot["partisan"], partisan_bins)
    
    # merge cat 2 and 3, we no longer make a distinction between them
    activity_data["last_month_user_bin"] = activity_data["last_month_user_bin"].replace(3, 2)
    
    binned_activity_data = activity_data.join(scores_to_plot, on="subreddit").groupby(["partisan", "month", "last_month_user_bin"])\
        .agg({"num_comments":"sum"})
    binned_activity_data.head(5)
    
    total_monthly_comments = binned_activity_data.reset_index().groupby(["month"]).agg({"num_comments":"sum"})
    total_monthly_comments = total_monthly_comments.rename(columns={"num_comments":"total_month_comments"})
    
    binned_activity_data = binned_activity_data.reset_index().set_index(["month"]).join(total_monthly_comments)
    binned_activity_data["num_comments"] /= binned_activity_data["total_month_comments"]
    binned_activity_data = binned_activity_data.reset_index()
    binned_activity_data.head(5)
    
    all_bins_activity_data = binned_activity_data.groupby(["month","partisan"]).agg({"num_comments":"sum"}).reset_index()
    
    months_to_plot = selected_months
    def df_for_user_bin(user_bin):
        return binned_activity_data[binned_activity_data["last_month_user_bin"] == user_bin]
    dfs = [all_bins_activity_data] + [df_for_user_bin(user_bin) for user_bin in [1, 2, 4, 5]]
    dfs = [df.pivot(index="month", columns="partisan", values="num_comments") for df in dfs]
    dfs = [df.reindex(months_to_plot[12:]) for df in dfs]
    

    months = all_bins_activity_data["month"].unique()
    months.sort()
    yrs = [datetime.strftime(datetime.strptime(m, '%Y-%m'),'%Y') for m in months]
    mos = [datetime.strftime(datetime.strptime(m, '%Y-%m'),'%B') for m in months]
    ticks = [mo[0] if mo != "January" else (mo[0] + "\n" + yr) for mo, yr in zip(mos, yrs)]
    ticks_yronly = ["" if mo != "January" else yr for mo, yr in zip(mos, yrs)]
    

    cmap = axis_colormap("partisan")
    #cmap = cm.get_cmap("coolwarm")
    cmap = cm.ScalarMappable(cmap=cmap)
    cmap.set_clim(-3, 3)

    m = cmap
    
    bin_width = partisan_bins[1]-partisan_bins[0]
    # since the values returned from digitize are offset
    offset_bin_centers = np.array([partisan_bins[0]-(bin_width/2)] + list(partisan_bins+(bin_width/2)))

    
    def plot_df(df, ax):

        partisan_values = offset_bin_centers[df.columns]
        z_scores = (partisan_values - mean) / std_dev

        if not df.empty:
            df.plot.bar(ax=ax, color=m.to_rgba(z_scores), stacked=True, legend=False, width=0.8, rasterized=False)

        ax.set_ylim(0, 0.5)

    gap_size = 0.8

    if just_first_part:
        gs = gs.subgridspec(1, 1)
        dfs = [dfs[0]]
    else:
        gs = gs.subgridspec(6, 1, height_ratios=[2,gap_size,1,1,1,2])

    axs = np.array([f.add_subplot(g) for g in gs])
    if not just_first_part:
        axs[1].axis('off')

    user_label_labels = [None, "Left-wing", "Center",
                         "Right-wing", "None\n(new or newly\npolitical users)", None]

    for i, df in enumerate(dfs):
        ax = axs[i if i == 0 else i + 1]
        plot_df(df, ax)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        if i == 0 or i == 4:
            ax.set_ylim(0, 1)

        if i > 0:
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        if i == 4:
            ax.set_xlabel("Month ($t$)")
        elif just_first_part:
            ax.set_xlabel("Month")
        else:
            ax.set_xlabel("")
            
            
        # Set tick labels
        ax.tick_params(axis = 'x', rotation = 0, labelsize=9)
        if i == 0:
            # just years
            #ax.set_xticklabels(ticks_yronly)
            plots.adjust_date_ticks(ax, months, do_y=False, include_extra_ticks=True)
        elif i == 4:
            # also use just years (change to ticks if you want to include months)
            ax.set_xticklabels(ticks_yronly)
        else:
            # no tick labels
            ax.set_xticklabels([])
            

        if i == 0 and not just_first_part:
            f.text(-0.08, 0.5,'All users',
                 horizontalalignment='right',
                 verticalalignment='center',
                 transform = ax.transAxes, weight='bold')
            f.text(-0.05, 1.1,'Proportion of\nall comments\nin the month',
                 horizontalalignment='center',
                 verticalalignment='bottom',
                 transform = ax.transAxes)

        if i == 1:
            f.text(-0.08, 1,'User label at\n$t-12$:',
                 horizontalalignment='right',
                 verticalalignment='center',
                 transform = ax.transAxes, weight='bold')

        label = user_label_labels[i]
        if label is not None:
            f.text(-0.08, 0.25 if i != 4 else 0.4,label,
                 horizontalalignment='right',
                 verticalalignment='center',
                 transform = ax.transAxes)

    #colorbar 


    #t_d_month = list(months).index("2015-06")
    #ax.axvline(t_d_month)

    #cax = plt.axes((-0.02, 0.11, 0.08, 0.017))
    cbar_width = 0.4
    cbar_height = 0.04

    # Place colorbar
    if not just_first_part:
        x, y, width, height = (0.95 - cbar_width, 1.23 - cbar_height, cbar_width, cbar_height)
    else:
        x, y, width, height = (0.95 - cbar_width, 1.23 - cbar_height, cbar_width, cbar_height)
    t = (axs[0].transAxes + f.transFigure.inverted()).transform_point
    x_t, y_t = t((x, y))
    width_t, height_t = t((x + width, y + height))
    width_t -= x_t
    height_t -= y_t
    cax = plt.axes((x_t, y_t, width_t, height_t))

    cbar = f.colorbar(m, cax=cax, orientation='horizontal')
    ctx = [-3, -2, -1, 0, 1, 2, 3]
    cbar.set_ticks(ctx)
    #cbar.set_ticklabels([("$%d\sigma$" % d) for d in ctx])
    cbar.set_ticklabels(["-3$\sigma$","","","0","","","3$\sigma$"])
    cax.tick_params(axis='both', which='major', labelsize=9)

    # if not just_first_part:
    #     plots.add_subplot_label(axs[0], "c")
    #     plots.add_subplot_label(axs[2], "d")
    
    return axs[0]
    
def render_polarization_1(partisan_dimen):

    figsize = (7,2.5)
    fig = plt.figure(figsize=figsize)
    gs0 = gridspec.GridSpec(10, 3, figure=fig, width_ratios=[1.6,1,1])
    gs0.update(wspace=0.4, hspace=0)

    gs_a = gs0[:, 0]
    gs_b = gs0[1:, 1:]
    
    source_data = {}

    ax_echo = render_echo_chamber_plot(fig, gs_a, partisan_dimen, sd=source_data, sd_key="a")
    plots.add_subplot_label(ax_echo, "a", y=18)
    
    ax_activity = render_partisan_activity_plot(fig, gs_b, partisan_dimen, just_first_part=True, sd=source_data, sd_key="b")
    plots.add_subplot_label(ax_activity, "b", y=30, x=-40)

    # TODO Add label to partisan colorbar
    
    return fig, [], source_data