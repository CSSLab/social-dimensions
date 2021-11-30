"""Plotting code for political polarization figures."""
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

def render_echo_chamber_community_plot(fig, gs, dimen_to_plot):
    data_suffix = "_filt_" + dimen_to_plot
    user_dists_table = "user_dists_bins=5_breadth=2_" + data_suffix

    path = os.path.join(data.DATA_PATH, \
        "echo_chamber_community_results_%s.parquet" % user_dists_table)
    result_community = pd.read_parquet(path)

    ax = fig.add_subplot(gs)
    n_subreddits = 25

    cmap = cm.ScalarMappable(cmap=axis_colormap("partisan"))
    cmap.set_clim(-2, 2)
    partisan_colors = cmap.to_rgba([-2,-1,0,1,2])
    partisan_colors[2] = [0.73,0.73,0.73,1]

    cmap = axis_colormap(dimen_to_plot)
    quantiles = axis_quantile_names(dimen_to_plot)

    to_plot = result_community[result_community["dimen"] == dimen_to_plot]
    to_plot = to_plot.sort_values("total_comments", ascending=False)
    to_plot = to_plot.iloc[0:5*n_subreddits]

    to_plot = to_plot.pivot(index="subreddit", columns="bin2", values="subreddit_avg_frac_bin2")
    to_plot["avg"] = np.sum(to_plot.columns.values * to_plot.values, axis=1)
    to_plot = to_plot.sort_values("avg")
    del to_plot["avg"]

    to_plot.plot.bar(ax=ax, stacked=True, color=partisan_colors, label=quantiles, width=0.4, legend=False)

    ax = plt.gca()
    ax.set_frame_on(False)
    plt.xticks(rotation=45, horizontalalignment="right")
    #ax.set_xticklabels([q.replace(' ','\n', 1) for q in quantiles], rotation='horizontal')
    ax.set_xlabel("")
    ax.set_ylabel("Avg. pct. of author activity")

    #plt.legend(labels=quantiles, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes, frameon=False)

    return ax


    
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

def render_abs_z_line_plot(fig, gs, partisan_dimen, metric="avg_abs_z"):
    
    ax = fig.add_subplot(gs)
    
    to_plot = pd.read_csv(os.path.join(data.DATA_PATH, "monthly_polarization_by_category_%s.csv" % partisan_dimen), index_col=0)

    avg_pivoted = to_plot.pivot(index="month", columns="category", values=metric)
    avg_pivoted = avg_pivoted[avg_pivoted.index > "2012"]
    avg_pivoted = avg_pivoted[avg_pivoted.index < "2019"]

    if metric == "avg_abs_z":
        before_time = avg_pivoted[avg_pivoted.index < "2016"]
        print("Abs z of all activity before 2012 = (min: %.4f max: %.4f)" % (np.amin(before_time["all"]), np.amax(before_time["all"])))
        
        idxmax = avg_pivoted["all"].idxmax()
        print("Abs z max: %s = %.4f" % (idxmax, avg_pivoted.loc[idxmax, "all"]))
        after_time = avg_pivoted[avg_pivoted.index > idxmax]
        print("Post-max min: %.4f" % np.amin(after_time["all"]))
    
        for c in ["center", "left", "right"]:
            print("Abs z %s nov 2015 -> nov 2016: %.4f -> %.4f" % (c, avg_pivoted.loc["2015-11", c], avg_pivoted.loc["2016-11", c]))

    if metric == "num_comments":
         #avg_pivoted = avg_pivoted[['center', 'left', 'right']]
         avg_pivoted['all'] = np.nan # hide this line

    colors = [
        [0, 0, 0, 1],
        [0.5 , 0.5, 0.5, 1],
        [0.2298057 , 0.29871797, 0.75368315, 1],
        [0.70567316, 0.01555616, 0.15023281, 1],
    ]

    avg_pivoted.plot.line(ax=ax, color=colors, legend=None)
    
    if metric == "num_comments":
        assert list(avg_pivoted.columns) == ["all", "center", "left", "right"]
        #ax.legend(loc='upper left', frameon=False, labels=["All", "Center", "Left", "Right"], ncol=1, handlelength=1)
    else:

        # Do annotated legend
        x_offset = -4
        y_offset = 7
        offsets = [
            (x_offset, -y_offset),
            (x_offset, y_offset),
            (x_offset, y_offset),
            (x_offset, -y_offset)
        ]
        for i, col in enumerate(avg_pivoted.columns):
            ax.annotate(str(col).capitalize(), (len(avg_pivoted)-1, avg_pivoted[col].iloc[-1]), xytext=offsets[i], textcoords='offset points',
                    xycoords='data', ha='right', va='center', color=colors[i],
                    weight='bold')
    
    plots.adjust_date_ticks(ax, avg_pivoted.index, do_y=False, include_extra_ticks=False)
    ax.set_xlabel("Month")

    ylabel = {"avg_abs_z": "Polarization", "num_comments": "Comments"}
    ax.set_ylabel(ylabel[metric])
    
    return ax

def render_polprob_ccdf(fig, gs, partisan_dimen, varying, political_activity_category='all'):
    
    ax = fig.add_subplot(gs)
    
    to_plot = pd.read_csv(os.path.join(data.DATA_PATH, "user_dynamic_polprob_ccdf_%s_%s_%s.csv" % (varying, partisan_dimen, political_activity_category)))
    to_plot = to_plot.set_index(["m1", "m2"])


    to_plot = to_plot.T
    to_plot.index = to_plot.index.astype(float)
    to_plot.plot.line(ax=ax, legend=False)
    
    if varying == 'sd_thresh':
        ax.set_xlabel("Polarization threshold")
        ax.set_ylabel("Frac. $|z_1|-|z_2| \\geq x$")

        #from matplotlib.ticker import FormatStrFormatter
        #ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    else:
        ax.set_xlabel("Comment threshold")
        ax.set_ylabel("Frac. $|z_1|-|z_2| \\geq 1$")
    
    if varying == 'sd_thresh':
        ax.legend(loc='upper right', frameon=False, prop={'size': 9}).set_title(None)
    # else:
    #     ax.get_legend().remove()


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



def render_proportion_extreme_activity_plot(fig, gs, partisan_dimen):
    
    ax = gs if isinstance(gs, matplotlib.axes.Axes) else fig.add_subplot(gs)

    activity_data = pd.read_parquet("/ada/data/reddit/parquet/%s_activity_data_12mo.parquet" % partisan_dimen)

    activity_data = activity_data.set_index("subreddit")
    
    comm_z_scores =(scores[partisan_dimen] - np.mean(scores[partisan_dimen]))/np.std(scores[partisan_dimen])
    activity_data["z_score"] = comm_z_scores
    activity_data["label"] = np.digitize(activity_data["z_score"], [-3, 3])

    # print num subreddits w/ label
    print("Proportion extreme plot: %d far-left communities; %d far-right communities" % (np.sum(comm_z_scores<-3), np.sum(comm_z_scores>3)))
    
    activity_data = activity_data.groupby(["month", "label"]).agg({"num_comments": "sum"})
    total_comments = activity_data.groupby(["month"]).agg({"num_comments": "sum"})
    
    activity_data = activity_data.reset_index().pivot(index="month", columns="label", values="num_comments")
    
    activity_data[[0, 1, 2]] = activity_data[[0, 1, 2]].divide(total_comments["num_comments"], axis=0)
    
    total_extreme = activity_data[0] + activity_data[2]
    print("Proportion extreme plot: " + ",".join(["%s: %s" % (d, total_extreme[d]) for d in ["2015-01", "2016-11"]]))
    
    cmap = cm.get_cmap('coolwarm')
    cmap = cm.ScalarMappable(cmap=cmap)
    cmap.set_clim(-3, 3)
    activity_data.plot.area(ax=ax, legend=False,color=cmap.to_rgba([-3,0,3]))
    
    ax.set_ylim(0, 1)
    ax.set_xlabel("Month")
    ax.set_ylabel("Proportion")
    
    plots.adjust_date_ticks(ax, activity_data.index, do_y=False, include_extra_ticks=True)


    return ax

def render_partisan_distribution_difference_plot(fig, gs, partisan_dimen):
    
    assert partisan_dimen == "partisan"

    ax = gs if isinstance(gs, matplotlib.axes.Axes) else fig.add_subplot(gs)

    dist_df = pd.read_csv(os.path.join(data.DATA_PATH,"12_deleted_dist_df.csv"), index_col=0)

    dist_df.plot.line(ax=ax)
    ax.legend(frameon=False)
    from matplotlib.ticker import AutoLocator
    ax.xaxis.set_major_locator(AutoLocator())
    ax.set_ylabel("Fraction")
    ax.set_xlabel(axis_label("partisan"))

    return ax
