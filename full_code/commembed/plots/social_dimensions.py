"""Plotting code for social dimension figures."""
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
from matplotlib.colors import Normalize
#import commembed.data as data
#spark = data.spark_context()
import re


matplotlib.rcParams['mathtext.fontset'] = 'custom'

embedding = load_embedding('reddit', 'master')
dimen_list = dimens.load_dimen_list('final')
scores = dimens.score_embedding(embedding, dimen_list)
scores_nonpctl = scores

from tsmoothie.smoother import LowessSmoother

from matplotlib.patches import Polygon
from matplotlib import colors


cluster_rename_map = {
    "US cities and sports": "USA",
    "Non-US cities and places": "Non-USA",
    "Cars and motorcycles": "Cars",
    "Fitness and self-improvement": "Self-improvement",
    "Porn (heterosexual)": "Pornography",
    "Crytocurrencies": "Cryptocurrencies",

    "General interest": "General interest$^†$",
    "Gaming": "Gaming$^†$"
}

def many_densities_plot(dimens_to_plot, clustering_name="reddit_unmerged", percentilize=True, export_raw_data=False, single_column=False):
    
    clustering = load_clustering(clustering_name)

    cluster_rename = cluster_rename_map
    if single_column:
        cluster_rename["Recreational drugs"] = "Drugs"

    clustering["cluster_name"] = clustering["cluster_name"].map(lambda x: cluster_rename.get(x, x))

    clustering_copy = clustering.copy()
    clustering_copy["cluster_id"] = 999
    all_name = "All\\ communities" if not single_column else "All"
    clustering_copy["cluster_name"] = "${\\bf %s}$" % all_name
    clustering_with_all_communities = pd.concat([clustering, clustering_copy])

    # percentileize scores
    if percentilize:
        #scores = scores_nonpctl.apply(lambda x: np.digitize(x, np.percentile(x, np.arange(0, 100))))
        scores = scores_nonpctl.apply(lambda x: 100 * np.argsort(np.argsort(x)) / len(x))
    else:
        scores = scores_nonpctl
        
    # include all communities row?
    include_all_communities_row = True
    if include_all_communities_row:
        clustering_to_plot = clustering_with_all_communities
    else:
        clustering_to_plot = clustering
        
    cluster_id_name_size = clustering_to_plot.groupby('cluster_id').apply(lambda x: (x.name, x["cluster_name"].iloc[0], len(x)))
    cluster_id_name_size = sorted(cluster_id_name_size, key=lambda x: x[2])

        
    cluster_names = [x[1] + " (%d)" % x[2] for x in cluster_id_name_size]
                         
    scatter_outliers_only = True
    show_density_estimate = True
    label_outliers =  not percentilize
    figure_size =  (9, (8/19.0)*len(cluster_names))

    if single_column:
        figure_size = (4.5, (8/19.0)*len(cluster_names))

    # end configuration

    plt.rcParams["image.composite_image"] = False

    to_plot = scores[dimens_to_plot]
    to_plot = to_plot.join(clustering_to_plot, on='community', how='inner')

    to_plot_nonpctl = scores_nonpctl[dimens_to_plot]
    to_plot_nonpctl = to_plot_nonpctl.join(clustering, on='community', how='inner')

    band_width = 0.1
    annotation_offset = 0.2
    density_n_bins = 40
    density_scaling_factor = 40 if percentilize else 0.25
    smooth_densities = percentilize
    
    smoother = LowessSmoother(smooth_fraction=0.2, iterations=1)
    
    cluster_ids = [x[0] for x in cluster_id_name_size]
    cluster_id_to_y = {cid: y for y, cid in enumerate(cluster_ids)}

    hide_outliers_for_clusters = ["Pornography"]
    hide_outliers_for_clusters = [clustering_to_plot[clustering_to_plot["cluster_name"] == cname]["cluster_id"].iloc[0] \
                                  for cname in hide_outliers_for_clusters]

    to_plot["y"] = to_plot["cluster_id"].map(cluster_id_to_y)# + (np.random.normal(size=len(to_plot)) * band_width) - (band_width / 2)

    fig, axs = plt.subplots(1, len(dimens_to_plot), figsize=figure_size, sharey=True)
    if not type(axs) is np.ndarray:
        axs = [axs]

    plt.subplots_adjust(wspace=0, hspace=0)

    glossary = []
    def annotate_outlier(ax, name, x, y, cluster_id, outliertype='minima',color=None):
        placement = {"minima":"top","maxima":"bottom"}[outliertype]
        
        glossary.append(name)
        
        ha = ('right' if outliertype == 'minima' else 'left')
        #ha = 'center'
        
        ax.annotate(name, xy=(x, y), size=8, ha=ha,
                    va=placement,
                    xytext=(x, y + annotation_offset * (-1 if placement == 'top' else 1)),
                   arrowprops=dict(color=color, arrowstyle="->"), c=color, zorder=999)

    metadata_rows = []
    

    cmap_name = {
        #"partisan": "coolwarm",
        #"gender": "PuOr",
        #"age": "PiYG"
    }

    for i, dimen in enumerate(dimens_to_plot):

        ax = axs[i]
        ax.set_frame_on(False)
        ax.tick_params(axis='both', length=0)
        ax.set_title(axis_name(dimen))
        
        
        density_bottoms = np.array(range(0, len(cluster_names)))
        tick_labels = []

        if i == 0:
            y_tick_offset = 0.2
            ax.set_yticks(density_bottoms + y_tick_offset)
            ax.set_yticklabels(cluster_names)
            ax.tick_params(axis='y', which='major', pad=25 if not single_column else 10)
            
        for b in density_bottoms:
            ax.axhline(b, color='#aaaaaa', linewidth=1, zorder=(-2 * b))


        ax.set_ylim(-0.1 if percentilize else -0.5, len(cluster_names) - 0.2)

        render_x = to_plot[dimen]
        x = to_plot_nonpctl[dimen]
        mean = np.mean(x)
        max_deviation = np.std(x)*3# np.percentile(np.abs(x - mean), 99)
        std_dev = np.std(x)
        render_thresh = std_dev * 2 if not percentilize else 0

        # cluster colormap
        cmap = axis_colormap(dimen)
        #if dimen == "partisan":
        #    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#3A4CC0","#F7F7F6","#B30326"])
        cmap = matplotlib.cm.ScalarMappable(cmap=cmap)
        cmap.set_clim(mean - max_deviation, mean + max_deviation)
        
        if percentilize:
            # Need to generate colormap mapping percentiles to raw scores
            
            pctl_scores = scores[dimen].sort_values()
            raw_scores = (scores_nonpctl[dimen][pctl_scores.index] - (mean - max_deviation)) / (max_deviation * 2)
            
            class PercentileNormalize(colors.Normalize):
                def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
                    self.vcenter = vcenter
                    colors.Normalize.__init__(self, vmin, vmax, clip)

                def __call__(self, value, clip=None):
                    #x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
                    x, y = pctl_scores, raw_scores
                    return np.ma.masked_array(np.interp(value, x, y))
            
            cmap = matplotlib.cm.ScalarMappable(cmap=axis_colormap(dimen), norm=PercentileNormalize())

        means = to_plot.groupby("cluster_name").agg({dimen: [np.mean, np.var, "count"]})
        means.columns = means.columns.droplevel(0)
        means['mean'] = (means['mean'] - np.mean(x)) / std_dev
        metadata_rows.extend([[dimen, cluster_name, row['mean'], row['var'], row['count']] for cluster_name, row in means.iterrows()])


        # density estimate
        if show_density_estimate:
            # bin starts
            density_bins = np.linspace(np.amin(render_x), np.amax(render_x), density_n_bins)
            bin_width = (density_bins[1]-density_bins[0])
            # add start and end bin for aesthetic reasons
            #density_bins = np.insert(density_bins, 0, density_bins[0]-bin_width)
            #density_bins = np.insert(density_bins, len(density_bins), density_bins[-1]+bin_width)
            
            # the bin midpoints are where the line is actually plotted
            bin_midpoints = density_bins+bin_width/2
            bin_midpoints = bin_midpoints[:-1]
            
            xlim_padding = 10 if percentilize else 0.05
            if single_column:
                xlim_padding = 0
            ax.set_xlim(np.amin(bin_midpoints) - xlim_padding, np.amax(bin_midpoints) + xlim_padding)
            
            density_bg_n_points = 500
            density_bg_x = np.linspace(np.amin(render_x), np.amax(render_x), density_bg_n_points)
            
            density_bg = cmap.to_rgba(density_bg_x).reshape((1, len(density_bg_x), 4))

            z_order = 0
            
            groupbys = to_plot.groupby("cluster_id")
            for cluster_id in cluster_ids:
                rows = groupbys.get_group(cluster_id)
                hist, bin_edges = np.histogram(rows[dimen], bins=density_bins, density=True)
                hist_orig = hist
                
                if smooth_densities:
                    smoother.smooth(hist)
                    hist = smoother.smooth_data[0]

                baseline = cluster_id_to_y[cluster_id]
                hx = bin_midpoints
                hy = baseline + (hist * density_scaling_factor)
                hy_orig = baseline + (hist_orig * density_scaling_factor)

                line,  = ax.plot(hx, hy, c='black', alpha=0.5, linewidth=1,zorder=z_order,solid_capstyle='butt')
                
                #if smooth_densities:
                #    ax.bar(hx, hist_orig * density_scaling_factor, width=2, color='black', alpha=0.2, linewidth=0.5,zorder=z_order,bottom=baseline)

                xmin, xmax, ymin, ymax = np.amin(bin_midpoints), np.amax(bin_midpoints), baseline, np.amax(hy)
                im = ax.imshow(density_bg, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                         origin='lower', zorder=z_order-1, interpolation='bilinear')

                xy = np.column_stack([hx, hy])
                xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
                clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
                ax.add_patch(clip_path)
                im.set_clip_path(clip_path)
                z_order -= 2


        to_scatter = to_plot
        if scatter_outliers_only:
            to_scatter = to_scatter[np.abs(x - mean) >= render_thresh]
        if not percentilize:
            ax.scatter(to_scatter[dimen], to_scatter["y"], alpha=0.2, c=cmap.to_rgba(to_scatter[dimen]),
                       facecolors='none', marker='|')

        
        ax.axvline(mean if not percentilize else 50, color='#333333', linestyle='--', linewidth=1)

        if label_outliers:
            minima_indices = to_plot.groupby("cluster_id").apply(lambda x: x.index[np.argsort(x[dimen].values)[-1]])
            maxima_indices = to_plot.groupby("cluster_id").apply(lambda x: x.index[np.argsort(x[dimen].values)[0]])
            
            for cid in cluster_ids:
                if cid in hide_outliers_for_clusters:
                    continue
                    
                to_plot_for_cluster = to_plot[to_plot["cluster_id"] == cid]
                
                argsort_idxs = np.argsort(to_plot_for_cluster[dimen].values)
                maxima = to_plot_for_cluster.index[argsort_idxs[-1]]
                minima = to_plot_for_cluster.index[argsort_idxs[0]]
            
                for outliertype, community in [("minima", minima), ("maxima", maxima)]:
                    if dimen.endswith("_neutral") and outliertype == "minima":
                        continue # Do not show left-side outliers on neutral plots

                    val = x[community]
                    val_for_color = mean + np.sign(val - mean) * np.maximum(std_dev*2, np.abs(val - mean))
                    color = cmap.to_rgba(val_for_color)
                    color_dim_factor = 6
                    fixed_alpha = 0.7
                    color = (color[0]/color_dim_factor, color[1]/color_dim_factor, color[2]/color_dim_factor, fixed_alpha)
                    
                    outlier_x = to_plot_for_cluster[dimen].loc[community]
                    outlier_y = to_plot_for_cluster["y"].loc[community]

                    annotate_outlier(ax, community, outlier_x, outlier_y, cid, outliertype=outliertype, color=color)

        ax.set_xlabel(axis_label(dimen, include_name=False), fontsize=7)
        if percentilize:
            ax.set_xticks([0,100])
            ax.tick_params(axis='x', labelsize=8)

            ticklabels = ax.get_xticklabels()
            ticklabels[0].set_ha("left")
            ticklabels[-1].set_ha("right")


    plt.subplots_adjust(wspace=0.15 if not single_column else 0.15)
    
    # Add colorbars for axes
    for i, dimen in enumerate(dimens_to_plot):
        ax = axs[i]
        cmap = axis_colormap(dimen)
        cmap = matplotlib.cm.ScalarMappable(cmap=cmap)
        cmap.set_clim(-3,3)
        
        cbar_width = 0.8
        cbar_height = 0.008
        cbar_y = -0.06 if single_column else -0.05
        x, y, width, height = (0.1, cbar_y - cbar_height, cbar_width, cbar_height)
        t = (ax.transAxes + fig.transFigure.inverted()).transform_point
        x_t, y_t = t((x, y))
        width_t, height_t = t((x + width, y + height))
        width_t -= x_t
        height_t -= y_t
        cax = plt.axes((x_t, y_t, width_t, height_t))

        cbar = fig.colorbar(cmap, cax=cax, orientation='horizontal')
        ctx = [-3, -2, -1, 0, 1, 2, 3]
        cbar.set_ticks(ctx)
        if single_column:
            cbar.set_ticklabels(["-3$\sigma$","","","0","","","3$\sigma$"])
        else:
            cbar.set_ticklabels(["%d$\sigma$" % d for d in ctx])
        cax.tick_params(axis='both', which='major', labelsize=8)
    
    
    if export_raw_data:
        assert percentilize
        assert clustering_name == "reddit_all_numbered"
        data_to_export = clustering[['cluster_name']].copy()
        data_to_export[[d + "_pctl" for d in dimens_to_plot]] = scores[dimens_to_plot]
        data_to_export[[d + "_raw" for d in dimens_to_plot]] = scores_nonpctl[dimens_to_plot]
        #data_to_export["over18"] = embedding.metadata["over18"]
        #data_to_export["over18"] = (embedding.metadata["over18"] == "True").astype(int)
        #print(np.sum(~((embedding.metadata["over18"] == "True") | (embedding.metadata["over18"] == "False"))))
        data_to_export = data_to_export.rename(columns={"edginess_pctl":"edgy_pctl", "edginess_raw":"edgy_raw"})
        data_to_export = data_to_export.rename(columns={"partisan_pctl":"partisan_pctl", "partisan_raw":"partisan_raw"})
        data_to_export = data_to_export.rename(columns={"partisan_neutral_pctl":"partisan_neutral_pctl", "partisan_neutral_raw":"partisan_neutral_raw"})
        data_to_export.to_csv("~/research/commembed/source_data/fig_2.csv")
        
    return fig, glossary

def print_skews_and_variances(pairs=None, clustering_name="reddit_unmerged", bimodal_clusters=[]):
    #
    # Calculate skew and variance figures for text
    #

    clustering = load_clustering(clustering_name)

    single_skew_clusters = [(cluster, dimen) for cluster in clustering["cluster_name"].unique() for dimen in ["age", "gender", "partisan"]]

    percentile_threshold = 20

    scores_pctl = scores_nonpctl.apply(lambda x: 100 * np.argsort(np.argsort(x)) / len(x))
    scores_z = scores_nonpctl.apply(lambda x: (x-np.mean(x)) / np.std(x))
    cscores_pctl = scores_pctl.join(clustering, on='community', how='inner')
    cscores_z = scores_z.join(clustering, on='community', how='inner')
        
    skew_rows = []
        
    for cluster, axis in single_skew_clusters:
        rows_pctl = cscores_pctl[cscores_pctl["cluster_name"] == cluster]
        rows_z = cscores_z[cscores_z["cluster_name"] == cluster]
        x_pctl = rows_pctl[axis]
        x_z = rows_z[axis]
        mean_z = np.mean(x_z)
        
        less, more = axis_less_more(axis)
        modifier_word = less if mean_z < 0 else more
        
        pctl_text = ""
        if mean_z < 0:
            frac = np.sum(x_pctl < percentile_threshold) * 100.0 / len(x_pctl)
            pctl_text = "%.0f%% are below the %dth percentile" % (frac, percentile_threshold)
            desc = "%s communities skew %s (%s)" % (cluster, modifier_word, pctl_text)
            skew_rows.append([cluster, axis, modifier_word, frac, percentile_threshold, mean_z, desc])
        else:
            frac = np.sum(x_pctl > (100-percentile_threshold)) * 100.0 / len(x_pctl)
            pctl_text = "%.0f%% are above the %dth percentile" % (frac, 100-percentile_threshold)
            desc = "%s communities skew %s (%s)" % (cluster, modifier_word, pctl_text)
            skew_rows.append([cluster, axis, modifier_word, frac, 100-percentile_threshold, mean_z, desc])
        
        
    skew_df = pd.DataFrame(skew_rows, columns=["cluster", "axis", "word", "frac", "pctl", "mean_z", "desc"]).set_index(["cluster", "axis"])

    pd.set_option('display.max_colwidth', None)
    if pairs is None:
        display(skew_df.sort_values("frac", ascending=False))
    else:
        display(skew_df.loc[pairs])

    bimodal_window = 20

    for cluster, axis in bimodal_clusters:
        rows_pctl = cscores_pctl[cscores_pctl["cluster_name"] == cluster]
        x_pctl = rows_pctl[axis]

        total_outside = np.sum(x_pctl < bimodal_window) + np.sum(x_pctl > (100 - bimodal_window))
        frac = total_outside * 100.0 / len(x_pctl)
        
        print("%s communities exhibit a bimodal distribution on the %s axis (%.0f%% are below the %dth percentile or above the %dth percentile)" \
            % (cluster, axis_name(axis), frac, bimodal_window, 100 - bimodal_window))
        


def render_distribution(fig, gs, selected_dimension, glossary=[], subplot_label=True, sd=None, sd_key=None):

    ax = fig.add_subplot(gs)
    dimen_data = [d[1] for d in dimen_list if d[0] == selected_dimension][0]
    
    scores_n = scores.apply(lambda x: (x-np.mean(x))/np.std(x), axis=0)
    x = scores_n[selected_dimension]
    norm = Normalize(-3, 3)
    
    #ks_statistic, p_value = kstest(x, 'norm')
    #print(ks_statistic, p_value)
    
    # show_gaussian = False
    # if show_gaussian:
        
    #     x_gauss = np.linspace(-6, 6, 100)
    #     gauss = stats.norm.pdf(x_gauss)
    #     ax.plot(x_gauss, gauss * 100)

    if sd is not None:
        sd[sd_key] = scores_n[[selected_dimension]]

    n, bins, patches = ax.hist(scores_n[selected_dimension], log=False, bins=100)
    #ax.yaxis.tick_right()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    #x = scores[selected_dimension]
    #mean = np.mean(x)
    #max_deviation = np.std(scores[selected_dimension])*3
    #norm = Normalize(mean - max_deviation, mean + max_deviation)
    
    # set a color for every bar (patch) according 
    # to bin value from normalized min-max interval
    cmap = axis_colormap(selected_dimension)
    for bin, patch in zip(bins, patches):
        color = cmap(norm(bin))
        patch.set_facecolor(color)
    
    ax.set_ylabel("Number of communities")
    ax.set_xlabel(axis_label(selected_dimension))
    
    def render_nice_line(x, label, y):
        height = y
            
            
        ha = 'left'
        x_offset = x + 0.005
        text_offset = 1.5
        if (x > -0.2 and x < 0) or x > 0.2:
            ha = 'right'
            x_offset = x - 0.005
            text_offset = -1.5
        
        #x = x - (bins[1]-bins[0])*0.5
        color = '#777777'
        textcolor = 'black'
        ax.plot([x, x], [0, height], color=color, linestyle='--', linewidth=1)
        ax.annotate(label, (x_offset, height), xytext=(text_offset, 0), textcoords='offset points',ha=ha, va='top', rotation=0,color=textcolor)
        
        
    # render examples     
    #sorted_scores = scores.sort_values(selected_dimension)
    #example_x = np.linspace(np.amin(sorted_scores[selected_dimension]), np.amax(sorted_scores[selected_dimension]), 18)
    #for ex in example_x:
    #    indx = np.argmax((sorted_scores[selected_dimension] >= ex).values)
    #    render_nice_line(sorted_scores[selected_dimension].iloc[indx], sorted_scores.index[indx])
        
        
    actual_seed = list([d[1] for d in dimen_list if d[0] == selected_dimension][0]['seeds'][0])
 
    cities = ["hillaryclinton", "The_Donald"] if selected_dimension.startswith("partisan") else []
    
    #render_nice_line(np.mean(scores.loc[dimen_data["negative_comms"], selected_dimension]), "Average of pairs' negative sides")
    #render_nice_line(np.mean(scores.loc[dimen_data["positive_comms"], selected_dimension]), "Average of pairs' positive sides")
    
    if selected_dimension.endswith("_neutral"):
        y = [50, 100] if len(cities) == 0 else [50,100,150,200]
    else:
        y = [100, 100] if len(cities) == 0 else [300,250,200,250]
    
    for i, comm in enumerate(sorted(actual_seed+cities, key=lambda x: scores_n.loc[x,selected_dimension])):
        render_nice_line(scores_n.loc[comm, selected_dimension], comm, y[i])
        glossary.append(comm)
        
    fig.canvas.draw()

    labels = [item.get_text() + "$\sigma$" if item.get_text() != "0" else item.get_text() for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels)
    
    if subplot_label:
        plots.add_subplot_label(ax, "c", y=20)

all_word_data = pd.read_parquet(data.DATA_PATH+"/top_words_for_plotting.parquet")
political_word_data = pd.read_parquet(data.DATA_PATH+"/top_words_political_only.parquet")

def render_extreme_subs(fig, gs, selected_dimension, glossary=[], subplot_label=True, political_words=False, sd=None, sd_key=None):

    # cluster colormap
    mean = np.mean(scores[selected_dimension])
    max_deviation = np.std(scores[selected_dimension])*3
    cmap = axis_colormap(selected_dimension)
    cmap = cm.ScalarMappable(cmap=cmap)
    cmap.set_clim(mean - max_deviation, mean + max_deviation)

    ax = fig.add_subplot(gs)
    
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if not political_words:
        ax.spines['bottom'].set_visible(False)
    ax.spines['bottom'].set_bounds(-7, 7)

    ax.spines['left'].set_visible(False)

    draw_little_numbers = not political_words
    
    n = 6 if political_words else 8 # was 10
    to_plot = scores.sort_values(selected_dimension)
    to_plot = pd.concat([to_plot.iloc[0:n][::-1], to_plot[-n:]])
    labelsize=9    
    data = []
    colors = []
    norm = Normalize(0, 1)
    ys = list(range(0, int(len(to_plot)/2)))
    ys = ys+ys
    start_y = int(len(to_plot)/2) + 1
    ys = list(range(start_y, start_y+int(len(to_plot)/2))) + list(range(start_y, start_y+int(len(to_plot)/2)))+ys
    
    sd_rows = []

    for name, row in to_plot.iterrows():
        i = len(data)
        val = row[selected_dimension]
        y = ys[i]
        x = (val - np.mean(scores[selected_dimension]))/np.std(scores[selected_dimension])
        color = cmap.to_rgba(val)
        
        ha = "left"
        if x < 0:
            ha = "right"
        
        if draw_little_numbers:
            ax.annotate("%.2f" % val, (np.sign(x) * 0.15, y), c="white", ha=ha, va="center", fontsize=6)
        ax.annotate("r/"+name, (x, y), xytext=(5 * np.sign(x), 0), textcoords='offset points', c="black", ha=ha, va="center",fontsize=labelsize)
        data.append(x)
        colors.append(color)
        glossary.append(name)
        sd_rows.append([name,"community",x])
        
    # do words
    word_data = political_word_data if political_words else all_word_data
    word_dimen = selected_dimension if selected_dimension in word_data.columns else 'age'
    word_data = word_data.sort_values(word_dimen)
    
    ban_words = []
        
    word_data = word_data[~word_data["word"].isin(ban_words)]
    
    to_plot = pd.concat([word_data.iloc[0:n][::-1], word_data[-n:]]).set_index("word")
    
    xmax = np.amax(to_plot[word_dimen]-np.mean(scores[word_dimen]))
    anno_x = 17

    if political_words:
        ax.annotate("Communities", (anno_x, ys[len(data)-(n//2)]-(0.5 if (n%2)==0 else 1)), rotation=-90, color='#444444', va='center', annotation_clip=False)
    
    for name, row in to_plot.iterrows():
        i = len(data)
        val = row[word_dimen]
        y = ys[i]
        x = (val - np.mean(scores[word_dimen])) / np.std(scores[word_dimen])
        color = cmap.to_rgba(val)
        
        ha = "left"
        if x < 0:
            ha = "right"

        # Redact offensive words 
        name = re.sub(r"n.gger", "n-gger", name)
        name = re.sub(r"f.ggot", "f-ggot", name)
        name = re.sub(r"r.tard", "r-tard", name)
        
        addtl_offset = True#np.abs(x)<0.2
        
        if draw_little_numbers:
            ax.annotate("%.2f" % val, (np.sign(x) * 0.15, y), c="black", ha=ha, va="center", fontsize=6)
        ax.annotate(name, (x, y), xytext=((10 if addtl_offset else 5) * np.sign(x), 0), textcoords='offset points', c="black", ha=ha, va="center",fontsize=labelsize)
        data.append(x)
        colors.append(color)
        sd_rows.append([name,"word",x])
    
    if sd is not None:
        sd[sd_key] = pd.DataFrame(sd_rows, columns=["item","type","score"]).set_index("item")
        
    ax.barh(ys, data, color=colors)
    
    width = np.amax(data)-np.amin(data)
    
    ax.set_xlim(np.amin(data)-(width/2), np.amax(data)+(width/2))
    
    ax.set_xlabel(axis_label(selected_dimension, include_name=False))

    if political_words:
        ax.annotate("Words", (anno_x, ys[-(n//2)]-(0.5 if (n%2)==0 else 1)), rotation=-90, color='#444444', va='center', annotation_clip=False)

        tx = [-6, -3, 0, 3,6]
        ax.set_xticks(tx)
        ax.set_xticklabels(["%d$\sigma$" % d if d != 0 else str(d) for d in tx])

        ax.axvline(0, linewidth=1, color='black', linestyle='--', alpha=0.5)
    
    if subplot_label:
        plots.add_subplot_label(ax, "d", x=-100, y=-10)
        
    return ax

def render_scatter(fig, gs):
    ax = fig.add_subplot(gs)
    
    ax.set_axis_off()
    
    to_plot = pd.read_csv('fig_1_clustering_projection.csv')
    #print(to_plot.shape)

    color_by_cluster = True
    c = None
    cmap = None
    if color_by_cluster:
        c = to_plot['cluster_id']
        cmap = 'tab20b'
    else:
        c = np.vstack([to_plot["r"],to_plot["g"],to_plot["b"]]).T

    s = 1.5
    ax.scatter(-to_plot["x"], to_plot["y"], c=c, cmap=cmap,
                alpha=0.25,
            #marker=',',
            s=s,
            lw=0)
    ax.axis('equal')
    plots.add_subplot_label(ax, "a", x=-9, y=-16)
    
    return ax



def render_schematic(fig, gs, selected_dimension, hide_numbers=False, glossary=[]):
    gs = list(gs.subgridspec(1, 3))
    ax = fig.add_subplot(gs[0])
    axs = np.array([ax] + [fig.add_subplot(g,sharey=ax) for g in gs[1:]])
    if not hide_numbers:
        plots.add_subplot_label(axs[0], "b")
        
    seed_pair = ["r/democrats", "r/Conservative"]
    found_pair = ["r/askhillarysupporters", "r/AskTrumpSupporters"]
    
    n_random_points = 2

    interesting_pair = np.array([[0, 0.2], [0.6, 0.8]])
    other_interesting_pair = interesting_pair + [0.2, -0.2]
    
    distort_amt = 0.05
    interesting_pair[0, 0] -= distort_amt
    interesting_pair[0, 1] += distort_amt
    other_interesting_pair[0, 0] += distort_amt
    other_interesting_pair[0, 1] -= distort_amt
    
    direction = np.array([1,1])
    direction = direction / np.linalg.norm(direction)

    random_points = np.random.rand(n_random_points, 2)
    random_points = np.array([[0.10593992, 0.76416452], [0.04775787, 0.91524662]]) # good looking
    
    all_points = np.concatenate((random_points, interesting_pair, other_interesting_pair), axis=0)

    color = [sns.color_palette()[7]]

    def draw_arrow_between_pair(ax, pair, subdued=False):
        ax.arrow(pair[0, 0], pair[0, 1], pair[1, 0]-pair[0, 0], pair[1,1]-pair[0, 1],
                 length_includes_head=True, alpha=(0.1 if subdued else 1),
                head_width=0.07, head_length=0.07, color='#666666')

    def draw_all_arrows(ax, pairs):
        for i in range(0, pairs.shape[0]):
            for j in range(0, pairs.shape[0]):
                #if np.random.rand() < 0.9:
                #    continue
                if i != j:
                    draw_arrow_between_pair(ax, pairs[[i, j], :], subdued=True)

    def annotate_community(ax, coords, name, above=False):
        glossary.append(name)
        ax.annotate(name, coords, xytext=(4 if name == "r/Conservative" else 0, 4 if above else -4), textcoords='offset points', 
            #arrowprops=dict(arrowstyle="-",shrinkA=0),
            ha='center', va='bottom' if above else 'top')
        #ax.scatter([coords[0]],[coords[1]],c='red' if above else 'blue')

    for i, ax in enumerate(axs.flatten()):
        i = i + 1
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        spine_color = '#aaaaaa'
        ax.spines['bottom'].set_color(spine_color)
        ax.spines['top'].set_color(spine_color) 
        ax.spines['right'].set_color(spine_color)
        ax.spines['left'].set_color(spine_color)
        
        ax.margins(0.1)
        ax.axis('equal')
        ax.set_axis_off()
        
        if i == 3:
            # draw tangent lines for all communities
            for point_i in range(0, all_points.shape[0]):
                
                point = all_points[point_i, :]
                point_on_line = direction * np.dot(direction, point)
                d = point_on_line - point
                ax.arrow(point[0], point[1], d[0], d[1], alpha=0.1)

        pair1_color = color
        pair2_color = color
        focus_blue = "#3B4CC0"
        focus_red = "#B40426"
        subd_blue = "#84A7FC"
        subd_red = "#F08A6C"

        if i == 1:
            pair1_color = [focus_blue, focus_red]

        if i == 2:
            #pair1_color = [focus_blue, focus_red]#[subd_blue, subd_red]
            pair2_color = [focus_blue, focus_red]

        if i == 3:
            pair1_color = [focus_blue, focus_red]
            pair2_color = [focus_blue, focus_red]

        ax.scatter(random_points[:, 0], random_points[:, 1], c=color)
        ax.scatter(interesting_pair[:, 0], interesting_pair[:, 1], c=pair1_color)
        ax.scatter(other_interesting_pair[:, 0], other_interesting_pair[:, 1], c=pair2_color)

        if not hide_numbers:
            ax.annotate(str(i), [-0.25, 0.9], fontsize=10, bbox=dict(boxstyle='square',fc='#eeeeee',ec='white'))

        if i == 1:
            draw_arrow_between_pair(ax, interesting_pair)
            annotate_community(ax, interesting_pair[0, :], seed_pair[0])
            annotate_community(ax, interesting_pair[1, :], seed_pair[1], above=True)
        if i == 2:
            draw_arrow_between_pair(ax, other_interesting_pair)
            annotate_community(ax, other_interesting_pair[0, :], found_pair[0])
            annotate_community(ax, other_interesting_pair[1, :], found_pair[1], above=False)
        if i == 2:
            draw_all_arrows(ax, all_points)
        if i == 3:
            ax.arrow(0, 0, 0.85, 0.85,
                     length_includes_head=True, alpha=0.5,
                    head_width=0.3, head_length=0.14, color='#666666', width=0.1)
            ax.annotate(axis_name(selected_dimension), (0.5, 0.5), ha='center', va='center', rotation=45)
            
from matplotlib.transforms import Bbox

    
def render_fig1_regular(selected_dimension):
    
    glossary = []
    source_data = {}
            
    scale_factor = 0.8
    fig = plt.figure(figsize=(9*scale_factor,5.5*scale_factor))

    gs0 = gridspec.GridSpec(6, 9, figure=fig)
    gs0.update(wspace=0.5, hspace=0.5)

    gs_scatter = gs0[0:4, :3]
    gs_schematic = gs0[0:2, 3:]

    gs_word_freqs = gs0[2:, 5:8]
    gs_dist = gs0[3:, :5]

    render_distribution(fig, gs_dist, selected_dimension, glossary=glossary, sd=source_data,sd_key="c")
    ax_subs = render_extreme_subs(fig, gs_word_freqs, selected_dimension, glossary=glossary, political_words=True, sd=source_data,sd_key="d")
    ax_scatter = render_scatter(fig, gs_scatter)
    render_schematic(fig, gs_schematic, selected_dimension, glossary=glossary)

    plots.nudge_axis(ax_scatter, -0.04, 0.06)

    subs_shrink_amt = 0.05
    #plots.nudge_axis(ax_subs, 0, -0.004 - subs_shrink_amt)
    plots.shrink_axis(ax_subs, 0, subs_shrink_amt)
    
    #plt.savefig('fig-1-composite.pdf', bbox_inches = "tight")
    #plt.show()
    
    return fig, glossary, source_data  
    
    
def render_fig1_compact(selected_dimension):
    
    glossary = []
            
    fig = plt.figure(figsize=(13,3.5))

    gs0 = gridspec.GridSpec(15, 8, figure=fig)


    gs_word_freqs = gs0[:, 3:]
    gs_dist = gs0[1:-2,:4]

    render_distribution(fig, gs_dist, selected_dimension, glossary=glossary, subplot_label=False)
    ax = render_extreme_subs(fig, gs_word_freqs, selected_dimension, glossary=glossary, subplot_label=False)
    ax.patch.set_alpha(0)
    
    #plt.savefig('fig-1-composite.pdf', bbox_inches = "tight")
    #plt.show()
    
    return fig, glossary
    
def render_fig1_grid(selected_dimensions=["age", "gender", "partisan", "affluence"]):
    
    glossary = []
    
    n = len(selected_dimensions)
            
    fig = plt.figure(figsize=(10,2.8*n))

    gs0 = gridspec.GridSpec(16*n, 8, figure=fig)


    for i, selected_dimension in enumerate(selected_dimensions):
        start = i * 16
        end = (i + 1) * 16
        
        gs_word_freqs = gs0[start:(end-2), 3:]
        gs_dist = gs0[(start+1):(end-3),:4]

        render_distribution(fig, gs_dist, selected_dimension, glossary=glossary, subplot_label=False)
        ax = render_extreme_subs(fig, gs_word_freqs, selected_dimension, glossary=glossary, subplot_label=False)
        ax.patch.set_alpha(0)

    #plt.savefig('fig-1-composite.pdf', bbox_inches = "tight")
    #plt.show()
    
    return fig, glossary