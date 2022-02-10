"""Plotting code for Figure 2."""
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