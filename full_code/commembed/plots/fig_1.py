"""Plotting code for Figure 1."""
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
    