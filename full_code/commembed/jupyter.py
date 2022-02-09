# To use in Jupyter:
# import sys
# sys.path.append("../")
# from commembed.jupyter import *
# %load_ext autoreload
# %autoreload 2

import collections
import numpy as np
import pandas as pd
import os
from . import dimens, linalg
from .dimens.validations import all_validations
import ipywidgets as widgets
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed

import matplotlib.colors
from matplotlib import cm
import colorcet as cc

def axis_name(dimension_name):
    name = dimension_name
    add_ness = "_neutral" in name
    add_b = "_b" in name
    name = name.replace('_neutral', '')
    name = name.split("_")[0]

    name = (name + "\\ B") if add_b else name
    name = (name + "-ness") if add_ness else name

    if dimension_name == "edginess":
        return "edgy"
    
    return name

def axis_less_more(dimension_name):
    labels = {
        "gender": ("masculine", "feminine"),
        "age": ("young", "old"),
        "partisan": ("left wing", "right wing"),
        "affluence": ("poor", "affluent"),
        "time": ("older", "newer"),
        "sociality": ("less social", "more social"),
        "edginess": ("kinder", "edgier")
    }
    
    return labels.get(dimension_name.split("_")[0], ("", ""))


def axis_label(dimension_name, include_name=True):

    less, more = axis_less_more(dimension_name)
    if dimension_name.endswith("_neutral"):
        if include_name:
            return "$\\bf{" + axis_name(dimension_name) + "}$"
        else:
            return ""
        
    if include_name:
        return "← "+less+"         $\\bf{" + axis_name(dimension_name) + "}$          "+more+" →"
    else:
        return "← "+less+"    "+more+" →"

def axis_quantile_names(dimension_name):
    less, more = axis_less_more(dimension_name)
    return [less.capitalize(), f"Leaning {less}", "Center", f"Leaning {more}", more.capitalize()]


def axis_colormap(dimension_name):
    if dimension_name.endswith("_neutral"):
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#aaaaaa","#aaaaaa","purple"])
    else:
        #cmap = cm.get_cmap('coolwarm')
        cmap = cm.get_cmap({"partisan": "coolwarm", "age":"PiYG", "gender": "PuOr"}.get(dimension_name.split("_")[0], "coolwarm"))
    return cmap

def save_paper_figure(fig_glossary_tuple, name, filetype='pdf'):
    fig = None
    glossary = None
    source_data = None

    if len(fig_glossary_tuple) == 3:
        fig, glossary, source_data = fig_glossary_tuple
    else:
        fig, glossary = fig_glossary_tuple
    
    if source_data:
        for key, df in source_data.items():
            df.to_csv('~/research/commembed/source_data/%s_%s.csv' % (name, key))

    # Generate glossary
    glossary = pd.DataFrame(index=list(set(glossary)))
    glossary.to_csv('/u/walleris/research/commembed/paper_resources/glossary_%s.txt' % name, header=False)
    
    # Save figure
    plt.savefig('/u/walleris/research/commembed/paper_resources/%s.%s' % (name, filetype), bbox_inches = "tight", dpi=300)

    tight_bbox_raw = fig.get_tightbbox(fig.canvas.get_renderer())
    width = tight_bbox_raw.x1 - tight_bbox_raw.x0
    height = tight_bbox_raw.y1 - tight_bbox_raw.y0
    print("Aspect ratio: %.4f" % (height/width))

    plt.close('all')
    print("Saved %s" % name)


def vector_columns(fmt="{c}"):
    return ', '.join([fmt.format(c = ("_%d" % d)) for d in range(0, 150)])

Embedding = collections.namedtuple("Embedding", "vectors metadata")

base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')

def load_embedding(dataset_name, embedding_name, load_metadata=True, filter_nsfw=False):
    embedding_folder = os.path.join(base_path, 'embeddings')
    vectors_path = os.path.join(embedding_folder, "%s-%s-vectors.tsv" % (dataset_name, embedding_name))
    metadata_path = os.path.join(embedding_folder, "%s-%s-metadata.tsv" % (dataset_name, embedding_name))
    meta = pd.read_csv(metadata_path, sep='\t', header=None)

    #if len(meta.columns) > 1:
    meta.columns = meta.iloc[0]
    meta = meta.reindex(meta.index.drop(0))
    #else:
    #    meta.columns = ["community"]

    meta.set_index(meta.columns[0], inplace=True)

    vectors = pd.read_csv(vectors_path, sep='\t', header=None)
    vectors.set_index(meta.index, inplace=True)
    vectors = vectors.divide(np.linalg.norm(vectors.values, axis=1), axis='rows')

    # Filter nsfw
    if filter_nsfw:
        vectors = vectors[meta["over18"] == "False"]
        meta = meta[meta["over18"] == "False"]

    return Embedding(vectors=vectors, metadata=meta)

def load_clustering(clustering_name):
    clustering = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'clusterings', clustering_name + '.csv'))
    clustering = clustering.set_index("name")
    return clustering


# See "020 Politics - prepare data" notebook
neutral_cutoffs = {
    "partisan_b": 0.49748743718592964,
    "partisan": 0.5125628140703518
}

def load_politics_z_df(partisan_dimen, political_activity_category="all", abs_value=False):
    import commembed.data as data
    spark = data.spark_context()

    embedding = load_embedding('reddit', 'master')
    dimen_list = dimens.load_dimen_list('final')
    scores = dimens.score_embedding(embedding, dimen_list)

    neutral_cutoff = neutral_cutoffs

    scores[partisan_dimen] = (scores[partisan_dimen]-np.mean(scores[partisan_dimen]))/np.std(scores[partisan_dimen])

    scores_filtered = scores
    scores_filtered = scores_filtered[scores_filtered[partisan_dimen+'_neutral'] > neutral_cutoff[partisan_dimen]]
    scores_filtered = scores_filtered[[partisan_dimen]].rename(columns={partisan_dimen:"partisan_dimen"})

    assert political_activity_category in ["left", "right", "all", "center"]

    if political_activity_category == "left":
        scores_filtered = scores_filtered[scores_filtered["partisan_dimen"] <= -1]
    elif political_activity_category == "right":
        scores_filtered = scores_filtered[scores_filtered["partisan_dimen"] >= 1]
    elif political_activity_category == "center":
        scores_filtered = scores_filtered[scores_filtered["partisan_dimen"] < 1]
        scores_filtered = scores_filtered[scores_filtered["partisan_dimen"] > -1]
        
    if abs_value:
        scores_filtered["partisan_dimen"] = np.abs(scores_filtered["partisan_dimen"])

    print("%d political subreddits selected" % len(scores_filtered))
    scores_df = spark.createDataFrame(scores_filtered.reset_index())
    scores_df.createOrReplaceTempView("scores")

    return scores_filtered, scores_df


def load_abs_z_df(partisan_dimen, political_activity_category="all"):
    import commembed.data as data
    spark = data.spark_context()

    embedding = load_embedding('reddit', 'master')
    dimen_list = dimens.load_dimen_list('final')
    scores = dimens.score_embedding(embedding, dimen_list)

    neutral_cutoff = neutral_cutoffs


    scores[partisan_dimen] = (scores[partisan_dimen]-np.mean(scores[partisan_dimen]))/np.std(scores[partisan_dimen])

    scores_filtered = scores

    if political_activity_category != "all_unfiltered":
        
        if partisan_dimen not in neutral_cutoff:
            print("!! Using non-partisan dimen %s; taking 10th percentile of neutral dimen" % partisan_dimen)
            neutral_cutoff[partisan_dimen] = np.percentile(scores[partisan_dimen+'_neutral'], 90)

        scores_filtered = scores_filtered[scores_filtered[partisan_dimen+'_neutral'] > neutral_cutoff[partisan_dimen]]
    
    scores_filtered = scores_filtered[[partisan_dimen]].rename(columns={partisan_dimen:"partisan_dimen"})

    assert political_activity_category in ["left", "right", "all", "center", "all_unfiltered"]

    if political_activity_category == "left":
        scores_filtered = scores_filtered[scores_filtered["partisan_dimen"] <= -1]
    elif political_activity_category == "right":
        scores_filtered = scores_filtered[scores_filtered["partisan_dimen"] >= 1]
    elif political_activity_category == "center":
        scores_filtered = scores_filtered[scores_filtered["partisan_dimen"] < 1]
        scores_filtered = scores_filtered[scores_filtered["partisan_dimen"] > -1]
        
    scores_filtered["partisan_dimen"] = np.abs(scores_filtered["partisan_dimen"])

    print("%d political subreddits selected" % len(scores_filtered))
    scores_df = spark.createDataFrame(scores_filtered.reset_index())
    scores_df.createOrReplaceTempView("scores")

    return scores_filtered, scores_df