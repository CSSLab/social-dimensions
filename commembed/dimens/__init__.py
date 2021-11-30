import sys
sys.path.append("../")
from .. import linalg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import pickle
import datetime


import os

def load_dimen_list(name):
    
    base_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    list_path = os.path.join(base_path, 'dimen_lists', name + '.txt')

    with open(list_path) as f:
        dimen_names = f.readlines()

    dimens = []

    for name in dimen_names:
        fname = name.strip() + '.pickle'
        dimen_path = os.path.join(base_path, 'dimens', fname)

        with open(dimen_path, 'rb') as handle:
            dimens.append((name.split('/')[-1].strip(), pickle.load(handle)))

    return dimens

def save_dimen(name, hash):
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dimens', name + '.pickle')
    with open(path, 'wb') as handle:
        pickle.dump(hash, handle)


def basic_dimension_report(embedding, name, dimen):

    vector = dimen["vector"]
    left_comms, right_comms = (dimen["left_comms"], dimen["right_comms"])
    
    dots = np.dot(embedding.vectors.values, vector / np.linalg.norm(vector))
    
    plt.figure(figsize=(12, 10))
    
    # Histogram
    ax = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax.set_title(name)
    ax.set_ylabel("Frequency")
    ax.set_xlabel(name)
    ax.hist(dots, bins=100)
    
    argsort = embedding.vectors.index[np.argsort(dots)]
    
    for comms, color in [(left_comms, 'g'), (right_comms, 'r')]:
        for i, comm in enumerate(comms):
            x = dots[embedding.vectors.index.get_loc(comm)]
            ax.axvline(x=x, color=color, linewidth=1)
            txt = ax.annotate(comm, (x, 200 - (i % 5) * 20), xytext=(0, 0),
                               textcoords='offset points',
                               horizontalalignment='center',
                               verticalalignment='bottom',
                              arrowprops=dict(arrowstyle="->"))
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
    
    for thresh in np.linspace(np.amin(dots), np.amax(dots), num=20):
        i = (np.argmax(dots[np.argsort(dots)] >= thresh))
        com = argsort[i]
        val = dots[embedding.vectors.index.get_loc(com)]
        txt = ax.annotate(com, (val, 0), xytext=(0, 20),
                           textcoords='offset points',
                           horizontalalignment='left',
                           verticalalignment='bottom', rotation=25,
                           arrowprops=dict(arrowstyle="->"))
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
    
    ax.axvline(x=np.mean(dots), color='k', linestyle='dashed', linewidth=1)

    # Highest subs
    ax2 = plt.subplot2grid((3, 2), (1, 0), rowspan=2)
    ax2.set_title("Communities with lowest %s" % name)
    
    ax3 = plt.subplot2grid((3, 2), (1, 1), rowspan=2)
    ax3.set_title("Communities with highest %s" % name)
    
    how_many_top = 50
    for communities, axis in [(embedding.vectors.index[np.argsort(dots)][-how_many_top:], ax3), (embedding.vectors.index[np.argsort(dots)][:how_many_top], ax2)]:
        y = np.arange(len(communities))
        x = [dots[embedding.vectors.index.get_loc(c)] for c in communities]
        
        axis.barh(y, x, align='center')
        axis.set_xlabel(name)
        axis.set_yticks(y)
        axis.set_yticklabels(communities)
    
    plt.tight_layout()
    plt.show()

def score_embedding(embedding, dimensions):
    columns = {}

    for name, data in dimensions:
        columns[name] = np.dot(embedding.vectors.values, data["vector"] / np.linalg.norm(data["vector"]))

    return pd.DataFrame(columns, index=embedding.vectors.index)

# Does not normalize dimension vector first.
def score_embedding_nonorm(embedding, dimensions):
    columns = {}

    for name, data in dimensions:
        columns[name] = np.dot(embedding.vectors.values, data["vector"])

    return pd.DataFrame(columns, index=embedding.vectors.index)

def score_vectors(vectors, dimensions):
    columns = {}

    print("Warning: input vectors must be normalized unless you WANT just the dot products. Think about it")

    for name, data in dimensions:
        columns[name] = np.dot(vectors.values, data["vector"] / np.linalg.norm(data["vector"]))

    return pd.DataFrame(columns, index=vectors.index)

def generate_random_dimension(embedding):
    # https://mathoverflow.net/a/26225
    vec = np.random.normal(size=embedding.vectors.shape[1])
    vec = vec / np.linalg.norm(vec)

    return {
        "method": "random",
        "type": "random",
        "vector": vec
    }

class DimenGenerator:
    
    def __init__(self, vectors):
        self.vectors = vectors

        self.name_mapping = {name.lower(): name for name in self.vectors.index}

        comm_names = list(self.vectors.index)
        cosine_sims = cosine_similarity(self.vectors)

        # Find each community's nearest neighbours
        ranks = cosine_sims.argsort().argsort()
        
        # Take n NNs
        nn_n = 10
        only_calculate_for = \
            (ranks > (len(comm_names) - nn_n - 2)) & \
            ~np.diag(np.ones(len(comm_names), dtype=bool))

        indices_to_calc = np.nonzero(only_calculate_for)

        index = []
        directions = []
        for i in range(0, len(indices_to_calc[0])):
            c1 = indices_to_calc[0][i]
            c2 = indices_to_calc[1][i]
            index.append((comm_names[c1], comm_names[c2]))
            directions.append(self.vectors.iloc[c2] - self.vectors.iloc[c1])

        print("%d valid directions, %d calculated." % (np.sum(only_calculate_for), len(directions)))
        self.directions_to_score = pd.DataFrame(index=pd.MultiIndex.from_tuples(index), data=directions)

    def generate_dimensions_from_seeds(self, seeds):
        return list(map(lambda x: self.generate_dimension_from_seeds([x]), seeds))

    def generate_dimension_from_seeds(self, seeds):
        
        seed_directions = self.vectors.loc[map(lambda x: x[1], seeds)].values - \
            self.vectors.loc[map(lambda x: x[0], seeds)].values

        seed_similarities = np.dot(self.directions_to_score, seed_directions.T)
        seed_similarities = np.amax(seed_similarities, axis=1)

        directions = self.directions_to_score.iloc[np.flip(seed_similarities.T.argsort())]

        # How many directions to take?
        num_directions = 10
        
        # make directions unique subreddits (subreddit can only occur once)
        ban_list = [s for sd in seeds for s in sd]
        i = -1 # to filter out seed pairs
        while (i < len(directions)) and (i < (num_directions + 1)):
            ban_list.extend(directions.index[i])
            
            l0 = directions.index.get_level_values(0)
            l1 = directions.index.get_level_values(1)
            directions = directions[(np.arange(0, len(directions)) <= i) | ((~l0.isin(ban_list)) & (~l1.isin(ban_list)))]

            i += 1

        # Add seeds to the top
        directions = pd.concat([pd.DataFrame(seed_directions, index=seeds), directions])
        
        direction_group = directions.iloc[0:num_directions]

        dimension = np.sum(direction_group.values, axis=0)

        return {
            "note": "generated from seed pairs",
            "seed": seeds,
            "vector": dimension,
            "left_comms": list(map(lambda x: x[0], direction_group.index)),
            "right_comms": list(map(lambda x: x[1], direction_group.index)),
        }

def load_copycat_dimen_list(dimen_list_name, embedding):
    """Duplicates dimen list that was created for a different embedding into target
    by recreating dimensions"""

    def reform_dimen_for_embedding(dimen_name, h, embedding):
        if dimen_name.endswith("_neutral"):
            vex = np.mean(embedding.vectors.loc[h["positive_comms"]].values+embedding.vectors.loc[h["negative_comms"]].values,axis=0)
        else:
            vex = np.mean(embedding.vectors.loc[h["positive_comms"]].values-embedding.vectors.loc[h["negative_comms"]].values,axis=0)
        vex = vex / np.linalg.norm(vex)
        return {"vector": vex}

        
    dimen_list = load_dimen_list(dimen_list_name)

    actual_dimen_list = [(dimen_name, reform_dimen_for_embedding(dimen_name, h, embedding)) for dimen_name, h in dimen_list]

    return actual_dimen_list

