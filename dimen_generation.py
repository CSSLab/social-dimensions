import numpy as np
import pandas as pd

import os

def load_embedding():
    embedding_folder = os.path.join(os.path.dirname(__file__), 'embedding')
    vectors_path = os.path.join(embedding_folder, "vectors.tsv")
    metadata_path = os.path.join(embedding_folder, "metadata.tsv")
    meta = pd.read_csv(metadata_path, sep='\t', header=None)

    meta.columns = meta.iloc[0]
    meta = meta.reindex(meta.index.drop(0))

    meta.set_index(meta.columns[0], inplace=True)

    vectors = pd.read_csv(vectors_path, sep='\t', header=None)
    vectors.set_index(meta.index, inplace=True)
    vectors = vectors.divide(np.linalg.norm(vectors.values, axis=1), axis='rows')

    return vectors, meta

def score_embedding(vectors, dimensions):
    columns = {}

    for name, data in dimensions:
        columns[name] = np.dot(vectors.values, data["vector"] / np.linalg.norm(data["vector"]))

    return pd.DataFrame(columns, index=vectors.index)


# cosine similarity of all vectors
def cosine_similarity(vectors):
    # normalize vectors
    vectors = vectors.divide(np.linalg.norm(vectors.values, axis=1), axis='rows')
    # dot
    sims = np.dot(vectors.values, vectors.values.T)

    return sims


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
