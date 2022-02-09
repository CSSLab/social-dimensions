
import numpy as np
import pandas as pd

# cosine similarity of all vectors
def cosine_similarity(vectors):
    # normalize vectors
    vectors = vectors.divide(np.linalg.norm(vectors.values, axis=1), axis='rows')
    # dot
    sims = np.dot(vectors.values, vectors.values.T)

    return sims


# cosine similarity of all vectors w one
def cosine_similarity_single(vectors, dimen):
    # normalize vectors
    vectors = vectors.divide(np.linalg.norm(vectors.values, axis=1), axis='rows')
    dimen = dimen / np.linalg.norm(dimen)
    # dot
    sims = np.dot(vectors.values, dimen)

    return pd.Series(data=sims, index=vectors.index)