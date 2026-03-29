import numpy as np
import deepchem as dc


# -----------------------------
# MORGAN FEATURES
# -----------------------------
def morgan_features(smiles_list):

    featurizer = dc.feat.CircularFingerprint(
        size=1024,
        radius=2
    )

    features = []

    for sm in smiles_list:
        features.append(
            featurizer.featurize(sm)[0]
        )

    return np.array(features)


# -----------------------------
# LOAD LLM EMBEDDINGS
# -----------------------------
def load_llm_embeddings(path):

    import pandas as pd

    df = pd.read_csv(path)

    embeddings = df.iloc[:, 1:].values

    return embeddings


# -----------------------------
# FEATURE FUSION
# -----------------------------
def fuse_features(feature_list):

    return np.concatenate(
        feature_list,
        axis=1
    )