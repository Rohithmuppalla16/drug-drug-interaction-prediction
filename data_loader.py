import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling


# --------------------------------
# VALID SMILES CHECK
# --------------------------------
def is_valid_smiles(smiles):
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except:
        return False


# --------------------------------
# LOAD DDI GRAPH
# --------------------------------
def load_ddi_graph():

    ddi = pd.read_csv(
        "https://raw.githubusercontent.com/liiniix/BioSNAP/master/ChCh-Miner/ChCh-Miner_durgbank-chem-chem.tsv",
        sep="\t"
    )

    ddi.rename(columns={
        "Drug1": "src",
        "Drug2": "dst"
    }, inplace=True)

    return ddi


# --------------------------------
# LOAD SMILES DATA
# --------------------------------
def load_smiles():

    smiles_df = pd.read_csv(
        "https://raw.githubusercontent.com/sshaghayeghs/molSMILES/main/structure%20links%202.csv"
    )

    smiles_df = smiles_df[
        ["DrugBank ID", "SMILES"]
    ].dropna()

    smiles_df["valid"] = smiles_df[
        "SMILES"
    ].apply(is_valid_smiles)

    smiles_df = smiles_df[
        smiles_df["valid"]
    ]

    return smiles_df


# --------------------------------
# CREATE GRAPH DATA
# --------------------------------
def create_pyg_graph(features, ddi_graph):

    nodes = np.unique(ddi_graph.values)

    node_map = {
        node: i for i, node in enumerate(nodes)
    }

    src = []
    dst = []

    for _, row in ddi_graph.iterrows():

        if row["src"] in node_map and \
           row["dst"] in node_map:

            src.append(node_map[row["src"]])
            dst.append(node_map[row["dst"]])

    edge_index = torch.tensor(
        [src + dst, dst + src],
        dtype=torch.long
    )

    x = torch.tensor(
        features[:len(nodes)],
        dtype=torch.float
    )

    return Data(
        x=x,
        edge_index=edge_index
    )


# =========================================
# SEMI-INDUCTIVE SPLIT (CORRECT VERSION)
# =========================================
def inductive_split(data, test_ratio=0.2, seed=42):

    torch.manual_seed(seed)

    num_nodes = data.num_nodes
    perm = torch.randperm(num_nodes)

    split = int(num_nodes * (1 - test_ratio))

    train_nodes = perm[:split]
    test_nodes = perm[split:]

    train_set = set(train_nodes.tolist())
    test_set = set(test_nodes.tolist())

    train_edges = []
    test_edges = []

    # Split edges
    for i in range(data.edge_index.size(1)):

        u = data.edge_index[0, i].item()
        v = data.edge_index[1, i].item()

        # Train edges: both nodes seen
        if u in train_set and v in train_set:
            train_edges.append([u, v])

        # Test edges: at least one unseen
        elif u in test_set or v in test_set:
            test_edges.append([u, v])

    train_edge_index = torch.tensor(train_edges).t().contiguous()
    test_edge_index = torch.tensor(test_edges).t().contiguous()

    # -----------------------------
    # Negative Sampling
    # -----------------------------
    train_neg = negative_sampling(
        edge_index=train_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=train_edge_index.size(1)
    )

    test_neg = negative_sampling(
        edge_index=data.edge_index,   # important
        num_nodes=num_nodes,
        num_neg_samples=test_edge_index.size(1)
    )

    # -----------------------------
    # Create Labels
    # -----------------------------
    train_edge_label_index = torch.cat(
        [train_edge_index, train_neg], dim=1
    )

    train_edge_label = torch.cat([
        torch.ones(train_edge_index.size(1)),
        torch.zeros(train_neg.size(1))
    ])

    test_edge_label_index = torch.cat(
        [test_edge_index, test_neg], dim=1
    )

    test_edge_label = torch.cat([
        torch.ones(test_edge_index.size(1)),
        torch.zeros(test_neg.size(1))
    ])

    # Train graph (message passing only on train edges)
    train_data = Data(
        x=data.x,
        edge_index=train_edge_index,
        edge_label_index=train_edge_label_index,
        edge_label=train_edge_label
    )

    # Test graph uses train graph for propagation
    test_data = Data(
        x=data.x,
        edge_index=train_edge_index,
        edge_label_index=test_edge_label_index,
        edge_label=test_edge_label
    )

    return train_data, test_data


# =========================================
# TRANSDUCTIVE SPLIT
# =========================================
def split_data_multi(data):

    transform = RandomLinkSplit(
        num_val=0.2,
        num_test=0.2,
        is_undirected=True
    )

    train_data, val_data, test_data = transform(
        Data(x=data.x, edge_index=data.edge_index_I)
    )

    # Attach second kernel manually
    train_data.edge_index_D = data.edge_index_D
    val_data.edge_index_D = data.edge_index_D
    test_data.edge_index_D = data.edge_index_D

    return train_data, val_data, test_data
# --------------------------------
# CREATE MULTI-KERNEL GRAPH
# --------------------------------
def create_multi_kernel_graph(features, ddi_graph):

    nodes = np.unique(ddi_graph.values)

    node_map = {node: i for i, node in enumerate(nodes)}

    edges = []

    for _, row in ddi_graph.iterrows():
        if row["src"] in node_map and row["dst"] in node_map:
            u = node_map[row["src"]]
            v = node_map[row["dst"]]
            edges.append((u, v))

    # --------------------------------
    # SPLIT INTO TWO KERNELS
    # --------------------------------
    np.random.shuffle(edges)

    split = len(edges) // 2

    edges_I = edges[:split]   # kernel 1
    edges_D = edges[split:]   # kernel 2

    def build_edge_index(edge_list):
        src = [e[0] for e in edge_list]
        dst = [e[1] for e in edge_list]

        return torch.tensor(
            [src + dst, dst + src],
            dtype=torch.long
        )

    edge_index_I = build_edge_index(edges_I)
    edge_index_D = build_edge_index(edges_D)

    x = torch.tensor(features[:len(nodes)], dtype=torch.float)

    return Data(
        x=x,
        edge_index_I=edge_index_I,
        edge_index_D=edge_index_D
    )