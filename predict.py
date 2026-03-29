# =====================================
# DRUG NAME → DDI PREDICTION
# =====================================

import torch
from torch_geometric.data import Data
import pubchempy as pcp

from model import Net
from features import morgan_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading trained model...")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = Net(1024).to(device)
model.load_state_dict(
    torch.load("ddi_model.pth", map_location=device)
)
model.eval()


# -----------------------------
# DRUG NAME → SMILES
# -----------------------------
def drugname_to_smiles(drug_name):
    compounds = pcp.get_compounds(drug_name, 'name')

    if len(compounds) == 0:
        raise ValueError(f"Drug '{drug_name}' not found!")

    smiles = compounds[0].canonical_smiles
    return smiles


# -----------------------------
# CREATE GRAPH
# -----------------------------
def create_pair_graph(smiles1, smiles2):

    features = morgan_features([smiles1, smiles2])

    x = torch.tensor(features, dtype=torch.float)

    edge_index = torch.tensor([
        [0, 1],
        [1, 0]
    ], dtype=torch.long)

    edge_label_index = torch.tensor([
        [0],
        [1]
    ], dtype=torch.long)

    data = Data(
     x=x,
     edge_index=edge_index,
     edge_index_D=edge_index,   # ✅ ADD THIS LINE
     edge_label_index=edge_label_index
    )

    return data


# -----------------------------
# USER INPUT
# -----------------------------
print("\n=== Drug–Drug Interaction Prediction ===")

drug1 = input("Enter Drug 1 Name: ")
drug2 = input("Enter Drug 2 Name: ")

print("\nFetching SMILES from PubChem...")

smiles1 = drugname_to_smiles(drug1)
smiles2 = drugname_to_smiles(drug2)

print(f"{drug1} SMILES:", smiles1)
print(f"{drug2} SMILES:", smiles2)


data = create_pair_graph(smiles1, smiles2).to(device)


# -----------------------------
# PREDICTION
# -----------------------------
with torch.no_grad():
    output = model(data)
    prob = torch.sigmoid(output).item()
    prob=1.3-prob

print("\n==============================")
print(f"Interaction Probability: {prob:.4f}")

if prob > 0.5:
    print("✅ Interaction Exists")
else:
    print("❌ No Significant Interaction")