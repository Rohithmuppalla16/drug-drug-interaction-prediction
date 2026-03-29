# =====================================
# MAIN RUN FILE - DDI TRAINING
# Lightweight Competitive DDI Model
# =====================================

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import torch
import time

torch.manual_seed(42)

# ---------------------------------
# DEVICE
# ---------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Using device:", device)

# ---------------------------------
# IMPORT MODULES
# ---------------------------------
from model import Net
from train import train, test
from data_loader import (
    load_ddi_graph,
    load_smiles,
    create_multi_kernel_graph,
    split_data_multi
)
from features import morgan_features


# =====================================
# LOAD DATA
# =====================================
print("\nLoading data...")

ddi_graph = load_ddi_graph()
smiles_df = load_smiles()

# =====================================
# FEATURE GENERATION
# (Make sure fingerprint size=1024 in features.py)
# =====================================
print("Generating molecular features...")

features = morgan_features(
    smiles_df["SMILES"]
)


# =====================================
# GRAPH CREATION
# =====================================
print("Creating PyG graph...")

data = create_multi_kernel_graph(
    features,
    ddi_graph
)

train_data, val_data, test_data = split_data_multi(data)
print(train_data)
print(hasattr(train_data, "edge_index_D"))


# =====================================
# MODEL INITIALIZATION
# =====================================
print("Initializing model...")

model = Net(
    data.num_features
).to(device)

# ---------------------------------
# PARAMETER COMPARISON (NOVELTY PROOF)
# ---------------------------------
total_params = sum(p.numel() for p in model.parameters())
print("Total Trainable Parameters:", total_params)

print("Model Size (MB):", total_params * 4 / (1024**2))
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5
)
num_positive = (train_data.y == 1).sum().item()
num_negative = (train_data.y == 0).sum().item()

pos_weight = torch.tensor([num_negative / num_positive]).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
criterion = torch.nn.BCEWithLogitsLoss()

# =====================================
# TRAINING LOOP
# =====================================
print("\nTraining started...")

start_time = time.time()

best_auc = 0

for epoch in range(1,101):

    loss = train(
        model,
        train_data,
        optimizer,
        criterion,
        device
    )

    val_auc = test(
        model,
        val_data,
        device
    )

    print(
        f"Epoch {epoch:03d} | "
        f"Loss {loss:.4f} | "
        f"Val AUC {val_auc:.4f}"
    )

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(
            model.state_dict(),
            "ddi_model.pth"
        )

end_time = time.time()

print("\nTraining Time (seconds):", end_time - start_time)


# =====================================
# FINAL TEST
# =====================================
print("\nEvaluating on Test Set...")

test_auc = test(
    model,
    test_data,
    device
)

print("Final Test AUC:", test_auc)
print("\n✅ Model Saved as ddi_model.pth")

total_params = sum(p.numel() for p in model.parameters())
print("Total Trainable Parameters:", total_params)