import torch
from sklearn.metrics import roc_auc_score


# -------------------------
# TRAIN FUNCTION
# -------------------------
def train(model, train_data, optimizer, criterion, device):

    model.train()
    optimizer.zero_grad()

    z = model.encode(
     train_data.x.to(device),
     train_data.edge_index.to(device),      # this is edge_index_I
     train_data.edge_index_D.to(device)     # ADD THIS
    )

    out = model.decode(
        z,
        train_data.edge_label_index.to(device)
    )

    loss = criterion(
        out,
        train_data.edge_label.float().to(device)
    )

    loss.backward()
    optimizer.step()

    return loss.item()


# -------------------------
# TEST FUNCTION
# -------------------------
@torch.no_grad()
def test(model, data, device):

    model.eval()

    z = model.encode(
     data.x.to(device),
     data.edge_index.to(device),
     data.edge_index_D.to(device)
    )
    out = model.decode(
        z,
        data.edge_label_index.to(device)
    ).sigmoid()

    auc = roc_auc_score(
        data.edge_label.cpu(),
        out.cpu()
    )

    return auc