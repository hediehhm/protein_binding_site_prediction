# train.py

import os
import csv
import yaml
import random
import logging
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from sklearn.metrics import matthews_corrcoef


# ============================================================
# DATASETS
# ============================================================

from dataloader.attention_dataset import (
    ProteinLigandDataset,
    collate_protein_ligand
)


# ============================================================
# MODELS
# ============================================================

from models.projected_cross_attention_transformer import (
    ProjectedCrossAttentionTransformer
)


# ============================================================
# CONFIG
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to config file"
)

args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# ============================================================
# REPRODUCIBILITY
# ============================================================

seed = config["seed"]

random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# EXPERIMENT SETUP
# ============================================================

exp_name = config["experiment_name"]

results_dir = os.path.join(
    "results",
    exp_name
)

os.makedirs(results_dir, exist_ok=True)

model_out = os.path.join(
    results_dir,
    "best_model.pt"
)

metrics_file = os.path.join(
    results_dir,
    "metrics.csv"
)

log_file = os.path.join(
    results_dir,
    "train.log"
)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filemode="w"
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

console.setFormatter(formatter)

logging.getLogger().addHandler(console)

logging.info("=" * 80)
logging.info(f"Starting experiment: {exp_name}")
logging.info("=" * 80)

# ============================================================
# DATASET TYPE
# ============================================================

dataset_type = config["dataset"]["type"]

# ============================================================
# LOAD DATASETS
# ============================================================

logging.info("Loading datasets...")

if dataset_type == "attention":

    train_dataset = ProteinLigandDataset(
        csv_file=config["data"]["train_csv"],
        esm_dir=config["data"]["train_esm_dir"],
        lig_dict_file=config["data"]["train_ligand_dict"],
        labels_dict=torch.load(config["data"]["train_labels"])
    )

    val_dataset = ProteinLigandDataset(
        csv_file=config["data"]["val_csv"],
        esm_dir=config["data"]["val_esm_dir"],
        lig_dict_file=config["data"]["val_ligand_dict"],
        labels_dict=torch.load(config["data"]["val_labels"])
    )

    collate_fn = collate_protein_ligand

else:
    raise ValueError(f"Unknown dataset type: {dataset_type}")

# ============================================================
# DATALOADERS
# ============================================================

batch_size = config["training"]["batch_size"]

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)

logging.info(f"Train samples: {len(train_dataset)}")
logging.info(f"Validation samples: {len(val_dataset)}")

# ============================================================
# MODEL SELECTION
# ============================================================

model_name = config["model"]["name"]

# ------------------------------------------------------------
# PROJECTED CROSS ATTENTION TRANSFORMER
# ------------------------------------------------------------

if model_name == "ProjectedCrossAttentionTransformer":

    sample_esm, sample_lig, _ = train_dataset[0]

    model = ProjectedCrossAttentionTransformer(

        esm_dim=sample_esm.shape[1],
        lig_dim=sample_lig.shape[1],

        proj_dim=config["model"]["proj_dim"],

        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],

        ff_dim=config["model"]["ff_dim"],

        dropout=config["model"]["dropout"]
    )

else:
    raise ValueError(f"Unknown model: {model_name}")

model = model.to(device)

# ============================================================
# MODEL INFO
# ============================================================

num_params = sum(
    p.numel()
    for p in model.parameters()
)

logging.info(f"Model: {model_name}")
logging.info(f"Parameters: {num_params:,}")

# ============================================================
# LOSS
# ============================================================

all_labels = torch.cat([
    labels
    for *_, labels in train_dataset
])

num_pos = all_labels.sum().item()
num_neg = all_labels.numel() - num_pos

pos_weight = torch.tensor(
    num_neg / (num_pos + 1e-8),
    device=device
)

bce_criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

bce_weight = config["loss"]["bce_weight"]
tversky_weight = config["loss"]["tversky_weight"]

def tversky_loss(
    preds,
    labels,
    alpha=0.2,
    beta=0.8,
    eps=1e-8
):

    preds = torch.sigmoid(preds)

    TP = (preds * labels).sum()
    FP = ((1 - labels) * preds).sum()
    FN = (labels * (1 - preds)).sum()

    return 1 - (
        (TP + eps)
        /
        (TP + alpha * FP + beta * FN + eps)
    )

# ============================================================
# OPTIMIZER
# ============================================================

optimizer_name = config["training"]["optimizer"]

lr = config["training"]["lr"]
weight_decay = config["training"]["weight_decay"]

if optimizer_name == "AdamW":

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

elif optimizer_name == "Adam":

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

else:
    raise ValueError(f"Unknown optimizer: {optimizer_name}")

# ============================================================
# METRICS CSV
# ============================================================

with open(metrics_file, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "epoch",
        "train_loss",
        "train_mcc",
        "val_loss",
        "val_mcc" 

    ])

# ============================================================
# TRAIN / EVAL FUNCTION
# ============================================================

def run_epoch(model, loader, train=True):

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    for batch in loader:

        # ====================================================
        # TRANSFORMER PIPELINE
        # ====================================================

        if dataset_type == "attention":

            (
                protein_emb,
                ligand_emb,
                labels,
                protein_mask,
                ligand_mask
            ) = batch

            protein_emb = protein_emb.to(device)
            ligand_emb = ligand_emb.to(device)

            labels = labels.to(device)

            protein_mask = protein_mask.to(device)
            ligand_mask = ligand_mask.to(device)

            valid_mask = ~protein_mask

            with torch.set_grad_enabled(train):

                logits = model(
                    protein_emb,
                    ligand_emb,
                    protein_pad_mask=protein_mask,
                    ligand_pad_mask=ligand_mask
                )

                logits_valid = logits[valid_mask]
                labels_valid = labels[valid_mask]

                bce = bce_criterion(
                    logits_valid,
                    labels_valid
                )

                tversky = tversky_loss(
                    logits_valid,
                    labels_valid
                )

                loss = (
                    bce_weight * bce +
                    tversky_weight * tversky
                )

                if train:

                    optimizer.zero_grad()

                    loss.backward()

                    optimizer.step()

        # ====================================================
        # CONCAT PIPELINE
        # ====================================================

        elif dataset_type == "concat":

            (
                embeddings,
                labels,
                pad_mask
            ) = batch

            embeddings = embeddings.to(device)

            labels = labels.to(device)

            pad_mask = pad_mask.to(device)

            valid_mask = ~pad_mask

            with torch.set_grad_enabled(train):

                logits = model(embeddings)

                logits_valid = logits[valid_mask]
                labels_valid = labels[valid_mask]

                bce = bce_criterion(
                    logits_valid,
                    labels_valid
                )

                tversky = tversky_loss(
                    logits_valid,
                    labels_valid
                )

                loss = (
                    bce_weight * bce +
                    tversky_weight * tversky
                )

                if train:

                    optimizer.zero_grad()

                    loss.backward()

                    optimizer.step()

        # ====================================================
        # METRICS
        # ====================================================

        total_loss += (
            loss.item() * labels_valid.size(0)
        )

        total_samples += labels_valid.size(0)

        all_preds.append(
            torch.sigmoid(
                logits_valid
            ).detach().cpu()
        )

        all_labels.append(
            labels_valid.detach().cpu()
        )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    preds_binary = (
        all_preds >= 0.5
    ).astype(int)

    mcc = matthews_corrcoef(
        all_labels,
        preds_binary
    )

    avg_loss = total_loss / total_samples

    return avg_loss, mcc

# ============================================================
# TRAINING LOOP
# ============================================================

num_epochs = config["training"]["num_epochs"]

best_mcc = -1e9

logging.info("Starting training...")

for epoch in range(num_epochs):

    train_loss, train_mcc = run_epoch(
        model,
        train_loader,
        train=True
    )

    val_loss, val_mcc = run_epoch(
        model,
        val_loader,
        train=False
    )


    # --------------------------------------------------------
    # LOGGING
    # --------------------------------------------------------

    logging.info(
        f"Epoch {epoch+1}/{num_epochs}, "
        f"Train Loss: {train_loss:.4f}, "
        f"Train MCC: {train_mcc:.4f}, "
        f"Val Loss: {val_loss:.4f}, "
        f"Val MCC: {val_mcc:.4f} "
        )

    # --------------------------------------------------------
    # CSV METRICS
    # --------------------------------------------------------

    with open(metrics_file, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            epoch + 1,
            train_loss,
            train_mcc,
            val_loss,
            val_mcc
        ])

    # --------------------------------------------------------
    # CHECKPOINTING
    # --------------------------------------------------------

    if val_mcc > best_mcc:

        best_mcc = val_mcc

        torch.save(
            model.state_dict(),
            model_out
        )

        logging.info(
            f"âœ” New best model saved "
            f"(Val MCC = {best_mcc:.4f})"
        )

logging.info("=" * 80)
logging.info("Training complete")
logging.info(f"Best validation MCC: {best_mcc:.4f}")
logging.info("=" * 80)
