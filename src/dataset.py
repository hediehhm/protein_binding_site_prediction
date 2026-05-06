import os
import torch
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset


class ProteinLigandDataset(Dataset):
    """
    Dataset for protein-ligand residue-level binding prediction.

    Each sample returns:
        - protein embedding (L x D)
        - ligand embedding (N x F)
        - binary residue labels (L)

    Expected inputs:
        csv_file: CSV with columns ['pdbid', 'codes']
        esm_dir: directory containing {pdbid}.npz files
        lig_dict_file: pickle file with ligand embeddings
        labels_dict: dict mapping pdbid -> label array
    """

    def __init__(self, csv_file, esm_dir, lig_dict_file, labels_dict):
        self.meta = pd.read_csv(csv_file)
        self.esm_dir = esm_dir
        self.labels_dict = labels_dict

        # Load ligand embeddings
        with open(lig_dict_file, "rb") as f:
            lig_raw = pickle.load(f)

        self.lig_dict = {
            entry["code"]: entry["embedding"] for entry in lig_raw
        }

        # Keep only valid ligand entries
        self.meta = self.meta[
            self.meta["codes"].isin(self.lig_dict.keys())
        ].reset_index(drop=True)

        if len(self.meta) == 0:
            raise ValueError("No valid ligand entries found.")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]

        pdbid = row["pdbid"]
        code = row["codes"]

        # --- Protein embedding ---
        esm_path = os.path.join(self.esm_dir, f"{pdbid}.npz")
        if not os.path.exists(esm_path):
            raise FileNotFoundError(f"Missing ESM file for {pdbid}")

        esm_data = np.load(esm_path)
        esm_emb = torch.tensor(
            esm_data[esm_data.files[0]], dtype=torch.float32
        )

        # --- Ligand embedding ---
        if code not in self.lig_dict:
            raise KeyError(f"Missing ligand embedding for {code}")

        lig_emb = torch.tensor(
            self.lig_dict[code], dtype=torch.float32
        )

        # --- Labels ---
        if pdbid not in self.labels_dict:
            raise KeyError(f"Missing labels for {pdbid}")

        labels = torch.tensor(
            self.labels_dict[pdbid], dtype=torch.float32
        )

        return esm_emb, lig_emb, labels
