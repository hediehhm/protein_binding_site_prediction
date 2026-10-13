# dataloader/attention_dataset.py

import os
import pickle

import numpy as np
import pandas as pd

import torch

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# ============================================================
# ATTENTION DATASET
# ============================================================

class ProteinLigandDataset(Dataset):

    """
    Dataset for transformer-based protein-ligand
    residue-level binding site prediction.

    Returns:
        esm_emb      : [L_protein, esm_dim]
        ligand_emb   : [L_ligand, lig_dim]
        labels       : [L_protein]
    """

    def __init__(
        self,
        csv_file,
        esm_dir,
        lig_dict_file,
        labels_dict,
    ):


        # ----------------------------------------------------
        # LOAD METADATA
        # ----------------------------------------------------

        self.meta = pd.read_csv(csv_file, dtype=str)

        self.esm_dir = esm_dir

        self.labels_dict = labels_dict

        # ----------------------------------------------------
        # LOAD LIGAND EMBEDDINGS
        # ----------------------------------------------------

        with open(lig_dict_file, "rb") as f:

            lig_raw = pickle.load(f)

        self.lig_dict = {
            entry["code"]: entry["embedding"]
            for entry in lig_raw
        }

        # ----------------------------------------------------
        # KEEP ONLY VALID ENTRIES
        # ----------------------------------------------------

        self.meta = self.meta[
            self.meta["codes"].isin(
                self.lig_dict.keys()
            )
        ].reset_index(drop=True)

        if len(self.meta) == 0:

            raise ValueError(
                "No valid ligand entries found."
            )

    def __len__(self):

        return len(self.meta)

    def __getitem__(self, idx):

        row = self.meta.iloc[idx]

        pdbid = row["pdbid"]

        ligand_code = row["codes"]

        # ----------------------------------------------------
        # LOAD ESM EMBEDDINGS
        # ----------------------------------------------------

        esm_path = os.path.join(
            self.esm_dir,
            f"{pdbid}.npz"
        )

        esm_data = np.load(esm_path)

        esm_emb = torch.tensor(
            esm_data[esm_data.files[0]],
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # LOAD LIGAND EMBEDDINGS
        # ----------------------------------------------------

        ligand_emb = torch.tensor(
            self.lig_dict[ligand_code],
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # LOAD LABELS
        # ----------------------------------------------------

        labels = torch.tensor(
            self.labels_dict[pdbid],
            dtype=torch.float32
        )


        return (
            esm_emb,
            ligand_emb,
            labels
        )


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_protein_ligand(batch):

    """
    Pads:
        proteins
        ligands
        labels

    Produces:
        protein_pad_mask
        ligand_pad_mask
    """

    proteins, ligands, labels = zip(*batch)

    # --------------------------------------------------------
    # SEQUENCE LENGTHS
    # --------------------------------------------------------

    protein_lengths = torch.tensor([
        p.size(0)
        for p in proteins
    ])

    ligand_lengths = torch.tensor([
        l.size(0)
        for l in ligands
    ])

    # --------------------------------------------------------
    # PAD SEQUENCES
    # --------------------------------------------------------

    proteins_pad = pad_sequence(
        proteins,
        batch_first=True
    )

    ligands_pad = pad_sequence(
        ligands,
        batch_first=True
    )

    labels_pad = pad_sequence(
        labels,
        batch_first=True
    )

    # --------------------------------------------------------
    # BUILD PADDING MASKS
    # --------------------------------------------------------

    protein_pad_mask = (
        torch.arange(
            proteins_pad.size(1)
        )[None, :]
        >=
        protein_lengths[:, None]
    )

    ligand_pad_mask = (
        torch.arange(
            ligands_pad.size(1)
        )[None, :]
        >=
        ligand_lengths[:, None]
    )

    return (
        proteins_pad,
        ligands_pad,
        labels_pad,
        protein_pad_mask,
        ligand_pad_mask,
    )
