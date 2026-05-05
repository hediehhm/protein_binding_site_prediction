# protein_binding_site_prediction

This repository contains my master thesis work on predicting protein binding sites using sequence-only information of proteins and ligand information.

## Overview

This work combines protein and ligand representations in a shared learning framework for binding site prediction.

## Data Source

The LP-PDBBind dataset is used in this work (see Citation section).

Original data can be obtained from:
[https://github.com/THGLab/LP-PDBBind/]

In this work, the provided CSV file was used as the starting point for preprocessing and model development.

## Data Processing and Label Construction

Starting from the LP-PDBBind dataset, the provided CSV file was used together with corresponding PDB structures to construct a sequence-level binding site prediction dataset.

The processing pipeline is as follows:

### 1. Ligand Identification

* Ligand SMILES and PDB structures were used to extract ligand 3-letter codes.
* Entries corresponding to peptide ligands (i.e., not small molecules) were excluded due to the focus on small-molecule binding.

### 2. Binding Residue Extraction

* Binding residues were defined using structural proximity.
* An amino acid residue was labeled as binding if any of its heavy atoms was within 5 Å of any ligand heavy atom.
* This produced residue-level binding annotations derived directly from 3D structure.

### 3. Sequence Label Construction

* Protein sequences were converted into a binary representation:

  * `1` → binding residue
  * `0` → non-binding residue
* The generated labels were aligned with the original protein sequences to ensure positional consistency.

### 4. Chain Filtering and Standardization

* Multi-chain complexes were processed as follows:

  * Heterogeneous chains were removed, as the model targets single-chain prediction.
  * For homo-multimeric proteins with identical binding residues, a single representative chain was retained.

### 5. Final Dataset

The resulting dataset consists of:

* Single-chain protein sequences
* Associated small-molecule ligands
* Binary residue-level binding annotations

This dataset serves as the input for sequence-based binding site prediction models.

## Feature Extraction

To transform raw biological entities into machine-learning representations, pretrained models were used for both proteins and ligands.

### Protein Representation (ESM2)

Protein sequences were encoded using the ESM2 protein language model, a transformer-based model trained on large-scale protein sequence data.

Each amino acid sequence is passed through the model to obtain contextualized residue-level embeddings. These embeddings capture biochemical and evolutionary information learned from large protein corpora.
GitHub: https://github.com/facebookresearch/esm

The resulting representation is used as input features for downstream binding site prediction.

### Ligand Representation (FARM)

Ligands, represented as SMILES strings, were encoded using the FARM model.
GitHub: https://github.com/thaonguyen217/farm_molecular_representation

FARM generates atom-level embeddings by learning contextual representations of atoms within molecular graphs derived from SMILES structures.

This results in a structured representation of each ligand, where each atom is associated with a learned feature vector.

### Final Representation

The model operates on paired inputs:

* Protein residue embeddings (ESM2)
* Ligand atom embeddings (FARM)

## References

Li, J., Guan, X., Zhang, O., Sun, K., Wang, Y., Bagni, D., & Head-Gordon, T. (2024). Leak Proof PDBBind: A Reorganized Dataset of Protein-Ligand Complexes for More Generalizable Binding Affinity Prediction. ArXiv, arXiv:2308.09639v2.

Wang, Y., et al. (2024). FARM: Functional Group-Aware Representations for Small Molecules.

Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., ... & Rives, A. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. Science, 379(6637), 1123-1130. doi.org

