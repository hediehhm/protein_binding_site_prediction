# protein_binding_site_prediction
Sequence-based protein binding site prediction (Master thesis work in progress)

This repository contains my master thesis work on predicting protein binding sites using sequence-only information of proteins and ligand information.

## Goal
Develop and evaluate models that predict binding residues from amino acid sequences without using structural data.

## Planned Work
- generate protein embeddings using esm2 protein language model.
- generate ligand embedddings using FARM method.
- find an efficient architecture to train a predictor.
- Evaluation

## Data Source

This project uses the LP-PDBBind dataset derived from the PDBbind database.

Original data can be obtained from:
[https://github.com/THGLab/LP-PDBBind/]

In this work, the provided CSV file was used as the starting point for preprocessing and model development.
