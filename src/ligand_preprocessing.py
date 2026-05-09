# ligand_preprocessing.py

import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolStandardize


# -----------------------------
# SMILES CLEANING FUNCTIONS
# -----------------------------

def remove_empty_smiles(df, smiles_col="SMILES"):
    """
    Remove rows with NaN or empty SMILES strings.
    """
    mask = df[smiles_col].notna() & (df[smiles_col].astype(str).str.strip() != "")
    return df[mask].reset_index(drop=True)


def canonicalize_smiles(smiles):
    """
    Convert SMILES into a canonical RDKit representation.
    Also applies basic RDKit cleanup (salt stripping, normalization).
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        clean_mol = MolStandardize.rdMolStandardize.Cleanup(mol)
        return Chem.MolToSmiles(clean_mol, canonical=True)

    except Exception:
        return None


def canonicalize_dataframe(df, smiles_col="SMILES", id_col="codes"):
    """
    Convert a dataframe with SMILES into canonical SMILES format.
    Keeps only valid molecules.
    
    Returns:
        cleaned_df with columns:
        - id_col
        - canonical_smiles
    """
    cleaned_rows = []

    for _, row in df.iterrows():
        smiles = row[smiles_col]
        ligand_id = row[id_col]

        canonical_smiles = canonicalize_smiles(smiles)

        if canonical_smiles is not None:
            cleaned_rows.append({
                id_col: ligand_id,
                "canonical_smiles": canonical_smiles
            })

    return pd.DataFrame(cleaned_rows)


# -----------------------------
# HIGH-LEVEL PIPELINE FUNCTION
# -----------------------------

def preprocess_ligand_csv(
    input_csv,
    output_csv=None,
    smiles_col="SMILES",
    id_col="codes"
):
    """
    Full preprocessing pipeline:
    1. Load CSV
    2. Remove empty SMILES
    3. Canonicalize SMILES
    4. Save cleaned CSV (optional)

    Returns:
        cleaned pandas DataFrame
    """

    df = pd.read_csv(input_csv)

    # Step 1: remove empty entries
    df = remove_empty_smiles(df, smiles_col=smiles_col)

    # Step 2: canonicalization
    cleaned_df = canonicalize_dataframe(
        df,
        smiles_col=smiles_col,
        id_col=id_col
    )

    # Step 3: save if needed
    if output_csv is not None:
        cleaned_df.to_csv(output_csv, index=False)

    return cleaned_df
