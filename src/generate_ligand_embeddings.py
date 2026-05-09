# generate_ligand_embeddings.py

```python
import pickle
import argparse

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from transformers import BertForMaskedLM, PreTrainedTokenizerFast

from ligand_preprocessing import preprocess_ligand_csv
from pooling import mean_pool_embedding

# =====================================================
# IMPORTANT:
#
# This script depends on the FARM molecular
# representation repository.
#
# Before running this script:
#
# 1. Clone the FARM repository:
#    git clone https://github.com/thaonguyen217/farm_molecular_representation.git
#
# 2. Add the FARM repository to your PYTHONPATH
#    or place it where Python can access it.
#
# =====================================================

from helpers import get_new_smiles_rep
from rdkit.Chem import MolFromSmiles as smiles_to_mol


# -----------------------------------------------------
# FARM REPRESENTATION
# -----------------------------------------------------

def generate_fg_smiles(smiles):
    """
    Convert canonical SMILES into FARM functional-group representation.
    """

    mol = smiles_to_mol(smiles)

    if mol is None:
        return None

    fg_smiles = get_new_smiles_rep(mol)

    # Truncate to BERT maximum sequence length
    tokens = fg_smiles.split()

    if len(tokens) > 512:
        fg_smiles = " ".join(tokens[:512])

    return fg_smiles.strip()


# -----------------------------------------------------
# TOKEN EMBEDDING EXTRACTION
# -----------------------------------------------------

def extract_token_embeddings(
    fg_smiles,
    tokenizer,
    model,
    device
):
    """
    Generate token-level embeddings using the FARM model.
    """

    encoded = tokenizer(
        fg_smiles,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():

        outputs = model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        last_hidden = outputs.last_hidden_state.squeeze(0)

    # Remove [CLS] and [SEP]
    token_embeddings = last_hidden[1:-1]

    return token_embeddings.cpu().numpy()


# -----------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------

def main(args):

    # ---------------------------------------------
    # Load and preprocess ligand CSV
    # ---------------------------------------------

    cleaned_df = preprocess_ligand_csv(
        input_csv=args.input_csv,
        output_csv=args.cleaned_csv,
        smiles_col=args.smiles_column,
        id_col=args.id_column
    )

    print(f"Loaded and cleaned {len(cleaned_df)} ligands.")


    # ---------------------------------------------
    # Load FARM tokenizer and model
    # ---------------------------------------------

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        "thaonguyen217/farm_molecular_representation"
    )

    model = BertForMaskedLM.from_pretrained(
        "thaonguyen217/farm_molecular_representation"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    print(f"Using device: {device}")


    # ---------------------------------------------
    # Generate embeddings
    # ---------------------------------------------

    token_embedding_results = []
    molecular_embedding_results = []

    for _, row in tqdm(cleaned_df.iterrows(), total=len(cleaned_df)):

        ligand_id = row[args.id_column]
        canonical_smiles = row["canonical_smiles"]


        # -----------------------------------------
        # FARM representation
        # -----------------------------------------

        fg_smiles = generate_fg_smiles(canonical_smiles)

        if fg_smiles is None:
            continue


        # -----------------------------------------
        # Token embeddings
        # -----------------------------------------

        token_embeddings = extract_token_embeddings(
            fg_smiles=fg_smiles,
            tokenizer=tokenizer,
            model=model,
            device=device
        )


        token_embedding_results.append({
            "code": ligand_id,
            "token_embedding": token_embeddings
        })


        # -----------------------------------------
        # Molecular embedding (mean pooling)
        # -----------------------------------------

        molecular_embedding = mean_pool_embedding(
            token_embeddings
        )

        molecular_embedding_results.append({
            "code": ligand_id,
            "molecular_embedding": molecular_embedding
        })


    # ---------------------------------------------
    # Save outputs
    # ---------------------------------------------

    with open(args.token_output, "wb") as f:
        pickle.dump(token_embedding_results, f)

    with open(args.molecular_output, "wb") as f:
        pickle.dump(molecular_embedding_results, f)


    print(f"Saved token embeddings to: {args.token_output}")
    print(f"Saved molecular embeddings to: {args.molecular_output}")


# -----------------------------------------------------
# ARGUMENT PARSER
# -----------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate ligand embeddings using FARM molecular representations."
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to ligand CSV file"
    )

    parser.add_argument(
        "--cleaned_csv",
        type=str,
        default="cleaned_smiles.csv",
        help="Path to save cleaned SMILES CSV"
    )

    parser.add_argument(
        "--token_output",
        type=str,
        default="token_embeddings.pkl",
        help="Output path for token embeddings"
    )

    parser.add_argument(
        "--molecular_output",
        type=str,
        default="molecular_embeddings.pkl",
        help="Output path for molecular embeddings"
    )

    parser.add_argument(
        "--smiles_column",
        type=str,
        default="SMILES",
        help="Name of SMILES column"
    )

    parser.add_argument(
        "--id_column",
        type=str,
        default="codes",
        help="Name of ligand ID column"
    )

    args = parser.parse_args()

    main(args)
```

---

# Why This Works Without `farm_utils.py`

The FARM-related code you actually need is relatively small:

* `get_new_smiles_rep()`
* tokenizer loading
* model loading
* embedding extraction

You are not implementing FARM internals.
You are using FARM inside your pipeline.

So instead of building a large wrapper module around FARM, the cleaner approach is:

* keep your own reusable preprocessing in `ligand_preprocessing.py`
* keep pooling logic in `pooling.py`
* directly use FARM components inside the main pipeline script

This keeps ownership and dependency boundaries very clear.

---

# Example Usage

```bash
python src/generate_ligand_embeddings.py \
    --input_csv data/val_smiles_codes.csv \
    --token_output outputs/token_embeddings.pkl \
    --molecular_output outputs/molecular_embeddings.pkl
```

---

# What `argparse` Is Doing Here

Instead of hardcoding:

```python
input_csv = "val_smiles_codes.csv"
```

You now pass parameters from the command line.

This allows the same script to work for:

* train set
* validation set
* test set
* external datasets

without modifying the code itself.
