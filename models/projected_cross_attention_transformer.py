import torch
import torch.nn as nn

"""
Projected Cross-Attention Transformer for ligand-specific
protein binding residue prediction.

The model integrates:
- Protein residue embeddings from ESM-2
- Ligand atom embeddings from FARM

Protein and ligand representations are projected into
a shared latent space, independently contextualized using
Transformer encoders, and integrated through protein-query /
ligand-key cross-attention.

Output:
Residue-level binding logits for each protein position.
"""


# ============================================================
# MODEL
# ============================================================

class ProjectedCrossAttentionTransformer(nn.Module):

    def __init__(
        self,
        esm_dim=1280,
        lig_dim=768,

        proj_dim=256,

        num_layers=2,
        num_heads=4,
        ff_dim=512,

        dropout=0.1
    ):

        super().__init__()

        # ====================================================
        # PROJECTION MLPs
        # ====================================================

        self.protein_proj = nn.Sequential(

            nn.Linear(esm_dim, 512),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(512, proj_dim)
        )

        self.ligand_proj = nn.Sequential(

            nn.Linear(lig_dim, 512),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(512, proj_dim)
        )

        # ====================================================
        # PROTEIN TRANSFORMER
        # ====================================================

        protein_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )

        self.protein_transformer = nn.TransformerEncoder(
            protein_layer,
            num_layers=num_layers
        )

        # ====================================================
        # LIGAND TRANSFORMER
        # ====================================================

        ligand_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )

        self.ligand_transformer = nn.TransformerEncoder(
            ligand_layer,
            num_layers=num_layers
        )

        # ====================================================
        # CROSS ATTENTION
        # ====================================================

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.cross_norm = nn.LayerNorm(
            proj_dim
        )

        # ====================================================
        # CLASSIFIER
        # ====================================================

        self.classifier = nn.Sequential(

            nn.Linear(proj_dim, 128),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(64, 1)
        )

    def forward(
        self,
        protein_emb,
        ligand_emb,
        protein_pad_mask=None,
        ligand_pad_mask=None
    ):

        # ====================================================
        # PROJECT TO SHARED SPACE
        # ====================================================

        protein_proj = self.protein_proj(
            protein_emb
        )

        ligand_proj = self.ligand_proj(
            ligand_emb
        )

        # ====================================================
        # PROTEIN TRANSFORMER
        # ====================================================

        protein_enc = self.protein_transformer(
            protein_proj,
            src_key_padding_mask=protein_pad_mask
        )

        # ====================================================
        # LIGAND TRANSFORMER
        # ====================================================

        ligand_enc = self.ligand_transformer(
            ligand_proj,
            src_key_padding_mask=ligand_pad_mask
        )

        # ====================================================
        # CROSS ATTENTION
        # ====================================================

        context, _ = self.cross_attn(
            query=protein_enc,
            key=ligand_enc,
            value=ligand_enc,
            key_padding_mask=ligand_pad_mask
        )

        # ====================================================
        # RESIDUAL FUSION
        # ====================================================

        enriched = protein_enc + self.cross_norm(
            context
        )

        # ====================================================
        # CLASSIFICATION
        # ====================================================

        out = self.classifier(
            enriched
        ).squeeze(-1)

        return out
