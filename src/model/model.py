import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        self.pe = pe.unsqueeze(0)  # Shape: (1, max_len, model_dim)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class TrajTransformer(nn.Module):
    def __init__(self, input_dim=5, model_dim=128, num_heads=4, num_layers=4, dropout=0.1, max_len=1000):
        super().__init__()

        self.model_dim = model_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, model_dim)

        # Learnable CLS token embedding (shape: [1, model_dim])
        self.cls_token = nn.Parameter(torch.randn(1, model_dim))

        # Positional encoding
        self.pos_encoder = PositionalEncoding(model_dim, max_len=max_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MAE Reconstruction Head
        self.reconstruct_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, input_dim)
        )

        # Next State Prediction (NSP) Head
        self.nsp_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 4)  # Δlat, Δlng, sin(bearing), cos(bearing), bearing_change, curvature
        )

    def forward(self, x, attn_mask=None, task='mae', mask_pos=None, key_padding_mask=None):
        """
        Args:
            x: Tensor of shape (B, T, input_dim)
            attn_mask: Optional attention mask (T, T) or (B, T, T)
            task: 'mae' or 'nsp'
            mask_pos: Boolean mask of shape (B, T) for MAE
        """
        B, T, _ = x.size()

        x = self.input_proj(x)  # (B, T, model_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) 

        if task == 'mae' and mask_pos is not None:
            mask_token = torch.zeros_like(x[:, 1:])
            mask_token = mask_token.expand(B, T, -1)
            x[:, 1:] = torch.where(mask_pos.unsqueeze(-1), mask_token, x[:, 1:])

        # Add positional encoding
        x = self.pos_encoder(x)

        if key_padding_mask is not None:
            cls_mask_bit = torch.zeros((B, 1), dtype=torch.bool, device=x.device)
            extend_mask = torch.cat([cls_mask_bit, key_padding_mask], dim=1)

            if extend_mask.all(dim=1).any():
                print("❌")
                extend_mask[:, 0] = False
        else:
            extend_mask = None

        # Transformer encoding
        x = self.transformer_encoder(x, mask=attn_mask, src_key_padding_mask=extend_mask)

        if task == 'mae':
            out = self.reconstruct_head(x[:, 1:])
            return out, mask_pos

        elif task == 'nsp':
            out = self.nsp_head(x[:, :-1])  # Predict next state for all except last point
            return out


        else:
            raise ValueError("Unknown task type")


    def _encode(self, x, mask=None):
        """
        x: (B, T, input_dim)
        mask: (B, T) bool, True indicates padding positions.
        Returns:
            hidden: (B, T+1, model_dim), including [CLS] at index 0.
        """
        B, _, _ = x.size()
        x = self.input_proj(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoder(x)

        if mask is not None:
            cls_mask_bit = torch.zeros((B, 1), dtype=torch.bool, device=mask.device)
            extend_mask = torch.cat([cls_mask_bit, mask], dim=1)
            if extend_mask.all(dim=1).any():
                extend_mask[:, 0] = False
            x = self.transformer_encoder(x, src_key_padding_mask=extend_mask)
        else:
            x = self.transformer_encoder(x)
        return x

    def get_encoder_outputs(self, x, mask=None):
        """
        Extract global and token-level route representations.
        Returns:
            cls_emb: (B, model_dim)
            seq_emb: (B, T, model_dim), all non-[CLS] tokens (S, r_i, E, D, ...)
        """
        hidden = self._encode(x, mask=mask)
        return hidden[:, 0], hidden[:, 1:]

    def get_embedding(self, x, mask=None):
        """
        Extract [CLS] embedding as trajectory representation.
        x: (B, T, D)
        mask: (B, T) bool, True indicates padding positions.
        Returns:
            cls_emb: (B, model_dim)
        """
        cls_emb, _ = self.get_encoder_outputs(x, mask=mask)
        return cls_emb

    def generate_span_mask(self, batch_size, seq_len, mask_ratio=0.3, span_len=3, device=None):
        num_to_mask = int((seq_len - 1) * mask_ratio)  # exclude CLS
        mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        for b in range(batch_size):
            masked = 0
            while masked < num_to_mask:
                start = torch.randint(1, seq_len - span_len + 1, (1,)).item()
                end = min(start + span_len, seq_len)
                mask[b, start:end] = True
                masked += end - start
        return mask.to(device) if device else mask
        
    def generate_mae_mask(self, batch_size, seq_len, mask_ratio=0.3, device=None):
        """
        Returns:
            mask_pos: (B, T) bool Tensor. True indicates the position is masked (needs reconstruction).
        """
        num_mask = int((seq_len - 1) * mask_ratio) # exclude CLS from masking
        masks = []
        for _ in range(batch_size):
            perm = torch.randperm(seq_len - 1) + 1 # Only mask from position 1 onwards
            mask = torch.zeros(seq_len, dtype=torch.bool)
            mask[perm[:num_mask]] = True
            masks.append(mask)
        mask_tensor = torch.stack(masks, dim=0)
        if device:
            mask_tensor = mask_tensor.to(device)
        return mask_tensor  # Shape: (B, T)

    def generate_causal_mask(self, seq_len, device=None):
        """
        Returns:
            causal_mask: (T, T) float Tensor. Masked positions are -inf.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        mask = mask.to(device) if device else mask
        return mask.masked_fill(mask, float('-inf'))

class GatedCrossAttentionFusion(nn.Module):
    def __init__(self, model_dim, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.gate_fc = nn.Linear(model_dim * 2, model_dim)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, context_emb, seq_emb, cls_emb, key_padding_mask=None):
        """
        context_emb: (B, D) query from dynamic context C
        seq_emb: (B, L, D) token-level route representations Z_seq
        cls_emb: (B, D) global route representation z_CLS
        key_padding_mask: (B, L) bool, True indicates padding positions to ignore
        """
        query = context_emb.unsqueeze(1)
        z_attn, _ = self.cross_attn(
            query=query,
            key=seq_emb,
            value=seq_emb,
            key_padding_mask=key_padding_mask
        )
        z_attn = z_attn.squeeze(1)

        gate_input = torch.cat([z_attn, cls_emb], dim=-1)
        g = torch.sigmoid(self.gate_fc(gate_input))
        h_fused = g * z_attn + (1 - g) * cls_emb
        return self.layer_norm(h_fused)

class ETARegressor(nn.Module):
    def __init__(
        self,
        encoder,
        num_drivers=None,
        driver_emb_dim=16,
        time_input_dim=10,
        hidden_dim=128,
        num_attn_heads=4,
        use_driver_emb=True,
    ):
        """
        ETA prediction head used in the fine-tuning stage.

        Args:
            encoder: Pre-trained TrajTransformer route encoder.
            num_drivers: Number of driver IDs for the optional embedding table.
            driver_emb_dim: Dimension of the driver ID embedding.
            time_input_dim: Dimension of cyclical temporal context features.
            hidden_dim: Hidden dimension of the regression MLP.
            num_attn_heads: Number of heads in gated cross-attention fusion.
            use_driver_emb: If True, x_context is expected to contain driver ID
                as its last column; otherwise, x_context only contains time features.
        """
        super().__init__()
        self.encoder = encoder
        self.use_driver_emb = use_driver_emb

        if use_driver_emb:
            if num_drivers is None:
                raise ValueError("num_drivers is required when use_driver_emb=True")
            self.driver_embedding = nn.Embedding(
                num_embeddings=num_drivers,
                embedding_dim=driver_emb_dim,
            )
            context_input_dim = time_input_dim + driver_emb_dim
        else:
            self.driver_embedding = None
            context_input_dim = time_input_dim

        self.context_fc = nn.Sequential(
            nn.Linear(context_input_dim, encoder.model_dim),
            nn.ReLU(),
            nn.LayerNorm(encoder.model_dim)
        )
        self.fusion_module = GatedCrossAttentionFusion(encoder.model_dim, num_heads=num_attn_heads)

        self.mlp_head = nn.Sequential(
            nn.Linear(encoder.model_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.linear = nn.Linear(encoder.model_dim * 2, 1)

    def forward(self, x_traj, x_context, padding_mask=None):
        """
        x_traj: (B, T, D) trajectory input
        x_context: (B, time_input_dim) or (B, time_input_dim + 1) when use_driver_emb
        padding_mask: (B, T) bool, True indicates padding positions
        return: eta_pred: (B,)
        """
        with torch.no_grad():
            cls_emb, seq_emb = self.encoder.get_encoder_outputs(x_traj, mask=padding_mask)

        if self.use_driver_emb:
            driverid = x_context[:, -1].long()
            time_proj = x_context[:, :-1]
            driver_emb = self.driver_embedding(driverid)
            combined_context = torch.cat([driver_emb, time_proj], dim=1)
        else:
            combined_context = x_context

        context_emb = self.context_fc(combined_context)
        h_final = self.fusion_module(
            context_emb, seq_emb, cls_emb, key_padding_mask=padding_mask
        )
        final_combined_emb = torch.cat([h_final, context_emb], dim=1)
        eta_pred = self.mlp_head(final_combined_emb).squeeze(-1)
        return eta_pred