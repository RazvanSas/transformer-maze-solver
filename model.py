
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model=64,
        num_heads=4,
        d_ff=256,
        dropout=0.1
    ):
        super().__init__()

        # Normalize before each sublayer.
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask):
        # x: (batch_size, sequence_length, d_model)

        normalized = self.ln1(x)

        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False
        )

        # First residual connection.
        x = x + self.dropout(attention_output)

        normalized = self.ln2(x)
        ff_output = self.feed_forward(normalized)

        # Second residual connection.
        x = x + self.dropout(ff_output)

        return x


class MazeTransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        action_token_ids,
        max_seq_len=66,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        dropout=0.1
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.position_embedding = nn.Embedding(
            max_seq_len,
            d_model
        )

        self.input_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.final_ln = nn.LayerNorm(d_model)

        # Initially predict scores for every vocabulary token.
        self.lm_head = nn.Linear(d_model, vocab_size)

        # Store the four vocabulary IDs in a tensor.
        self.register_buffer(
            "action_token_ids",
            torch.tensor(action_token_ids, dtype=torch.long)
        )

        # True means that attention to this position is blocked.
        causal_mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool),
            diagonal=1
        )

        self.register_buffer("causal_mask", causal_mask)

    def forward(self, input_ids):
        # input_ids: (B, 66)
        batch_size, seq_len = input_ids.shape

        positions = torch.arange(
            seq_len,
            device=input_ids.device
        )

        token_vectors = self.token_embedding(input_ids)
        position_vectors = self.position_embedding(positions)

        x = token_vectors + position_vectors
        x = self.input_dropout(x)

        mask = self.causal_mask[:seq_len, :seq_len]

        for block in self.blocks:
            x = block(x, mask)

        x = self.final_ln(x)

        # Only the final sequence position is used for classification.
        final_state = x[:, -1, :]           # (B, 64)

        all_logits = self.lm_head(final_state)  # (B, 74)

        # Keep only UP, DOWN, LEFT, RIGHT, in that order.
        action_logits = all_logits.index_select(
            dim=1,
            index=self.action_token_ids
        )                                   # (B, 4)

        return action_logits
