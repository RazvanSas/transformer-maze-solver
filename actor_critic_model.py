"""
Actor-critic version of the 8x8 MazeTransformer.

The Transformer and actor head use the same module names/shapes as MazeTransformer,
so pretrained supervised weights can be loaded. The newly initialized value_head
predicts V(s), a scalar state value.
"""
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, num_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask):
        normalized = self.ln1(x)
        attention_output, _ = self.attention(
            normalized, normalized, normalized,
            attn_mask=causal_mask, need_weights=False,
        )
        x = x + self.dropout(attention_output)
        normalized = self.ln2(x)
        ff_output = self.feed_forward(normalized)
        x = x + self.dropout(ff_output)
        return x


class ActorCriticMazeTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        action_token_ids,
        max_seq_len=66,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        dropout=0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

        # Same actor output layer as the supervised MazeTransformer.
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.register_buffer(
            "action_token_ids",
            torch.tensor(action_token_ids, dtype=torch.long),
        )

        causal_mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask)

        # New critic head: predicts one scalar V(s) per state.
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, input_ids):
        # input_ids: (B, sequence_length)
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)

        token_vectors = self.token_embedding(input_ids)
        position_vectors = self.position_embedding(positions)
        x = self.input_dropout(token_vectors + position_vectors)

        mask = self.causal_mask[:seq_len, :seq_len]
        for block in self.blocks:
            x = block(x, mask)

        x = self.final_ln(x)
        final_state = x[:, -1, :]  # (B, d_model)

        all_logits = self.lm_head(final_state)
        action_logits = all_logits.index_select(
            dim=1, index=self.action_token_ids
        )  # (B, 4)

        values = self.value_head(final_state).squeeze(-1)  # (B,)
        return action_logits, values
