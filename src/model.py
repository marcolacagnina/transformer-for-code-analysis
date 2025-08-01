import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Injects some information about the relative or absolute position of the tokens in the sequence."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return x


class ComplexityClassifier(nn.Module):
    """
    A Transformer-based model to classify code complexity.
    It uses an embedding layer, positional encoding, a Transformer encoder, and a final linear layer.
    """

    def __init__(self, vocab_size: int, num_labels: int, d_model: int, nhead: int, num_layers: int, max_len: int,
                 dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Create a boolean padding mask from the attention mask
        # True values in src_key_padding_mask are positions that will be ignored.
        padding_mask = (attention_mask == 0)

        # Forward pass
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # Use the embedding of the [CLS] token (at position 0) for classification
        cls_embedding = x[:, 0, :]
        cls_embedding = self.layer_norm(cls_embedding)
        cls_embedding = self.dropout(cls_embedding)

        logits = self.classifier(cls_embedding)
        return logits