import torch
import torch.nn as nn
import torch.nn.functional as F
from easy_tpp.config_factory import DataSpecConfig


class GRUTPPEncoder(nn.Module):
    """
    This module implements a neural marked temporal point process using a Gated Recurrent Unit (GRU).

    The process involves encoding the input sequence of events with their associated marks and timestamps.
    Each event is processed through a GRU.

    Args:
        config (DataSpecConfig): Configuration specifying data properties.
        emb_dim (int): Dimension of event type embeddings.
        hidden_dim (int): Hidden dimension of GRU.
        mlp_dim (int): Dimension of the MLP layer.
        num_layers (int): Number of GRU layers.
        dropout (float): Dropout rate.

    Methods:
        forward(batch): Defines the forward pass of the encoder.
            Args:
                batch (dict): Dictionary containing preprocessed tensors:
                    - type_seqs: Event type sequences [batch_size, seq_len].
                    - time_seqs: Absolute time sequences [batch_size, seq_len].
                    - time_delta_seqs: Time delta sequences [batch_size, seq_len].
                    - sequence_length: Original sequence lengths before padding.
    """

    def __init__(
        self,
        config: DataSpecConfig,
        emb_dim=32,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
    ):
        super(GRUTPPEncoder, self).__init__()
        self.config = config

        # Event type embedding layer
        self.event_embedding = nn.Embedding(
            config.num_event_types + 1,  # Add 1 for padding token
            emb_dim,
            padding_idx=config.pad_token_id,
        )

        # GRU for sequence encoding
        self.input_dim = emb_dim + 2  # embedding + time_delta + time_since_start
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # # Output projection
        # self.mlp = nn.Linear(hidden_dim, mlp_dim)
        # self.event_type_projection = nn.Linear(mlp_dim, config.num_event_types)
        # self.time_projection = nn.Linear(mlp_dim, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        """Forward pass of the encoder

        Args:
            batch: Dictionary containing preprocessed tensors:
                - type_seqs: Event type sequences [batch_size, seq_len]
                - time_seqs: Absolute time sequences [batch_size, seq_len]
                - time_delta_seqs: Time delta sequences [batch_size, seq_len]
                - sequence_length: Original sequence lengths before padding

        Returns:
            hidden_states: Hidden states of the GRU [batch_size, seq_len, hidden_dim]
        """
        batch_size = batch["type_seqs"].size(0)

        # Get embeddings for event types
        event_embeddings = self.event_embedding(batch["type_seqs"])  # [B, L, E]
        event_embeddings = self.dropout(event_embeddings)

        # Stack temporal features
        time_features = torch.stack(
            [batch["time_delta_seqs"], batch["time_seqs"]], dim=-1
        )  # [B, L, 2]

        # Combine event and temporal features
        combined_features = torch.cat(
            [event_embeddings, time_features], dim=-1
        )  # [B, L, E+2]

        # sequence length before packing
        max_seq_len = batch["type_seqs"].size(1)

        # Pack padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(
            combined_features,
            batch["sequence_length"].cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # Process through GRU
        packed_output, _ = self.gru(packed_input)

        # Unpack sequence
        hidden_states, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            padding_value=0,
            total_length=max_seq_len,
        )

        # print("Shape of combined features: ", combined_features.shape)
        # print("Shape of hidden states: ", hidden_states.shape)

        return hidden_states
