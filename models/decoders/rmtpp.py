import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, NamedTuple


class DecoderOutput(NamedTuple):
    """Container for decoder outputs to make the interface cleaner"""

    time_delta: torch.Tensor  # Predicted time delta
    mark_logits: torch.Tensor  # Log probabilities for marks
    intensity_integral: torch.Tensor  # Integral of intensity function
    base_intensity: torch.Tensor  # Base intensity values


class RMTPPDecoder(nn.Module):
    """
    Pure decoder implementation for RMTPP.
    Computes marked intensity integral and predicts both time and marks.

    Args:
        hidden_dim (int): Dimension of the hidden state from encoder
        num_event_types (int): Number of possible event types/marks
        mlp_dim (int): Dimension of the MLP layer
        device (torch.device): Device to place the model parameters on
    """

    def __init__(
        self, hidden_dim: int, num_event_types: int, mlp_dim: int, device: torch.device
    ):
        super(RMTPPDecoder, self).__init__()

        # Intensity function parameters
        self.intensity_w = nn.Parameter(
            torch.tensor(0.1, dtype=torch.float, device=device)
        )
        self.intensity_b = nn.Parameter(
            torch.tensor(0.1, dtype=torch.float, device=device)
        )

        # MLP for hidden state processing
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_dim), nn.Sigmoid())

        # Output projections
        self.mark_projection = nn.Linear(mlp_dim, num_event_types)
        self.time_projection = nn.Linear(mlp_dim, 1)

    def compute_intensity_integral(
        self, base_intensity: torch.Tensor, time_delta: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the integral of the intensity function over [0, time_delta].

        Args:
            base_intensity: Base intensity from hidden states [batch_size, seq_len, 1]
            time_delta: Time until next event [batch_size, seq_len, 1]

        Returns:
            integral: Integrated intensity [batch_size, seq_len, 1]
        """
        # Compute integral using the closed form solution
        integral = (1.0 / self.intensity_w) * (
            torch.exp(base_intensity + self.intensity_w * time_delta + self.intensity_b)
            - torch.exp(base_intensity + self.intensity_b)
        )
        return integral

    def forward(self, hidden_states: torch.Tensor) -> DecoderOutput:
        """
        Forward pass of the decoder.

        Args:
            hidden_states: Hidden states from encoder [batch_size, seq_len, hidden_dim]

        Returns:
            DecoderOutput containing:
                - time_delta: Predicted time until next event [batch_size, seq_len, 1]
                - mark_logits: Log probabilities for each mark [batch_size, seq_len, num_event_types]
                - intensity_integral: Integral of intensity function [batch_size, seq_len, 1]
                - base_intensity: Base intensity values [batch_size, seq_len, 1]
        """
        # Process hidden states through MLP
        mlp_output = self.mlp(hidden_states)

        # Get base intensity and time prediction
        base_intensity = self.time_projection(mlp_output)
        time_delta = torch.exp(base_intensity)  # Ensure positive time delta

        # Compute intensity integral
        intensity_integral = self.compute_intensity_integral(base_intensity, time_delta)

        # Compute mark probabilities
        mark_logits = self.mark_projection(mlp_output)
        mark_logits = F.log_softmax(mark_logits, dim=-1)

        return DecoderOutput(
            time_delta=time_delta,
            mark_logits=mark_logits,
            intensity_integral=intensity_integral,
            base_intensity=base_intensity,
        )


class RMTPPLoss(nn.Module):
    """
    Loss computation for RMTPP.
    Handles sequence padding consistently with RNN packed sequences.
    """

    def __init__(
        self,
        device: torch.device,
        event_weights: Optional[torch.Tensor] = None,
        ignore_index: int = None,
    ):
        super(RMTPPLoss, self).__init__()
        self.device = device
        self.ignore_index = ignore_index
        self.event_loss = (
            nn.NLLLoss(
                weight=event_weights, reduction="none", ignore_index=ignore_index
            )
            if event_weights is not None
            else nn.NLLLoss(reduction="none", ignore_index=ignore_index)
        )

    def create_sequence_mask(
        self, sequence_length: torch.LongTensor, max_len: int
    ) -> torch.BoolTensor:
        """
        Create a boolean mask for variable length sequences.

        Args:
            sequence_length: Length of each sequence [batch_size]
            max_len: Maximum sequence length

        Returns:
            mask: Boolean mask [batch_size, max_len]
        """
        batch_size = sequence_length.size(0)
        mask = (
            torch.arange(max_len, device=sequence_length.device)[None, :]
            < sequence_length[:, None]
        )
        return mask

    def forward(
        self,
        decoder_output: DecoderOutput,
        time_target: torch.Tensor,
        mark_target: torch.Tensor,
        sequence_length: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the combined loss using decoder outputs.

        Args:
            decoder_output: Output from the decoder
            time_target: True time deltas [batch_size, seq_len]
            mark_target: True marks/event types [batch_size, seq_len]
            sequence_length: Length of each sequence [batch_size]

        Returns:
            tuple: (time_loss, mark_loss, total_loss)
        """
        # Create sequence mask
        mask = self.create_sequence_mask(sequence_length, time_target.size(1))

        # Apply mask to all relevant tensors
        base_intensity = decoder_output.base_intensity.squeeze(-1)
        intensity_integral = decoder_output.intensity_integral.squeeze(-1)

        # Compute masked time loss
        time_components = base_intensity - intensity_integral
        masked_time_loss = time_components * mask
        time_loss = -torch.sum(masked_time_loss) / mask.sum()

        # Compute masked mark loss
        mark_loss_per_event = self.event_loss(
            decoder_output.mark_logits.view(-1, decoder_output.mark_logits.size(-1)),
            mark_target.view(-1),
        )
        mark_loss_per_event = mark_loss_per_event.view_as(mark_target) * mask
        mark_loss = torch.sum(mark_loss_per_event) / mask.sum()

        # Combined loss
        total_loss = time_loss + mark_loss

        return time_loss, mark_loss, total_loss
