import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, NamedTuple


class DecoderOutput(NamedTuple):
    """Container for decoder outputs to make the interface cleaner"""

    time_output: torch.Tensor  # Predicted time until next event
    # time_delta: torch.Tensor  # Predicted time delta
    mark_logits: torch.Tensor  # Log probabilities for marks
    # intensity_integral: torch.Tensor  # Integral of intensity function
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
        time_output = self.time_projection(mlp_output)
        base_intensity = self.intensity_b

        # Compute mark probabilities
        mark_logits = self.mark_projection(mlp_output)
        mark_logits = F.log_softmax(mark_logits, dim=-1)

        return DecoderOutput(
            time_output=time_output,
            mark_logits=mark_logits,
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
        decoder: RMTPPDecoder,  # Add decoder as an argument
        event_weights: Optional[torch.Tensor] = None,
        ignore_index: int = None,
    ):
        super(RMTPPLoss, self).__init__()
        self.device = device
        self.decoder = decoder  # Store decoder reference
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
        mask = (
            torch.arange(max_len, device=sequence_length.device)[None, :]
            < sequence_length[:, None]
        )
        return mask

    def compute_intensity_integral(
        self, time_output: torch.Tensor, time_delta: torch.Tensor
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
        integral = (1.0 / self.decoder.intensity_w) * (
            torch.exp(
                time_output
                + self.decoder.intensity_w * time_delta
                + self.decoder.intensity_b
            )
            - torch.exp(time_output + self.decoder.intensity_b)
        )
        return integral

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

        # print("Time target shape: ", time_target.shape)

        # Create sequence mask
        mask = self.create_sequence_mask(sequence_length, time_target.size(1))

        # compute intensity integral
        time_output = decoder_output.time_output.squeeze()
        base_intensity = self.decoder.intensity_b
        intensity_integral = self.compute_intensity_integral(time_output, time_target)

        # negative log likelihood of TPP
        time_loglikelihood = (
            time_output
            + self.decoder.intensity_w * time_target
            + base_intensity
            + -1.0 * intensity_integral
        )

        # Apply mask to all relevant tensors
        # Compute masked time loss
        masked_time_loss = time_loglikelihood * mask
        negative_time_loglikelihood = -torch.sum(masked_time_loss) / mask.sum()

        # Compute masked mark loss
        mark_loss_per_event = self.event_loss(
            decoder_output.mark_logits.view(-1, decoder_output.mark_logits.size(-1)),
            mark_target.view(-1),
        )
        mark_loss_per_event = mark_loss_per_event.view_as(mark_target) * mask
        mark_loss = torch.sum(mark_loss_per_event) / mask.sum()

        # Combined loss
        total_loss = negative_time_loglikelihood + mark_loss

        return negative_time_loglikelihood, mark_loss, total_loss
