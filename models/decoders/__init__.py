import torch
import torch.nn as nn
import torch.nn.functional as F


class RMTPPDecoder(nn.Module):
    """
    Recurrent Marked Temporal Point Process (RMTPP) Decoder.

    This decoder implements the intensity function and mark distribution for the RMTPP model.
    It matches both the original RMTPP implementation and the loss function structure from RMTPPLoss.

    Args:
        hidden_dim (int): Dimension of the hidden state from encoder
        num_event_types (int): Number of possible event types/marks
        device (torch.device): Device to place the model parameters on
    """

    def __init__(self, hidden_dim: int, num_event_types: int, device: torch.device):
        super(RMTPPDecoder, self).__init__()

        # Intensity function parameters
        self.intensity_w = nn.Parameter(
            torch.tensor(0.1, dtype=torch.float, device=device)
        )
        self.intensity_b = nn.Parameter(
            torch.tensor(0.1, dtype=torch.float, device=device)
        )

        # Mark prediction layers
        self.mark_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_event_types),
        )

        # Time prediction layer
        self.time_projection = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        """
        Forward pass of the decoder.

        Args:
            hidden_states (torch.Tensor): Hidden states from the encoder [batch_size, hidden_dim]

        Returns:
            tuple: (time_output, mark_output)
                - time_output: Predicted time until next event [batch_size, 1]
                - mark_output: Log probabilities for each event type [batch_size, num_event_types]
        """
        # Project hidden states to time prediction
        time_output = self.time_projection(hidden_states)

        # Project hidden states to mark predictions
        mark_logits = self.mark_projection(hidden_states)
        mark_output = F.log_softmax(mark_logits, dim=-1)

        return time_output, mark_output

    def compute_loss(
        self,
        time_output: torch.Tensor,
        mark_output: torch.Tensor,
        time_target: torch.Tensor,
        mark_target: torch.Tensor,
        event_loss_fn: nn.Module,
        alpha: float = 1.0,
    ) -> tuple:
        """
        Compute the loss for both time and mark predictions.

        Args:
            time_output (torch.Tensor): Predicted time deltas [batch_size, 1]
            mark_output (torch.Tensor): Predicted mark log probabilities [batch_size, num_event_types]
            time_target (torch.Tensor): True time deltas [batch_size]
            mark_target (torch.Tensor): True marks/event types [batch_size]
            event_loss_fn (nn.Module): Loss function for event type prediction
            alpha (float): Weight for the time loss component

        Returns:
            tuple: (time_loss, mark_loss, total_loss)
        """
        # Time loss computation (matches RMTPPLoss implementation)
        time_loss = -1 * torch.mean(
            time_output.squeeze()
            + self.intensity_w * time_target
            + self.intensity_b
            + (
                torch.exp(time_output.squeeze() + self.intensity_b)
                - torch.exp(
                    time_output.squeeze()
                    + self.intensity_w * time_target
                    + self.intensity_b
                )
            )
            / self.intensity_w
        )

        # Mark loss computation
        mark_loss = event_loss_fn(mark_output, mark_target)

        # Combined loss
        total_loss = alpha * time_loss + mark_loss

        return time_loss, mark_loss, total_loss

    def get_intensity(
        self, time_delta: torch.Tensor, last_hidden: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the intensity function at given time deltas.

        Args:
            time_delta (torch.Tensor): Time since last event [batch_size]
            last_hidden (torch.Tensor): Hidden state at last event [batch_size, hidden_dim]

        Returns:
            torch.Tensor: Intensity values at the given times [batch_size]
        """
        base_intensity = self.time_projection(last_hidden).squeeze()
        intensity = torch.exp(
            base_intensity + self.intensity_w * time_delta + self.intensity_b
        )
        return intensity
