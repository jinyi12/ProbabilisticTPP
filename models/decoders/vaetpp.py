import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, Tuple, Optional, NamedTuple


class VAEDecoderOutput(NamedTuple):
    """Container for decoder outputs to make the interface cleaner"""

    time_logits: torch.Tensor  # Predicted time until next event
    mark_logits: torch.Tensor  # Log probabilities for marks
    base_intensity: torch.Tensor  # Base intensity values
    mu: torch.Tensor
    logvar: torch.Tensor


class VAETPPDecoder(nn.Module):
    def __init__(self, n_in, n_hid, z_dim, num_event_types):
        """
        Args:
            n_in (int): Dimension of the input features (hidden states from encoder).
            n_hid (int): Dimension of the hidden layer.
            z_dim (int): Dimension of the latent space.
            num_event_types (int): Number of event types for classification.
        """
        super().__init__()

        self.fc1 = nn.Linear(n_in, n_hid)  # Input to hidden
        self.fc21 = nn.Linear(n_hid, z_dim)  # Hidden to latent mean
        self.fc22 = nn.Linear(n_hid, z_dim)  # Hidden to latent log variance
        self.fc3 = nn.Linear(z_dim, n_hid)  # Latent to hidden
        self.fc4_time = nn.Linear(n_hid, 1)  # Hidden to time output
        self.fc4_mark = nn.Linear(n_hid, num_event_types)  # Hidden to mark logits

        # Intensity function parameters
        self.intensity_w = nn.Parameter(torch.tensor(0.1, dtype=torch.float))
        self.intensity_b = nn.Parameter(torch.tensor(0.1, dtype=torch.float))

        # Activation functions
        self.softplus = nn.Softplus()
        self.silu = nn.SiLU()

    def encode(self, x, eps=1e-8):
        """Encoder forward pass: encodes input data to the latent space."""
        h1 = self.silu(self.fc1(x))  # Activation with SiLU
        mu = self.fc21(h1)  # Mean of the latent space
        logvar = self.fc22(h1)  # Log variance of the latent space
        logvar_pos = self.softplus(logvar) + eps  # Ensure positive variance
        return mu, logvar_pos

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + epsilon * stddev."""
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)  # Sample noise
        return mu + eps * std

    def decode(self, z):
        """Decoder forward pass: decodes latent space data to output space."""
        h3 = F.sigmoid(self.fc3(z))  # Activation with Sigmoid for logit outputs
        time_logits = self.fc4_time(h3)  # Predict time logits until next event
        mark_logits = self.fc4_mark(h3)  # Predict event type logits
        return time_logits, mark_logits

    def forward(self, hidden_states: torch.Tensor) -> VAEDecoderOutput:
        """
        Forward pass: encodes, reparameterizes, and decodes input data.

        Args:
            hidden_states (torch.Tensor): Input features (e.g., hidden states from an encoder).

        Returns:
            VAEDecoderOutput: Predicted time, marks, and latent parameters.
        """
        mu, logvar = self.encode(hidden_states)  # Encode to latent space
        z = self.reparameterize(mu, logvar)  # Sample latent representation
        time_logits, mark_logits = self.decode(z)  # Decode to outputs

        mark_logits = F.log_softmax(mark_logits, dim=-1)  # Log probabilities for marks

        # Return predictions and latent space parameters
        return VAEDecoderOutput(
            time_logits=time_logits,
            mark_logits=mark_logits,
            base_intensity=self.intensity_b,
            mu=mu,
            logvar=logvar,
        )


class VAETPPLoss(nn.Module):
    def __init__(
        self,
        device: torch.device,
        decoder: VAETPPDecoder,
        event_weights: Optional[torch.Tensor] = None,
        ignore_index: int = None,
        beta_start=0.0,
        beta_end=1.0,
        n_steps: int = 1000,
        warmup_steps: int = 1000,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0,
    ):
        super(VAETPPLoss, self).__init__()
        self.device = device
        self.decoder = decoder
        self.current_step = 0
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.n_steps = n_steps
        self.warmup_steps = warmup_steps
        self.ignore_index = ignore_index
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.event_loss = (
            nn.NLLLoss(
                weight=event_weights, reduction="none", ignore_index=ignore_index
            )
            if event_weights is not None
            else nn.NLLLoss(reduction="none", ignore_index=ignore_index)
        )

    def compute_l1_regularization(self, model: nn.Module) -> torch.Tensor:
        """Compute L1 regularization for all parameters."""
        l1_reg = torch.tensor(0.0, requires_grad=True, device=self.device)
        for param in model.parameters():
            l1_reg = l1_reg + torch.norm(param, 1)
        return self.l1_lambda * l1_reg

    def compute_l2_regularization(self, model: nn.Module) -> torch.Tensor:
        """Compute L2 regularization for all parameters."""
        l2_reg = torch.tensor(0.0, requires_grad=True, device=self.device)
        for param in model.parameters():
            l2_reg = l2_reg + torch.norm(param, 2)
        return self.l2_lambda * l2_reg

    def beta_schedule(
        self,
        step: int,
        n_steps: int,
        beta_start: float,
        beta_end: float,
        warmup_steps: int,
    ) -> float:
        """Linearly anneal beta from `beta_start` to `beta_end` over `n_steps`."""
        if step <= warmup_steps:
            return beta_start
        else:
            return min(
                beta_end,
                beta_start + (beta_end - beta_start) * (step - warmup_steps) / n_steps,
            )

    def cosine_beta_annealing_schedule(
        self,
        step: int,
        n_steps: int,
        beta_start: float,
        beta_end: float,
        warmup_steps: int,
    ) -> float:
        """Cyclic cosine annealing schedule for beta.

        Args:
            step (int): Current training step.
            n_steps (int): Total number of training steps for one cycle.
            beta_start (float): Initial beta value.
            beta_end (float): Final beta value.
            warmup_steps (int): Number of warmup steps.
        """
        if step <= warmup_steps:
            return beta_start
        else:
            cycle_step = (step - warmup_steps) % n_steps
            progress = cycle_step / n_steps
            return beta_end + 0.5 * (beta_start - beta_end) * (
                1 + torch.cos(torch.tensor(progress * 3.141592653589793))
            )

    def create_sequence_mask(
        self, sequence_length: torch.LongTensor, max_len: int
    ) -> torch.BoolTensor:
        mask = (
            torch.arange(max_len, device=sequence_length.device)[None, :]
            < sequence_length[:, None]
        )
        return mask

    def compute_intensity_integral(
        self, time_logits: torch.Tensor, time_delta: torch.Tensor
    ) -> torch.Tensor:
        integral = (1.0 / self.decoder.intensity_w) * (
            torch.exp(
                torch.clamp(
                    time_logits
                    + self.decoder.intensity_w * time_delta
                    + self.decoder.intensity_b,
                    max=10,
                )
            )
            - torch.exp(torch.clamp(time_logits + self.decoder.intensity_b, max=10))
        )
        return integral

    def tppLoss(
        self,
        time_logits: torch.Tensor,
        mark_logits: torch.Tensor,
        time_target: torch.Tensor,
        mark_target: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        base_intensity = self.decoder.intensity_b
        intensity_integral = self.compute_intensity_integral(time_logits, time_target)

        # Time log-likelihood
        time_loglikelihood = (
            time_logits
            + self.decoder.intensity_w * time_target
            + base_intensity
            - intensity_integral
        )
        masked_time_loss = time_loglikelihood * mask
        negative_time_loglikelihood = -torch.sum(masked_time_loss) / mask.sum()

        # Mark loss
        mark_loss_per_event = self.event_loss(
            mark_logits.view(-1, mark_logits.size(-1)), mark_target.view(-1)
        )
        mark_loss_per_event = mark_loss_per_event.view_as(mark_target) * mask
        mark_loss = torch.sum(mark_loss_per_event) / mask.sum()

        return negative_time_loglikelihood, mark_loss

    def KLDLoss(self, mu, logvar, mask):
        """KL divergence with masking."""
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        maskedKLD = KLD * mask
        return maskedKLD.sum() / mask.sum()

    def forward(
        self,
        decoder_output: VAEDecoderOutput,
        time_target: torch.Tensor,
        mark_target: torch.Tensor,
        sequence_length: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = self.create_sequence_mask(sequence_length, time_target.size(1))

        # Loss computation
        time_loss, mark_loss = self.tppLoss(
            decoder_output.time_logits.squeeze(),
            decoder_output.mark_logits,
            time_target,
            mark_target,
            mask,
        )
        kld = self.KLDLoss(decoder_output.mu, decoder_output.logvar, mask)

        # add regularization
        l1_reg = self.compute_l1_regularization(self.decoder)
        l2_reg = self.compute_l2_regularization(self.decoder)

        recon = time_loss + mark_loss + l1_reg + l2_reg

        # beta = self.beta_schedule(
        #     self.current_step,
        #     self.n_steps,
        #     self.beta_start,
        #     self.beta_end,
        #     self.warmup_steps,
        # )

        beta = self.cosine_beta_annealing_schedule(
            self.current_step,
            self.n_steps,
            self.beta_start,
            self.beta_end,
            self.warmup_steps,
        )

        elbo = recon + beta * kld

        self.current_step += 1

        return time_loss, mark_loss, elbo
