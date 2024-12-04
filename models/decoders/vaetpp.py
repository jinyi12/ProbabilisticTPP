import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, Tuple, Optional, NamedTuple


class VAEDecoderOutput(NamedTuple):
    """Container for decoder outputs to make the interface cleaner"""

    time_output: torch.Tensor  # Predicted time until next event
    # time_delta: torch.Tensor  # Predicted time delta
    mark_logits: torch.Tensor  # Log probabilities for marks
    # intensity_integral: torch.Tensor  # Integral of intensity function
    base_intensity: torch.Tensor  # Base intensity values
    mu: torch.Tensor
    logvar: torch.Tensor

class VAETPPDecoder(nn.Module):
    """
    Decoder implementation for TPP using VAE.
    Computes marked intensity integral and predicts both time and marks with VAE.
    
    Args:
        hidden_dim (int): Dimension of the hidden state from encoder
        latent_dim (int): Dimension of latent space
        num_event_types (int): Number of possible event types/marks
        mlp_dim (int): Dimension of the MLP layer
        device (torch.device): Device to place the model parameters on
    """
    def __init__(self, 
        hidden_dim: int, 
        latent_dim: int,
        num_event_types: int, 
        mlp_dim: int, 
        device: torch.device
    ):
        super(VAETPPDecoder, self).__init__()               
        self.latent_dim = latent_dim
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        
        # Intensity function parameters
        self.intensity_w = nn.Parameter(torch.tensor(0.1, dtype=torch.float, device=device))
        self.intensity_b = nn.Parameter(torch.tensor(0.1, dtype=torch.float, device=device))
        
        # MLP for hidden state processing
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.Sigmoid()
        )
        
        # Output projections
        self.mark_projection = nn.Linear(mlp_dim, num_event_types)
        self.time_projection = nn.Linear(mlp_dim, 1)
        
        self.device = device

    def reparameterize(self, mu, logvar):
        """
        Implements: z = mu + epsilon*stdev.
        
        Args: 
            mu: Mean of the latent distribution
            logvar: Log var of the latent distribution
            
        Returns:
            Sampled latent vector
        """
        stdev = torch.exp(0.5*logvar)
        eps = torch.randn_like(stdev)
        return mu + eps*stdev
    def decode(self, z):
        """Implements decoder forward pass."""
        h3 = F.relu(self.fc3(z))
        return self.mlp(h3)
    def encode(self, x):
        """Encoder forward pass."""
        return self.fc21(x), self.fc22(x)

    def forward(self, hidden_states: torch.Tensor) -> VAEDecoderOutput:
        """
        Forward pass of the VAE decoder.
        
        Args:
            hidden_states: Hidden states from encoder [batch_size, seq_len, hidden_dim]
            
        Returns:
            VAEDecoderOutput containing:
                - time_delta: Predicted time until next event
                - mark_logits: Log probabilities for each mark
                - intensity_integral: Integral of intensity function
                - base_intensity: Base intensity values
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Encode to latent space, sample + decode latent vector, send through MLP
        mu, logvar = self.encode(hidden_states)
        z = self.reparameterize(mu, logvar)
        mlp_output = self.decode(z)
        
        # Get base intensity and time prediction
        time_output = self.time_projection(mlp_output)
        base_intensity = self.intensity_b
        
        # Compute mark probabilities
        mark_logits = self.mark_projection(mlp_output)
        mark_logits = F.log_softmax(mark_logits, dim=-1)
        
        # get elbo, recon kld losses     
        return VAEDecoderOutput(
            time_output=time_output,
            mark_logits=mark_logits,
            base_intensity=base_intensity,
            mu = mu,
            logvar = logvar
        )

class VAETPPLoss(nn.Module):
    """
    Loss computation for VAETPP.
    Combines TPP likelihood loss with VAE ELBO loss.
    """
    def __init__(
        self,
        device: torch.device,
        decoder: VAETPPDecoder,  
        event_weights: Optional[torch.Tensor] = None,
        ignore_index: int = None,
        beta: float = 1.0,  # Weight for KLD term
    ):
        super(VAETPPLoss, self).__init__()
        self.device = device
        self.decoder = decoder
        self.beta = beta
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
        """Create a boolean mask for variable length sequences."""
        mask = (
            torch.arange(max_len, device=sequence_length.device)[None, :]
            < sequence_length[:, None]
        )
        return mask

    def compute_intensity_integral(
        self, time_output: torch.Tensor, time_delta: torch.Tensor
    ) -> torch.Tensor:
        """Compute the integral of the intensity function over [0, time_delta]."""
        integral = (1.0 / self.decoder.intensity_w) * (
            torch.exp(
                time_output
                + self.decoder.intensity_w * time_delta
                + self.decoder.intensity_b
            )
            - torch.exp(time_output + self.decoder.intensity_b)
        )
        return integral

    def tppLoss(
        self,
        time_output: torch.Tensor,
        mark_logits: torch.Tensor,
        time_target: torch.Tensor,
        mark_target: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute TPP-specific losses (time and mark components)"""
        # Compute intensity integral
        base_intensity = self.decoder.intensity_b
        intensity_integral = self.compute_intensity_integral(time_output, time_target)

        # Negative log likelihood of TPP
        time_loglikelihood = (
            time_output
            + self.decoder.intensity_w * time_target
            + base_intensity
            - intensity_integral
        )

        # Apply mask and compute time loss
        masked_time_loss = time_loglikelihood * mask
        negative_time_loglikelihood = -torch.sum(masked_time_loss) / mask.sum()

        # Compute mark loss
        mark_loss_per_event = self.event_loss(
            mark_logits.view(-1, mark_logits.size(-1)),
            mark_target.view(-1),
        )
        mark_loss_per_event = mark_loss_per_event.view_as(mark_target) * mask
        mark_loss = torch.sum(mark_loss_per_event) / mask.sum()

        return negative_time_loglikelihood, mark_loss

    def KLDLoss(self, mu, logvar, mask):
        """
        Compute the KL divergence loss w masking
        """
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = torch.sum(KLD, dim=-1)  # shape like [batch_size, seq_len]
        # Now KLD and mask have same dimensions for multiplication
        maskedKLD = KLD * mask
        return maskedKLD.sum() / mask.sum()
    
    def forward(
        self,
        decoder_output: VAEDecoderOutput,
        time_target: torch.Tensor,
        mark_target: torch.Tensor,
        sequence_length: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the combined VAE-TPP loss.

        Args:
            decoder_output: Output from the VAE decoder
            time_target: True time deltas [batch_size, seq_len]
            mark_target: True marks/event types [batch_size, seq_len]
            sequence_length: Length of each sequence [batch_size]

        Returns:
            tuple: (time_loss, mark_loss, kl_loss, total_loss, elbo_loss)
        """
        # Create sequence mask
        mask = self.create_sequence_mask(sequence_length, time_target.size(1))

        # Compute TPP-specific losses
        time_loss, mark_loss = self.tppLoss(
            decoder_output.time_output.squeeze(),
            decoder_output.mark_logits,
            time_target,
            mark_target,
            mask,
        )

        # KL Divergence loss
        kld = self.KLDLoss(decoder_output.mu, decoder_output.logvar, mask)

        # Reconstruction loss = sum of time and mark losses
        recon = time_loss + mark_loss
        elbo = recon + self.beta * kld

        return time_loss, mark_loss, elbo #elbo is total loss