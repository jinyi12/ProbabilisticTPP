from .encoders.gru import GRUTPPEncoder
from .decoders.rmtpp import RMTPPDecoder, RMTPPLoss
from .decoders.vaetpp import VAETPPDecoder, VAETPPLoss
from torch import nn


class TPPModel(nn.Module):
    def __init__(self, config, hidden_dim, mlp_dim, device):
        super(TPPModel, self).__init__()
        self.encoder = GRUTPPEncoder(config, hidden_dim=hidden_dim)
        self.decoder = RMTPPDecoder(
            hidden_dim=hidden_dim,
            num_event_types=config.num_event_types,
            mlp_dim=mlp_dim,
            device=device,
        )
        self.criterion = RMTPPLoss(
            device=device, ignore_index=config.pad_token_id, decoder=self.decoder
        )

    def forward(self, batch):
        hidden_states = self.encoder(batch)
        decoder_output = self.decoder(hidden_states)
        return decoder_output

    def compute_loss(self, batch, decoder_output):
        time_loss, mark_loss, total_loss = self.criterion(
            decoder_output,
            batch["time_delta_seqs"],
            batch["type_seqs"],
            batch["sequence_length"],
        )
        return time_loss, mark_loss, total_loss


class VAETPPModel(nn.Module):
    def __init__(
        self,
        config,
        hidden_dim,
        latent_dim,
        mlp_dim,
        device,
        beta_start=0,
        beta_end=1,
        beta_steps=1000,
        l1_lambda=0,
        l2_lambda=0,
    ):
        super(VAETPPModel, self).__init__()
        self.encoder = GRUTPPEncoder(config, hidden_dim=hidden_dim)
        self.decoder = VAETPPDecoder(
            n_in=hidden_dim,
            n_hid=mlp_dim,
            z_dim=latent_dim,
            num_event_types=config.num_event_types,
        )
        self.criterion = VAETPPLoss(
            device=device,
            ignore_index=config.pad_token_id,
            decoder=self.decoder,
            beta_start=beta_start,
            beta_end=beta_end,
            n_steps=beta_steps,
            l1_lambda=l1_lambda,
            l2_lambda=l2_lambda,
        )

    def forward(self, batch):
        hidden_states = self.encoder(batch)
        decoder_output = self.decoder(hidden_states)
        return decoder_output

    def compute_loss(self, batch, decoder_output):
        time_loss, mark_loss, elbo = self.criterion(
            decoder_output,
            batch["time_delta_seqs"],
            batch["type_seqs"],
            batch["sequence_length"],
        )
        return time_loss, mark_loss, elbo
