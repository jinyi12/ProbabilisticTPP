from .encoders.gru import GRUTPPEncoder
from .decoders.rmtpp import RMTPPDecoder, RMTPPLoss
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
