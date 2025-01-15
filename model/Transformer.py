import torch
import torch.nn as nn

from model.Components import Encoder, Decoder

class Transformer(nn.Module):
    def __init__(self, vocab_dim, hidden_dim, num_heads=2, enc_depth=6, dec_depth=6):
        super(Transformer, self).__init__()
        self.vocab_dim = vocab_dim
        self.input_dim = hidden_dim
        self.output_dim = hidden_dim
        self.num_heads = num_heads

        self.encoder = nn.ModuleList([Encoder(hidden_dim, hidden_dim, num_heads) for _ in range(enc_depth)])
        self.decoder = nn.ModuleList([Decoder(hidden_dim, hidden_dim, num_heads) for _ in range(dec_depth)])

        self.fc = nn.Linear(hidden_dim, vocab_dim)

    def forward(self, x, target, teacher_forcing_ratio=0.5):
        encoder_out = self._encode(x)

        sos_token = -1 * torch.ones(x.size(0), 1, self.input_dim)
        last_token = sos_token

        decoder_out = self._decode(last_token, encoder_out)
        last_token = decoder_out[:, -1, :].unsqueeze(1)

        for token_pos in range(target.size(1) - 1):
            if torch.rand(1) < teacher_forcing_ratio:
                last_token = target[:, token_pos, :].unsqueeze(1)
            else:
                last_token = self._decode(decoder_out, encoder_out)[:,-1,:].unsqueeze(1)

            decoder_out = torch.cat([decoder_out, last_token], dim=1)
            last_token = self.fc(decoder_out)

        return last_token
        
    def _encode(self, x):
        for enc in self.encoder:
            x = enc(x)
        return x
    
    def _decode(self, x, encoder_out):
        for dec in self.decoder:
            x = dec(x, encoder_out)
        return x
