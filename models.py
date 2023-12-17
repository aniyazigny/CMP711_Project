import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, input_tokens):
        return self.embedding(input_tokens)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, device, max_len=512):
        super(TransformerModel, self).__init__()
        self.device = device
        # Token Embedding
        self.embedding = TokenEmbedding(vocab_size, d_model)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

        # Transformer Decoder
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_decoder_layers)

        # Linear layer for final output
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).float()
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        # Token Embedding
        src_embedded = self.embedding(src)
        tgt_embedded = self.embedding(tgt)

        # Positional Encoding
        src_encoded = self.positional_encoding(src_embedded)
        tgt_encoded = self.positional_encoding(tgt_embedded)

        # Transformer Encoder
        encoder_output = self.transformer_encoder(src_encoded)

        # Masking for Transformer Decoder
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0))
        tgt_mask = tgt_mask.to(self.device)

        # Transformer Decoder
        decoder_output = self.transformer_decoder(tgt_encoded, encoder_output, tgt_mask)

        # Linear layer for final output
        output = self.fc_out(decoder_output)

        return output




