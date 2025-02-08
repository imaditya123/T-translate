import unittest
import torch

from src.model import (Attention, MultiheadAttention, FeedForward, Embeddings,
                         TransformerEncoderLayer, TransformerEncoder,
                         TransformerDecoderLayer, TransformerDecoder,
                         TransformerModel)

class TestTransformerComponents(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 4
        self.embed_dim = 8
        self.head_size = 2
        self.filter_dim = 16
        self.vocab_size = 10
        self.dropout_rate = 0.1
        self.n_layers = 2
        self.src = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        self.tgt = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        self.mask = torch.ones((self.batch_size, self.seq_len), dtype=torch.bool)

    def test_attention(self):
        attn = Attention(self.embed_dim, self.embed_dim // self.head_size, self.dropout_rate)
        qkv = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        output = attn(qkv, qkv, qkv, self.mask)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim // self.head_size))

    def test_multihead_attention(self):
        attn = MultiheadAttention(self.embed_dim, self.head_size, self.dropout_rate)
        qkv = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        output = attn(qkv, qkv, qkv, self.mask)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))

    def test_feedforward(self):
        ff = FeedForward(self.embed_dim, self.filter_dim, self.dropout_rate)
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        output = ff(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))

    def test_embeddings(self):
        emb = Embeddings(self.vocab_size, self.embed_dim, self.seq_len, self.dropout_rate)
        output = emb(self.src)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))

    def test_transformer_encoder_layer(self):
        encoder_layer = TransformerEncoderLayer(self.embed_dim, self.head_size, self.dropout_rate, self.filter_dim)
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        output = encoder_layer(x, self.mask)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))

    def test_transformer_encoder(self):
        encoder = TransformerEncoder(self.embed_dim, self.head_size, self.dropout_rate, self.vocab_size, self.seq_len, self.n_layers)
        output = encoder(self.src, self.mask)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))

    def test_transformer_decoder_layer(self):
        decoder_layer = TransformerDecoderLayer(self.embed_dim, self.head_size, self.dropout_rate, self.filter_dim)
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        enc_out = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        output = decoder_layer(x, self.mask, enc_out, self.mask)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))

    def test_transformer_decoder(self):
        decoder = TransformerDecoder(self.embed_dim, self.head_size, self.dropout_rate, self.vocab_size, self.seq_len, self.n_layers)
        enc_out = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        output = decoder(self.tgt, enc_out, self.mask, self.mask)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))

    def test_transformer_model(self):
        model = TransformerModel(i_vocab_size=self.vocab_size, t_vocab_size=self.vocab_size, d_model=self.embed_dim, num_heads=self.head_size,
                                 num_encoder_layers=self.n_layers, num_decoder_layers=self.n_layers, hidden_dim=self.embed_dim, dropout=self.dropout_rate)
        output = model(self.src, self.tgt, self.mask, self.mask)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.vocab_size))

if __name__ == "__main__":
    unittest.main()
