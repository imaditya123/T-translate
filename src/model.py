import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Attention Model
 
class Attention(nn.Module):
  def __init__(self,embed_dim,head_dim,dropout_rate):
    super(Attention,self).__init__()

    self.query=nn.Linear(embed_dim,head_dim)
    self.key=nn.Linear(embed_dim,head_dim)
    self.value=nn.Linear(embed_dim,head_dim)
    self.dropout=nn.Dropout(dropout_rate)


  def forward(self,query,key,value,mask=None):
    d_k=query.size(-1)

    q=self.query(query)
    k=self.key(key)
    v=self.value(value)

    scores=q @ k.transpose(1,2) /math.sqrt(d_k)
    if mask is not None:
      # mask = mask.unsqueeze(1)
      scores = scores.masked_fill(mask == 0, float('-inf'))

    weights=F.softmax(scores,dim=-1)
    weights=self.dropout(weights)
    out=weights @ v
    return out
  
#  Multihead attention

class MultiheadAttention(nn.Module):
  def __init__(self,embed_dim,head_size,dropout_rate):
    super(MultiheadAttention,self).__init__()

    # Validate that embed_dim is divisible by head_size
    assert embed_dim % head_size == 0, "embed_dim must be divisible by head_size"

    self.head_dim=embed_dim//head_size
    self.embed_dim=embed_dim
    self.head_size=head_size

    self.attn_heads=nn.ModuleList([Attention(embed_dim,self.head_dim,dropout_rate) for _ in range(head_size)])
    self.out_layer=nn.Linear(self.head_dim*head_size,embed_dim)
    self.dropout=nn.Dropout(dropout_rate)


  def forward(self,query,key,value,mask=None):

    out=torch.cat([h(query,key,value,mask) for h in self.attn_heads],dim=-1)
    out=self.dropout(self.out_layer(out))
    return out
  
# Feed forward

class FeedForward(nn.Module):
  def __init__(self,embed_dim,filter_dim,dropout_rate):
    super(FeedForward,self).__init__()

    self.linear1=nn.Linear(embed_dim,filter_dim)
    self.gelu=nn.GELU()
    self.linear2=nn.Linear(filter_dim,embed_dim)
    self.dropout=nn.Dropout(dropout_rate)

  def forward(self,x):
    x=self.linear1(x)
    x=self.gelu(x)
    x=self.linear2(x)
    x=self.dropout(x)

    return x
  
# Embedding

class Embeddings(nn.Module):
  def __init__(self,vocab_size,embed_dim,max_position_embeddings,dropout_rate):
    super(Embeddings,self).__init__()

    self.token_embeddings=nn.Embedding(vocab_size,embed_dim)
    self.position_embeddings=nn.Embedding(max_position_embeddings,embed_dim)

    self.layer_norm=nn.LayerNorm(embed_dim,eps=1e-6)
    self.dropout=nn.Dropout(dropout_rate)

  def forward(self,input_ids):

    seq_len=input_ids.size(-1)
    position_ids=torch.arange(seq_len,dtype=torch.long,device=input_ids.device).unsqueeze(0)

    token_embeddings=self.token_embeddings(input_ids)
    position_embeddings=self.position_embeddings(position_ids)

    embeddings=token_embeddings+position_embeddings
    embeddings=self.layer_norm(embeddings)
    embeddings=self.dropout(embeddings)

    return embeddings

 

# Encoder layer
class TransformerEncoderLayer(nn.Module):
  def __init__(self,embed_dim,head_size,dropout_rate,filter_dim):
    super(TransformerEncoderLayer,self).__init__()

    self.attn_layer_norm=nn.LayerNorm(embed_dim,eps=1e-6)
    self.attn=MultiheadAttention(embed_dim,head_size,dropout_rate)
    self.attn_dropout=nn.Dropout(dropout_rate)

    self.ffwd_layer_norm=nn.LayerNorm(embed_dim,eps=1e-6)
    self.ffwd=FeedForward(embed_dim,filter_dim,dropout_rate)
    self.ffwd_dropout=nn.Dropout(dropout_rate)

  def forward(self,x,mask=None):
    y=self.attn_layer_norm(x)
    y=self.attn(y,y,y,mask)
    y=self.attn_dropout(y)
    x=x+y

    y=self.ffwd_layer_norm(x)
    y=self.ffwd(y)
    y=self.attn_dropout(y)
    x=x+y

    return x
  
# Encoder

class TransformerEncoder(nn.Module):
  def __init__(self,embed_dim,head_size,dropout_rate,vocab_size,max_position_embeddings,n_layers):
    super(TransformerEncoder,self).__init__()

    self.embeddings=Embeddings(vocab_size,embed_dim,max_position_embeddings,dropout_rate)
    self.layers=nn.ModuleList([TransformerEncoderLayer(embed_dim,head_size,dropout_rate,max_position_embeddings) for _ in range(n_layers)])

  def forward(self,x,mask=None):
    x=self.embeddings(x)
    out=x
    for layer in self.layers:
      out=layer(out,mask)

    return out
  
# Decoder layer
class TransformerDecoderLayer(nn.Module):
  def __init__(self,embed_dim,head_size,dropout_rate,filter_dim):
    super(TransformerDecoderLayer,self).__init__()

    self.attn_layer_norm=nn.LayerNorm(embed_dim,eps=1e-6)
    self.attn=MultiheadAttention(embed_dim,head_size,dropout_rate)
    self.attn_dropout=nn.Dropout(dropout_rate)

    self.enc_dec_layer_norm=nn.LayerNorm(embed_dim,eps=1e-6)
    self.enc_dec=MultiheadAttention(embed_dim,head_size,dropout_rate)
    self.enc_dec_dropout=nn.Dropout(dropout_rate)

    self.ffwd_layer_norm=nn.LayerNorm(embed_dim,eps=1e-6)
    self.ffwd=FeedForward(embed_dim,filter_dim,dropout_rate)
    self.ffwd_dropout=nn.Dropout(dropout_rate)

  def forward(self,x,mask,encoder_out=None,encoder_mask=None):
    y=self.attn_layer_norm(x)
    y=self.attn(y,y,y,mask)
    y=self.attn_dropout(y)
    x=x+y

    if encoder_out is not None:
      y=self.enc_dec_layer_norm(x)
      y=self.enc_dec(y,encoder_out,encoder_out,encoder_mask)
      y=self.enc_dec_dropout(y)
      x=x+y

    y=self.ffwd_layer_norm(x)
    y=self.ffwd(y)
    y=self.attn_dropout(y)
    x=x+y

    return x

#  Deocder

class TransformerDecoder(nn.Module):
  def __init__(self,embed_dim,head_size,dropout_rate,vocab_size,max_position_embeddings,n_layers):
    super(TransformerDecoder,self).__init__()

    self.embeddings=Embeddings(vocab_size,embed_dim,max_position_embeddings,dropout_rate)
    self.layers=nn.ModuleList([TransformerDecoderLayer(embed_dim,head_size,dropout_rate,max_position_embeddings) for _ in range(n_layers)])

  def forward(self,x,encoder_out=None,mask=None,encoder_mask=None):
    x=self.embeddings(x)
    out=x
    for layer in self.layers:
      out=layer(out,mask,encoder_out,encoder_mask)
    return out
  
# Transformer

class TransformerModel(nn.Module):
    def __init__(self, i_vocab_size, t_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, hidden_dim=2048, dropout=0.1):
        """
        input_dim: Vocabulary size for source language.
        output_dim: Vocabulary size for target language.
        """
        super(TransformerModel, self).__init__()
        self.encoder = TransformerEncoder(vocab_size=i_vocab_size,max_position_embeddings= d_model,head_size= num_heads,embed_dim=  hidden_dim,dropout_rate= dropout,n_layers=num_encoder_layers)
        self.decoder = TransformerDecoder(vocab_size=t_vocab_size, max_position_embeddings= d_model,head_size= num_heads,embed_dim=  hidden_dim,dropout_rate= dropout,n_layers=num_decoder_layers)

        self.out_layer=nn.Linear(hidden_dim,t_vocab_size)


    def make_tgt_mask(self, tgt):
        """
        Create a mask to hide future tokens in the target sequence.
        """
        tgt_seq_len = tgt.size(1)
        # Lower triangular matrix: 1's where allowed, 0's elsewhere.
        mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len), device=tgt.device)).bool()
        return mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: Source sequence tensor (batch_size, src_seq_len)
        tgt: Target sequence tensor (batch_size, tgt_seq_len)
        src_mask: Optional mask for source (e.g., padding mask)
        tgt_mask: Optional mask for target (e.g., future masking)
        """
        memory = self.encoder(src, src_mask)
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)
        output = self.decoder(tgt, memory, tgt_mask,src_mask)
        return self.out_layer(output)