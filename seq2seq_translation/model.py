import torch 
import torch.nn as nn
import math

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 10*-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha and bias are learnable params
        self.beta = nn.Parameter(torch.ones(features))
        
    
    def forward(self, x):
         # x: (batch, seq_lem, hidden_size)
         # Keep the dimension for broadcasting
         mean  = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
         # Keep the dimension for broadcasting
         std = x.std(dim = -1, keepdim = True) # (B, seq_lem, 1)
         # eps , negligibly small value to avoid zero div
         return self.alpha * (x - mean) / (std + self.eps) + self.bias
     
     
     
    
class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout = float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1 # Think its d_ff = mlp_ratio * d_model - Not sure though
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        # (B, seq_len, d_model) -> (B, seq_lem, d_ff) -> (B,, seq_len, d_model)
        x_out = self.linear_1(x)
        x_out = self.dropout(x_out)
        x_out = self.linear2(x_out)
        return x_out
    

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model # embed_dim or hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    
    def forward(self, x):
        # (B, seq_len) -> (B, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper   # sqrt(d_k)
        
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len: int, dropout_p_value:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout_p_value) # p
        
        
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a tensor of shape (seq_len)
        
        # ****** Sine and Cosine Positional embeddings
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(dim = 1) # (seq_len, 1)
        
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 **(2i / d_model))
        
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(dim=0) # (1, seq_len, d_model)
        
        # Register the positional encoding as a buffer as its not a learnale param (sine | cos)
        self.register_buffer('pe', pe)     # This tensor is part of the model’s state, but don’t try to optimize it. Just keep it around, move it to the correct device, and save/load it with everything else.
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)  # (batch, seq_len, d_model)
        return self.dropout(x)
        
        
class ResidualConnection(nn.Module):
    
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = dropout
        self.norm = LayerNormalization(features)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
    

class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # d_model should be divisible bby h
        assert d_model % h ==0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo
        self.dropout = dropout
    
    @staticmethod
    def attention(query ,key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula 
        # (B, h, seq_len, d_k) -> (B, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_scores = attention_scores.softmax(dim = -1) # (B, h, seq_len, seq_len) 
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        # Back to d_k
        # (B, h, seq_len, seq_len) -> (B, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask):
        query = self.w_q(q)  #(B, seq_len, d_model) -> (B, seq_len, d_model)
        key =   self.w_k(k)  # (B, seq_len, d_model) -> (B, seq_len, d_model)
        value = self.w_v(v)  # (B, seq_len, d_model) -> (B, seq_len, d_model)
        
        # (B, seq_len, d_model) -> (B, seq_len, h, d_k)  -> (B, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key   = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine em all heads
        # (B, h, seq_len, d_k) -> (B, seq_len, h, d_k) -> (B, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        # Multiply by Wo
        # (B, seq_len, d_model) -> (B, seq_len, d_model)
        return self.w_o(x)
    

class EncoderBlock(nn.Module):
    
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, 
                 dropout: float) -> None:
        super().__init__()
        self.attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda: self.attention_block(x, x, x, src_mask))  # q, k, v
        x = self.residual_connections[1](x, self.feed_forward_block)  # sublayer - FeedForwardBlock
        return x
        

class Encoder(nn.Module):
    
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)
        
    
class DecoderBlock(nn.Module):
    
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, 
                 dropout: float) -> None:
        super().__init__()
        self.attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block  # q from decoder, k, v from encoder output
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda: self.attention_block(x, x, x, tgt_mask))  # q, k, v
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)) # sublayer - cross attention block
        x = self.residual_connections[2](x, self.feed_forward_block)  # sublayer - FeedForwardBlock
        return x
    
    
# Will have Masked Multi Head Attention
class Decoder(nn.Module):
    
    def __init__(self, features:int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers # N
        self.norm = LayerNormalization(features)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
        

class ProjectionLayer(nn.Module):
    
    def __init__(self, d_model , vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x) -> None:
        # (B, seq_len, d_model) -> (B, seq_len, vocab_size)
        return self.proj(x)


class Transformer(nn.Module):
    
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings,
                 tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, 
                 projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    def encode(self, src, src_mask):
        # (B, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor,
               tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # B, seq_len, d_model
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (B, seq_len, vocab_size)
        return self.projection_layer(x)


def build_transformer(src_vocab_size:int, tgt_vocab_size: int, src_seq_len: int, 
                      tgt_seq_len: int, d_model: int = 512, N:int = 6, num_heads: int = 8, 
                      dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    
    # Create embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    # Create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    # Create the decoder blocks
    decoder_blocks = []
    for i in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
        
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, projection_layer)
    
    # Initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer