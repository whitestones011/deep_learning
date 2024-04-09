import math

import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_length=512):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # initialize positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)

        # initialize position of indices in the sequence
        # 'unsqueeze' function aligns the tensor shape with the shape of the input embeddings
        position = torch.arange(0, max_seq_length, dtype=float).unsqueeze(1)

        # scaler for positional indices
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=float) *
            -(math.log(10000.0) / d_model)
        )
        # apply scaler to positional indices combined with sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # unsqueeze to add batch dimension
        pe = pe.unsqueeze(0)

        # set matrix as non-trainable using register_buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # add the positional encodings to the whole sequence embeddings contained in tensor x
        x = x + self.pe[:, :x.size(1)]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        # number of attention heads handling embedding size head_dim
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # linear transformations for attention inputs
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        
        # final concatenated output
        self.output_layer = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        # splits the input across the heads
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0,2,1,3).contiguous().view(batch_size * self.num_heads, -1 , self.head_dim)

    def compute_attention(self, query, key, mask=None):
        # calculates the attention weights inside each head
        scores = torch.matmul(query, key.permute(1,2,0))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e9"))
        attention_weights = nn.functional.softmax(scores, dim=-1)
        return attention_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.split_heads(
            self.query_linear(query), batch_size
        )
        key = self.split_heads(
            self.key_linear(key), batch_size
        )
        value = self.split_heads(
            self.value_linear(value), batch_size
        )

        attention_weights = self.compute_attention(query, key, mask)

        output = torch.matmul(attention_weights, value)
        output = output.view(
            batch_size, 
            self.num_heads, 
            -1, 
            self.head_dim
        ).permute(0,2,1,3).contiguous().view(batch_size, -1 , self.d_model)

        return self.output_layer(output)


class FeedForwardTransformation(nn.Module):
    """
    parms: d_model the embedinggs dimensionality
    params: d_ff the dimension between linear layers
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(
            self.relu(
                self.fc1(x)
            )
        )


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardTransformation(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.attention(x,x,x,mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.poistional_encoding = PositionalEncoder(d_model, max_seq_length)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.poistional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        # the causal (masked) self-attention and cross-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardTransformation(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, causal_mask, cross_mask):        
        self_attn_output = self.self_attention(x,x,x,causal_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        # x - decoder information flow, becomes cross-attention query
        # y - encoder output, becomes cross-attention key and values
        cross_attn_output = self.cross_attention(x,y,y,cross_mask)
        
        x = self.norm2(x + self.dropout(cross_attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoder(d_model, max_seq_length)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        # linear layer (head) for next-word prediction
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, self_mask):
        x = self.embedding(x)
        x = self.poistional_encoding(x)
        for layer in self.layers:
            x = layer(x, self_mask)
        # Apply the forward pass through the model head
        x = self.fc(x)
        return nn.functional.log_softmax(x, dim=-1)


class ClassifierHead(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        logits = self.fc(x)
        return nn.functional.log_softmax(logits, dim=-1)
        
