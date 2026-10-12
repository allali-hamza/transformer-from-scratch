import torch 
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )

    def forward(self, x):
        return self.embedding(x) * (self.d_model ** 0.5)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)

        # on crée une matrice de max_seq_len ligne de d_model colonne 
        pe = torch.zeros(max_seq_len, d_model)  

        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)    

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)




class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**(-6)):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))   # multiplier
        self.bias = nn.Parameter(torch.zeros(1))   # sommer 

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):

        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)


        self.linear1 = nn.Linear(
            in_features=d_model,
            out_features=d_ff
        )

        self.relu = nn.ReLU()

        self.linear2 = nn.Linear(
            in_features=d_ff,
            out_features=d_model
        )


    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# my_model = FeedForwardBlock(
#     d_model=512,
#     d_ff=2048,
#     dropout=0.1
# )

# x = torch.randn(16, 20, 512)

# # print(x.shape)

# # y = my_model.forward(x)

# # print(y.shape)



class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model should be divisible by h"

        self.d_k = d_model // h

        self.W_q = nn.Linear(
            in_features=d_model,
            out_features=d_model
        )

        self.W_k = nn.Linear(
            in_features=d_model,
            out_features=d_model
        )

        self.W_v = nn.Linear(
            in_features=d_model,
            out_features=d_model
        )

        self.W_o = nn.Linear(
            in_features=d_model,
            out_features=d_model
        )

        self.dropout = nn.Dropout(p=dropout)


    @staticmethod
    def attention(query, key, value, mask, dropout):

        d_k = query.shape[-1]

        scores = torch.matmul(query, key.transpose(-2, -1))

        scores = scores / math.sqrt(d_k)

        if mask is not None:
            scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(
            input=scores,
            dim=-1
        )

        if dropout is not None:
            attention_weights = dropout(attention_weights)

        output = attention_weights @ value

        return output, attention_weights        




    def forward(self, q, k, v, mask):
        query = self.W_q(q)
        key = self.W_k(k)
        value = self.W_v(v)

        query = query.reshape(query.shape[0], query.shape[1], self.h, self.d_k)
        query = query.transpose(1, 2)

        key = key.reshape(key.shape[0], key.shape[1], self.h, self.d_k)
        key = key.transpose(1, 2)

        value = value.reshape(value.shape[0], value.shape[1], self.h, self.d_k)
        value = value.transpose(1, 2)


        x, self.attetion_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).reshape(x.shape[0], -1, self.d_model)

        return self.W_o(x)

    


class ResidualConnection(nn.Module):

    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()


    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))



class EncoderBlock(nn.Module):

    def __init__(self, self_attention: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout), ResidualConnection(dropout)]
        )

    def forward(self, x, mask):
        attention_function = lambda x: self.self_attention(x, x, x, self.mask)
        x = self.residual_connections[0](x, attention_function)

        feed_forward_function = lambda x: self.feed_forward_block(x)
        x = self.residual_connections[1](x, feed_forward_function)

        return x



class Encoder(nn.Module):

    def __ini__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)