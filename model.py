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
        attention_function = lambda x: self.self_attention(x, x, x, mask)
        x = self.residual_connections[0](x, attention_function)

        feed_forward_function = lambda x: self.feed_forward_block(x)
        x = self.residual_connections[1](x, feed_forward_function)

        return x



class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):


    def __init__(self, masked_self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward: FeedForwardBlock, dropout: float):
        super().__init__()

        self.masked_self_attention_block = masked_self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward = feed_forward


        self.residual_connections = nn.ModuleList(
            ResidualConnection(dropout) for _ in range(3)
        )



    def forward(self, x, encoder_output, src_mask, target_mask):

        first_layer_forward = lambda x: self.masked_self_attention_block(x, x, x, target_mask)
        x = self.residual_connections[0](x, first_layer_forward)

        second_layer_forward = lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)
        x = self.residual_connections[1](x, second_layer_forward)

        x = self.residual_connections[2](x, self.feed_forward)


        return x


class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()

        self.layers = layers
        self.normalize = LayerNormalization()

    def forward(self, x, encoder_output, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)
        return self.normalize(x)


class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocabulary_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocabulary_size = vocabulary_size

        self.linear = nn.Linear(
            in_features=d_model,
            out_features=vocabulary_size
        )

    def forward(self, x):
        x = self.linear(x)
        return torch.log_softmax(self.linear(x), dim=-1)



class Transformer(nn.Module):

    def __init__(
            self,
            input_embedding: InputEmbeddings,
            out_embedding: InputEmbeddings,
            source_positional_embedding: PositionalEncoding,
            target_positional_embedding: PositionalEncoding,
            encoder: Encoder,
            decoder: Decoder,
            projection_layer: ProjectionLayer
    ):
        super().__init__()

        self.input_embedding = input_embedding
        self.out_embedding = out_embedding
        self.source_positional_embedding = source_positional_embedding
        self.target_positional_embedding = target_positional_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.projection_layer = projection_layer



    def encode(self, x, mask):


        matrix_embedding = self.input_embedding(x)
        matrix_embedding_with_pos = self.source_positional_embedding(matrix_embedding)

        encoder_output = self.encoder(matrix_embedding_with_pos, mask)

        return encoder_output


    def decode(self, x, encoder_output, source_mask, target_mask):

        matrix_embedding = self.out_embedding(x)
        matrix_embedding_with_pos = self.target_positional_embedding(matrix_embedding)

        decoder_output = self.decoder(matrix_embedding_with_pos, encoder_output, source_mask, target_mask)

        return decoder_output


    def project(self, x):
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048

) -> Transformer:

    # the embedding layers
    src_embedding = InputEmbeddings(
        d_model=d_model,
        vocab_size=src_vocab_size
    )
    tgt_embedding = InputEmbeddings(
        d_model=d_model,
        vocab_size=tgt_vocab_size
    )

    # Positional encoding layers
    src_pos = PositionalEncoding(
        d_model=d_model,
        max_seq_len=src_seq_len,
        dropout=dropout
    )
    tgt_pos = PositionalEncoding(
        d_model=d_model,
        max_seq_len=tgt_seq_len,
        dropout=dropout
    )

    # The encoder blocks
    encoder_blocks = []
    for _ in range(N):

        encoder_self_attention = MultiHeadAttentionBlock(
            d_model=d_model,
            h=h,
            dropout=dropout
        )

        feed_forward_block = FeedForwardBlock(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        encoder_block = EncoderBlock(
            self_attention=encoder_self_attention,
            feed_forward_block=feed_forward_block,
            dropout=dropout
        )

        encoder_blocks.append(encoder_blocks)


    # the decoder blocks
    decoder_blocks = []
    for _ in range(N):

        decoder_self_attention = MultiHeadAttentionBlock(
            d_model=d_model,
            h=h,
            dropout=dropout
        )

        decoder_cross_attention = MultiHeadAttentionBlock(
            d_model=d_model,
            h=h,
            dropout=dropout
        )

        feed_forward_block = FeedForwardBlock(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        decoder_block = DecoderBlock(
            masked_self_attention_block=decoder_self_attention,
            cross_attention_block=decoder_cross_attention,
            feed_forward=feed_forward_block,
            dropout=dropout
        )

        decoder_blocks.append(decoder_block)



