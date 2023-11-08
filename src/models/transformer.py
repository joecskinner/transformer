
import tensorflow as tf
from src.models.shared_embedding import SharedEmbedding
from src.models.positional_encoding import PositionalEncoding


class ResidualAndLayerNorm(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.norm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x, sublayer_output):
        x = self.add([x, sublayer_output])
        return self.norm(x)


class BaseMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.rln = ResidualAndLayerNorm()


class MultiHeadSelfAttention(BaseMultiHeadAttention):
    def call(self, x, masked=False):
        output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=masked
        )
        return self.rln(output, x)


class MultiHeadCrossAttention(BaseMultiHeadAttention):
    def call(self, x, encoder_output):
        attention_output, scores = self.mha(
            query=x,
            value=encoder_output,
            key=encoder_output,
            return_attention_scores=True
        )
        self.last_attention_scores = scores
        return self.rln(attention_output, x)


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc1 = tf.keras.layers.Dense(d_ff, activation='relu')
        self.fc2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.rln = ResidualAndLayerNorm()

    def call(self, x):
        output = self.fc1(x)
        output = self.fc2(output)
        output = self.dropout(output)
        return self.rln(output, x)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, d_ff, num_heads, dropout):
        super().__init__()
        # TODO: key_dim = d_model // num_heads
        key_dim = d_model

        self.mhsa = MultiHeadSelfAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout)

        self.ff = FeedForward(d_model, d_ff, dropout)

    def call(self, x):
        x = self.mhsa(x)
        return self.ff(x)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, d_model, d_ff, shared_embedding, num_layers, num_heads, dropout, vocab_size):
        super().__init__()
        # TODO: actually share the embedding
        # When you do so, you can remove vocab_size as init param
        # self.shared_embedding = shared_embedding
        self.shared_embedding = SharedEmbedding(
            vocab_size=vocab_size,
            d_model=d_model)

        self.positional_encoding = PositionalEncoding(
            max_length=2048,
            d_model=d_model)

        self.dropout = tf.keras.layers.Dropout(dropout)

        self.encoder_stack = [
            EncoderLayer(d_model=d_model,
                         d_ff=d_ff,
                         dropout=dropout,
                         num_heads=num_heads)
            for _ in range(num_layers)
        ]

    def call(self, x):
        x = self.shared_embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for encoder_layer in self.encoder_stack:
            x = encoder_layer(x)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, d_ff, num_heads, dropout):
        super().__init__()
        # TODO: key_dim = d_model // num_heads
        key_dim = d_model

        self.mmhsa = MultiHeadSelfAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout)

        self.mhca = MultiHeadCrossAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout)

        self.ff = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout)

    def call(self, x, encoder_output):
        x = self.mmhsa(x, masked=True)
        x = self.mhca(x, encoder_output)

        self.last_attention_scores = self.mhca.last_attention_scores

        return self.ff(x)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, d_model, d_ff, shared_embedding, num_layers, num_heads, dropout, vocab_size):
        super().__init__()
        # TODO: actually share the embedding
        # When you do so, you can remove vocab_size as init param
        # self.shared_embedding = shared_embedding
        self.shared_embedding = SharedEmbedding(
            vocab_size=vocab_size,
            d_model=d_model)

        self.positional_encoding = PositionalEncoding(
            max_length=2048,
            d_model=d_model)

        self.dropout = tf.keras.layers.Dropout(dropout)

        self.decoder_stack = [
            DecoderLayer(d_model=d_model,
                         d_ff=d_ff,
                         dropout=dropout,
                         num_heads=num_heads)
            for _ in range(num_layers)
        ]

    def call(self, x, encoder_output):
        x = self.shared_embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for decoder_layer in self.decoder_stack:
            x = decoder_layer(x, encoder_output)

        self.last_attention_scores = self.decoder_stack[-1].last_attention_scores
        return x


class Transformer(tf.keras.Model):
    def __init__(self, *, vocab_size, input_vocab_size, target_vocab_size, d_model, d_ff, dropout, num_layers, num_heads):
        super().__init__()

        # TODO: use a single tokenizer with a single vocab_size
        #
        # As per the original paper:
        #
        #   "Sentences were encoded using byte-pair encoding, which has a shared source-
        #   target vocabulary of about 37000 tokens."
        #
        # Once I have done this I can remove the seperate input_vocab_size and target_vocab_size

        # TODO: actually share the embedding
        # Instantiate SharedEmbedding
        # self.shared_embedding = SharedEmbedding(
        #     vocab_size=vocab_size,
        #     d_model=d_model)

        # Instantiate Encoder and Decoder
        self.encoder = Encoder(
            d_model=d_model,
            d_ff=d_ff,
            shared_embedding=None,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            vocab_size=input_vocab_size)

        self.decoder = Decoder(
            d_model=d_model,
            d_ff=d_ff,
            shared_embedding=None,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            vocab_size=target_vocab_size)

        # # Create Dense layer with shared weights
        # self.linear = tf.keras.layers.Dense(vocab_size, use_bias=True,
        #                                     kernel_initializer=tf.keras.initializers.Constant(self.shared_embedding.get_embedding().numpy()))
        # # Ensure the kernel of the dense layer is the same as the embedding layer.
        # # This is in accordance with the "3.4 Embeddings and Softmax" section
        # # from the "Attention Is All You Need" paper by Vaswani et al. (2017).
        # self.linear.kernel = self.shared_embedding.get_embedding()
        # Note: for now I am not going to share the weights until I have an end-to-end pipeline working first.
        # TODO: address the above.
        self.linear = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        encoder_input, decoder_input = inputs

        # Pass encoder_input through the Encoder
        encoder_output = self.encoder(encoder_input)

        # Pass encoder_output and decoder_input through the Decoder
        decoder_output = self.decoder(decoder_input, encoder_output)

        # Pass decoder_output through the linear layer
        logits = self.linear(decoder_output)

        # Drop the mask from the logits
        try:
            del logits._keras_mask
        except AttributeError:
            pass

        return logits
