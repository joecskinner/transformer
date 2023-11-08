import tensorflow as tf


class SharedEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        """
        Initialize the SharedEmbedding layer.

        Parameters:
        - vocab_size (int): Size of the vocabulary, i.e., the number of unique tokens.
        - d_model (int): Dimension of the embedding space.

        Attributes:
        - vocab_size: Size of the vocabulary.
        - d_model: Dimension of the embedding space.
        - embedding: Embedding layer that converts token IDs to vectors.
        """
        super(SharedEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                   output_dim=d_model,
                                                   mask_zero=True)

    def call(self, x):
        """
        Forward pass for the SharedEmbedding layer.

        Parameters:
        - x (Tensor): Input tensor, typically token IDs.

        Returns:
        - Tensor: The embedded input sequence, scaled by the square root of 'd_model'.
        """
        x = self.embedding(x)
        return self.scale_by_d_model(x)

    def scale_by_d_model(self, x):
        """Scale the input tensor by the square root of d_model."""
        return x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))

    def get_embedding(self):
        """Retrieve the embedding matrix scaled by the square root of 'd_model'."""
        return self.scale_by_d_model(self.embedding.get_weights()[0])

    def compute_mask(self, *args, **kwargs):
        """
        Compute the mask from the Embedding layer.

        The mask indicates where the token ID is 0 (typically used for padding). 
        This method delegates the mask computation to the internal `Embedding` layer 
        which generates a mask when `mask_zero=True`.

        Parameters:
        - args: Positional arguments, usually the input tensor.
        - kwargs: Keyword arguments, may include the input mask.

        Returns:
        - Tensor or None: A boolean mask tensor for valid tokens (True) and padding tokens (False).
        """
        return self.embedding.compute_mask(*args, **kwargs)
