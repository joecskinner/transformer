import numpy as np
import tensorflow as tf


def positional_encoding(max_length, d_model):
    """
    Compute the positional encodings for the input sequence positions.

    Args:
        max_length: int, maximum sequence length.
        d_model: int, embedding dimension.

    Returns:
        positional_encodings: numpy array of shape (max_len, d_model).
    """
    # Create a zero matrix to hold the positional encodings
    positional_encodings = np.zeros((max_length, d_model))

    # Generate a sequence representing positions in the sequence
    pos = np.arange(max_length)[:, np.newaxis]

    # Generate a sequence representing the dimension indices up to d_model/2
    # Note: We only generate up to half the dimensions because of the alternating
    # sine and cosine positional encoding structure
    dimensions = np.arange(d_model // 2)[np.newaxis, :]

    # Calculate the denominators for the sine and cosine terms according to the
    # Transformer's positional encoding formula
    denominator_terms = 1 / np.power(10000, (2 * dimensions / d_model))

    # Apply the sine function for even indexed dimensions (0, 2, 4, ...)
    positional_encodings[:, 0::2] = np.sin(pos * denominator_terms)

    # Apply the cosine function for odd indexed dimensions (1, 3, 5, ...)
    positional_encodings[:, 1::2] = np.cos(pos * denominator_terms)

    return tf.cast(positional_encodings, dtype=tf.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_length, d_model):
        """
        Initialize the PositionalEncoding layer.

        Parameters:
        - max_length (int): Maximum length of the sequences for which 
                            positional encoding will be generated.
        - d_model (int): Dimension of the embedding space.

        Attributes:
        - d_model: Dimension of the embedding space.
        - pos_encoding: A tensor containing the precomputed positional encodings.
        """
        super().__init__()
        self.d_model = d_model

        # Compute the positional encodings once during layer initialization.
        # This is possible since the positional encodings are deterministic
        # and can be precomputed for all possible positions in a sequence up to
        # 'max_length'.
        self.pos_encoding = positional_encoding(max_length, d_model)

    def call(self, x):
        """
        Forward pass for the PositionalEncoding layer.

        Parameters:
        - x (Tensor): Input tensor, typically the output of an embedding layer.

        Returns:
        - Tensor: The input tensor with positional encodings added.
        """
        # Compute the length of the input sequence(s) by retrieving the size
        # along the second dimension (axis=1).
        # This is needed to select the corresponding slice from the precomputed
        # positional encoding tensor.
        length = tf.shape(x)[1]

        # Add the positional encodings to the input tensor.
        # The positional encodings are added to the embedded token vectors
        # to provide the model with information about the position of a token
        # within a sequence.
        return x + self.pos_encoding[tf.newaxis, :length, :]

    def compute_mask(self, inputs, mask=None):
        """
        Propagate the input tensor's mask.

        As PositionalEncoding only adds positional information without changing token count,
        the incoming mask remains unchanged.

        Args:
            inputs: Input tensor, typically from an embedding layer.
            mask: Boolean mask tensor for input entries.

        Returns:
            tf.Tensor: Unchanged mask tensor.
        """
        return mask
