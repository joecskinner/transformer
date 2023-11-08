import tensorflow as tf


class Translator():
    """A class used to translate sentences using a provided tokenizer and transformer model."""

    def __init__(self, tokenizer, transformer):
        self.tokenizer = tokenizer
        self.transformer = transformer

    def __call__(self, sentence: str, max_length: int) -> str:
        """
        Translates the provided sentence into the target language.

        Note: The transformer is trained to expect sequences in batch form. As the
              translator class is just for demonstrations purposes I have written 
              this class to work with a single sentence at a time.

        Args:
            sentence: A string containing the sentence to be translated.
            max_length: The maximum length of the translated sentence in terms
                        of the number of tokens.

        Returns:
            The translated sentence.
        """
        # Convert string to tensor
        input = tf.constant(sentence)

        # Add batch dimension
        inputs = input[tf.newaxis]

        # Tokenize input
        inputs = self.tokenizer.tokenize(inputs)

        # Convert RaggedTensor to Tensor
        inputs = inputs.to_tensor()

        # Retrieve [START] annd [END] tokens from the tokenizer
        start = self.tokenizer.tokenize([''])[0][0][tf.newaxis]
        end = self.tokenizer.tokenize([''])[0][1][tf.newaxis]

        # Initialize TensorArray for output tokens using the [START] token
        output = tf.TensorArray(
            dtype=tf.int32, size=0, dynamic_size=True)
        output = output.write(0, start)

        # Iterate over the maximum length of the translated sentence
        for i in tf.range(max_length):
            # Prepare the output tokens for the transformer
            outputs = tf.transpose(output.stack())

            # Predict the next token
            logits = self.transformer((inputs, outputs), training=False)
            predicted_token = tf.argmax(logits[:, -1:, :], axis=-1)
            predicted_token_casted = tf.cast(predicted_token, tf.int32)
            output = output.write(i+1, predicted_token_casted[0])

            # Break if [END] token is predicted
            if predicted_token_casted == end:
                break

        # Convert output tokens to tensor and remove batch dimension
        output = tf.transpose(output.stack())
        # Convert to RaggedTensor for detokenization
        output_ragged = tf.RaggedTensor.from_tensor(output)
        # Detokenize output tokens to text
        translated_sentence = self.tokenizer.detokenize(output_ragged)[0]

        return translated_sentence
