import os
from collections import Counter, defaultdict
import numpy as np
import tensorflow as tf


def calculate_pair_frequencies(token_count):
    """
    Calculate the frequency of each pair of tokens in the tokenized words.

    Parameters:
    - token_count (dict): A dictionary of tokenized words and their respective frequencies.

    Returns:
    - pair_counts (defaultdict of int): A dictionary with pairs of tokens as keys and their frequency as values.
    """
    pair_counts = defaultdict(
        int)  # Initialize a default dictionary to hold pair frequencies.

    # Loop through each token and its frequency in the input token_count dictionary.
    for k, v in token_count.items():
        # Loop through each adjacent character pair in the token and update the pair's frequency.
        for pair in [(k[i], k[i+1]) for i in range(len(k)-1)]:
            # Add the token's frequency to the pair's total count.
            pair_counts[pair] += v

    return pair_counts


def merge(token_counts_in, vocab):
    """
    Merge the most frequent pair of tokens and update the vocabulary and token counts.

    Parameters:
    - token_counts_in (dict): A dictionary of tokenized words and their respective frequencies.
    - vocab (list): The current vocabulary list.

    Returns:
    - token_counts_out (dict): The updated token counts after the merge.
    - vocab (list): The updated vocabulary after the merge.
    - bool: True if a merge occurred, False otherwise.
    """
    # Calculate the frequency of each pair of tokens in the tokenized words.
    pair_frequencies = calculate_pair_frequencies(token_counts_in)

    # Check if there are no pairs to merge.
    if len(pair_frequencies) < 1:
        return token_counts_in, vocab, False, {}

    # Find the most frequent pair of tokens.
    new_token = max(pair_frequencies, key=pair_frequencies.get)

    # Add new token to vocabulary
    vocab.append("".join(new_token))

    # Store merge rule
    merge_target = "".join(new_token)
    merge_rule = {new_token: merge_target}

    # Initialize an empty dictionary to hold the updated token counts.
    token_counts_out = {}

    # Loop through each tokenized word and its frequency in the input token_count dictionary.
    for k, v in token_counts_in.items():
        new_key = []
        i = 0
        # Loop through each character in the token.
        while i < len(k):

            # Check if the current and next characters form the new token, and append the new token if true.
            if i == len(k) - 1:
                new_key.append(k[i])
                i += 1
            elif k[i] + k[i+1] == "".join(new_token):
                new_key.append("".join(new_token))
                # Skip the next character since we merged it with the current one.
                i += 2
            else:
                new_key.append(k[i])
                i += 1
        # Update the token counts with the new tokenized word and its frequency.
        token_counts_out[tuple(new_key)] = v

    return token_counts_out, vocab, True, merge_rule


def bpe_from_dataset(dataset, vocab_size, reserved_tokens):
    """
    Generate a vocabulary using Byte Pair Encoding (BPE) algorithm.

    Parameters:
    - dataset (list of str): A list of sentences from which to build the vocabulary.
    - vocab_size (int): The desired size of the output vocabulary.
    - reserved_tokens (list of str): A list of tokens that should be included in the vocabulary.

    Returns:
    - vocab (list): The generated vocabulary.
    """
    # Count the frequency of each word in the dataset.
    word_count = Counter()
    for sentence in dataset:
        words = sentence.split()
        # Prepend "Ġ" (U+0120) to each word except the first one in a sentence
        for i, word in enumerate(words):
            if i != 0:  # Do not prepend "Ġ" to the first word
                word = "Ġ" + word

            # Separate the period from the word unless it's isolated.
            if word.endswith('.'):
                word = word[:-1]  # Exclude the period
                word_count[word] += 1
                word_count["."] += 1  # Count the period as a separate word
            else:
                word_count[word] += 1

    # Create the initial vocabulary with reserved tokens and unique characters from the dataset.
    vocab = reserved_tokens + \
        sorted({letter for word in word_count.keys() for letter in word})

    # Decompose words into characters and initialize token count.
    token_counts = {tuple([*k]): v for k, v in word_count.items()}

    # Store all merge rules
    merge_rules = {}

    # Iteratively call merge until vocabularly size has been reached or no more merges are possible.
    while len(vocab) < vocab_size:
        # Attempt to merge the most frequent pair of characters.
        token_counts, vocab, merged, merge_rule = merge(token_counts, vocab)

        # Break the loop if no merge occurred.
        if merged is False:  # If no merge occurred, stop the loop.
            break

        # Accumulate merge rules
        merge_rules.update(merge_rule)

    return vocab, merge_rules


def save(filepath, vocab, merge_rules):
    os.makedirs(filepath, exist_ok=True)

    with open("{}/vocab.txt".format(filepath), 'w') as f:
        for token in vocab:
            f.write(token + "\n")

    with open("{}/merge_rules.txt".format(filepath), 'w') as file:
        for key, value in merge_rules.items():
            file.write(" ".join(key) + " " + value + "\n")


def load(filepath):
    vocab = []
    with open("{}/vocab.txt".format(filepath), 'r') as file:
        for line in file:
            vocab.append(line.strip())
    loaded_merge_rules = {}
    with open("{}/merge_rules.txt".format(filepath), 'r') as file:
        for line in file:
            parts = line.strip().split(" ")
            # All parts but the last form the key
            key_tuple = tuple(parts[:-1])
            value = parts[-1]  # The last part is the value
            loaded_merge_rules[key_tuple] = value

    return vocab, loaded_merge_rules


def save(filepath, vocab, merge_rules):
    os.makedirs(filepath, exist_ok=True)

    with open("{}/vocab.txt".format(filepath), 'w') as f:
        for token in vocab:
            f.write(token + "\n")

    with open("{}/merge_rules.txt".format(filepath), 'w') as file:
        for key, value in merge_rules.items():
            file.write(" ".join(key) + " " + value + "\n")


def load(filepath):
    vocab = []
    with open("{}/vocab.txt".format(filepath), 'r') as file:
        for line in file:
            vocab.append(line.strip())
    loaded_merge_rules = {}
    with open("{}/merge_rules.txt".format(filepath), 'r') as file:
        for line in file:
            parts = line.strip().split(" ")
            # All parts but the last form the key
            key_tuple = tuple(parts[:-1])
            value = parts[-1]  # The last part is the value
            loaded_merge_rules[key_tuple] = value

    return vocab, loaded_merge_rules


class CustomBPETokenizer(tf.Module):
    """
    A tokenizer that uses Byte Pair Encoding (BPE).

    This class takes in reserved tokens and a path to BPE data to perform tokenization and detokenization
    using the BPE algorithm. Tokens and merge rules are loaded from the BPE data path and are used to convert
    text strings to sequences of token indices and vice versa.
    """

    def __init__(self, reserved_tokens, bpe_data_path):
        """
        Initialize the CustomBPETokenizer.

        Parameters:
        - reserved_tokens (list of str): A list of tokens that should be included in the vocabulary.
        - bpe_data_path (str): The path to the BPE data (vocab and merge rules).

        The vocab and merge rules are loaded from files and two mappings are created:
        _token_nums: a mapping from tokens to their corresponding indices.
        _num_tokens: a mapping from indices to their corresponding tokens.
        """
        self._reserved_tokens = reserved_tokens

        vocab, self._merge_rules = load(bpe_data_path)
        self._token_nums = {k: i for i, k in enumerate(vocab)}
        self._num_tokens = {v: k for k, v in self._token_nums.items()}

    def tokenize(self, strings):
        """
        Tokenize a list of strings into token ids using BPE.

        Parameters:
        - tf.RaggedTensor of dtype tf.int32: Tokenized output represented as lists of token ids.

        Returns:
        - list of list of int: The tokenized strings represented as lists of token ids.
        """
        if tf.is_tensor(strings):
            strings = strings.numpy()
            strings = [str(s, 'utf-8') for s in strings]

        all_res = []

        start_token_id = self._token_nums.get('[START]', None)
        end_token_id = self._token_nums.get('[END]', None)
        if start_token_id is None or end_token_id is None:
            raise ValueError(
                "The tokens [START] and [END] must be defined in the _token_nums dictionary.")

        for string in strings:
            words = string.split()
            pre_tokenized_text = []
            for i, word in enumerate(words):
                if i != 0:  # Do not prepend "Ġ" to the first word
                    word = "Ġ" + word

                if word.endswith('.'):
                    pre_tokenized_text.append(word[:-1])
                    pre_tokenized_text.append(word[-1])
                else:
                    pre_tokenized_text.append(word)

            splits = [[l for l in word] for word in pre_tokenized_text]

            for pair, merge in self._merge_rules.items():
                for idx, split in enumerate(splits):
                    i = 0
                    while i < len(split) - 1:
                        if split[i] == pair[0] and split[i + 1] == pair[1]:
                            split = split[:i] + [merge] + split[i + 2:]
                        else:
                            i += 1
                    splits[idx] = split

            res = [self._token_nums[token] if token in self._token_nums else self._token_nums['[UNK]']
                   for word in splits for token in word]

            res = [start_token_id] + res + [end_token_id]

            all_res.append(res)

        return tf.ragged.constant(all_res, dtype=tf.int32)

    def detokenize(self, tokenized):
        """
        Convert a sequence of token ids into a string.

        Parameters:
        - tokenized (tf.RaggedTensor): The token ids to be detokenized.

        Returns:
        - tf.Tensor: The detokenized text strings.

        Detokenization is performed by converting token ids back into their string representation,
        concatenating them, and replacing the "Ġ" symbol with a space.
        """
        # Ensure that the input is a RaggedTensor
        if not isinstance(tokenized, tf.RaggedTensor):
            raise ValueError("Input must be a tf.RaggedTensor.")

        # Convert the RaggedTensor to a numpy array to work with regular python lists
        tokenized = tokenized.numpy()

        sentences = []
        for token_sequence in tokenized:
            tokens = [self._num_tokens[num]
                      if num in self._num_tokens else '[UNK]' for num in token_sequence]

            # Remove the [START] and [END] tokens
            tokens = [token for token in tokens if token not in [
                '[START]', '[END]']]

            sentence = ''.join(tokens)
            sentence = sentence.replace('Ġ', ' ')
            sentences.append(sentence.strip())

        # Convert the list of sentences to a tf.Tensor
        return tf.convert_to_tensor(sentences, dtype=tf.string)

    def lookup(self, token_ids):
        """
        Convert a sequence of token ids into tokens using the vocabulary.

        Parameters:
        - token_ids (list, tf.Tensor, or tf.RaggedTensor): A sequence of token ids to be converted into tokens.

        Returns:
        - tf.RaggedTensor of strings: The tokens corresponding to the input token ids.
        """
        # Ensure the input is a RaggedTensor
        if not isinstance(token_ids, tf.RaggedTensor):
            if isinstance(token_ids, tf.Tensor):
                token_ids = tf.RaggedTensor.from_tensor(token_ids)
            elif isinstance(token_ids, (list, np.ndarray)):
                token_ids = tf.RaggedTensor.from_tensor(np.array(token_ids))
            else:
                raise ValueError(
                    "Input token_ids must be a list, np.ndarray, tf.Tensor, or tf.RaggedTensor.")

        # Convert the RaggedTensor to a numpy array to work with regular python lists
        token_ids = token_ids.numpy()

        # Use the _num_tokens mapping to convert token ids to tokens
        decoded_tokens = []
        for seq in token_ids:
            tokens = [self._num_tokens.get(tid, '[UNK]') for tid in seq]
            decoded_tokens.append(tokens)

        # Convert the list of token sequences to a tf.RaggedTensor
        return tf.ragged.constant(decoded_tokens, dtype=tf.string)

    def get_vocab_size(self):
        return len(self._token_nums)
