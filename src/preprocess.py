import logging
import json
import sys
import math
import tensorflow_datasets as tfds
from pathlib import Path


class Vocabulary(tfds.features.text.SubwordTextEncoder):
    """
    Extends the SubwordTextEncoder with two additional special tokens:
     - <START>
     - <END>
    """
    START = "<START>"
    END = "<END>"

    @property
    def start_idx(self):
        return self._token_to_ids(self.START)[0] + 1  # Add one for padding

    @property
    def end_idx(self):
        return self._token_to_ids(self.END)[0] + 1  # Add one for padding

    @classmethod
    def build_from_corpus(cls,
                          corpus_generator,
                          target_vocab_size,
                          max_subword_length=20,
                          max_corpus_chars=None,
                          reserved_tokens=None):
        reserved_tokens = [cls.START, cls.END] if reserved_tokens is None else reserved_tokens + [cls.START, cls.END]
        return super(Vocabulary, cls).build_from_corpus(corpus_generator, target_vocab_size, max_subword_length,
                                                        max_corpus_chars, reserved_tokens)

    def encode(self, s, include_start_token=False, include_end_token=False):
        encoded = super(Vocabulary, self).encode(s)
        if include_start_token:
            encoded = [self.start_idx] + encoded
        if include_end_token:
            encoded = encoded + [self.end_idx]
        return encoded

    # TODO: Optionally remove start/end tokens in decode()


class VocabularyNotFoundException(Exception):
    pass


def get_vocab(vocab_path: Path):
    try:
        return Vocabulary.load_from_file(str(vocab_path).replace(".subwords", ""))
    except Exception:
        raise VocabularyNotFoundException()


def create_vocab(input_file: Path, target_vocab_size: int):
    with input_file.open("r") as input:
        return Vocabulary.build_from_corpus(
            corpus_generator=(line.strip() for line in input),
            target_vocab_size=target_vocab_size
        )


def get_or_create_vocab(input_file: Path, vocab_path: Path, target_vocab_size):
    try:
        vocab = get_vocab(vocab_path)
        logging.info("Loaded existing vocabulary")
    except VocabularyNotFoundException:
        logging.info("Started building vocabulary")
        vocab = create_vocab(input_file, target_vocab_size)
        logging.info("Saving vocabulary of size {}".format(vocab.vocab_size))
        vocab.save_to_file(str(vocab_path).replace(".subwords", ""))
    return vocab


def preprocess(input_file: Path, output: Path, vocab_file: Path, target_vocab_size, min_length, max_length,
               max_examples):

    vocab = get_or_create_vocab(input_file, vocab_file, target_vocab_size)

    def encode(text):
        return vocab.encode(text, include_start_token=True, include_end_token=True)

    encoded_examples = []
    logging.info("Opening input file")
    with open(str(input_file), "r") as f:
        for i, text in enumerate(f):
            text = text.strip()
            if i > 0 and i % 10000 == 0:
                logging.info("Processed {} lines from input file, {} encoded examples so far...".format(
                    i, len(encoded_examples)))
            encoded = encode(text)
            if len(encoded) < min_length or len(encoded) > max_length:
                continue
            encoded_examples.append({"text": text, "encoded": encoded})
            if len(encoded_examples) >= max_examples:
                break
    logging.info("Finished with {} examples".format(len(encoded_examples)))

    logging.info("Writing output file '{}'".format(str(output)))
    with open(str(output), "w") as f:
        for encoded_example in encoded_examples:
            f.write(json.dumps(encoded_example, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Using raw textual input, encodes and writes json")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--target-vocab-size", type=int, default=2 ** 15)
    parser.add_argument("--min-length", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=math.inf)
    parser.add_argument("--max-examples", type=int, default=math.inf)
    parser.add_argument("--verbose", action="store_true")

    params = parser.parse_args()

    if params.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    preprocess(Path(params.input), Path(params.output), Path(params.vocab), params.target_vocab_size, params.min_length,
               params.max_length, params.max_examples)
