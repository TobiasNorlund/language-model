import tensorflow as tf
import logging
import os
import math
import tensorflow_datasets as tfds
from pathlib import Path

VOCAB_FILE_PREFIX = "vocab"


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


def get_or_create_vocab(input_file: Path, output_dir: Path, target_vocab_size):
    try:
        vocab = get_vocab(output_dir / VOCAB_FILE_PREFIX)
        logging.info("Loaded existing vocabulary")
    except VocabularyNotFoundException:
        logging.info("Started building vocabulary")
        vocab = create_vocab(input_file, target_vocab_size)
        logging.info("Saving vocabulary of size {}".format(vocab.vocab_size))
        vocab.save_to_file(str(output_dir / VOCAB_FILE_PREFIX))
    return vocab


def preprocess(input_file: Path, output_dir: Path, output_name, target_vocab_size, min_length, max_length,
               max_examples):
    if not output_dir.exists():
        os.mkdir(str(output_dir))

    vocab = get_or_create_vocab(input_file, output_dir, target_vocab_size)

    def encode(text):
        return vocab.encode(text, include_start_token=True, include_end_token=True)

    encoded_examples = []
    logging.info("Opening input file")
    with open(str(input_file), "r") as f:
        for i, text in enumerate(f):
            if i > 0 and i % 10000 == 0:
                logging.info("Processed {} lines from input file, {} encoded examples so far...".format(
                    i, len(encoded_examples)))
            encoded = encode(text.strip())
            if len(encoded) < min_length or len(encoded) > max_length:
                continue
            encoded_examples.append(encoded)
            if len(encoded_examples) >= max_examples:
                break
    logging.info("Finished with {} examples".format(len(encoded_examples)))
    encoded_examples.sort(key=lambda x: len(x))
    logging.info("Sorted examples")

    def encoded_example_gen():  # Only way I could figure out how to turn encoded_examples into a Dataset
        for example in encoded_examples:
            yield example

    def serialize(encoded_example):
        proto = tf.train.Example(
            features=tf.train.Features(feature={
                "text": tf.train.Feature(int64_list=tf.train.Int64List(value=encoded_example))
            }))
        return proto.SerializeToString()

    def tf_serialize(encoded_example):
        tf_string = tf.py_function(serialize, [encoded_example], tf.string)
        return tf.reshape(tf_string, ())

    output_file_name = str(output_dir / output_name) + ".tfrecord"
    logging.info("Generating '{}'".format(output_file_name))
    ds = tf.data.Dataset.from_generator(encoded_example_gen, output_types=(tf.int32))
    ds = ds.map(tf_serialize)
    writer = tf.data.experimental.TFRecordWriter(output_file_name)
    writer.write(ds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Using raw textual input, encodes and creates a TFRecord")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
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

    preprocess(Path(params.input), Path(params.output_dir), params.output_name, params.target_vocab_size,
               params.min_length, params.max_length, params.max_examples)
