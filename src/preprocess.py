import tensorflow as tf
import logging
import os
import math
import tensorflow_datasets as tfds
from pathlib import Path


VOCAB_FILE_PREFIX = "vocab"


class VocabularyNotFoundException(Exception):
    pass


def get_vocab(output_dir: Path):
    if len(list(output_dir.glob("vocab*"))) > 0:
        return tfds.features.text.SubwordTextEncoder.load_from_file(str(output_dir / VOCAB_FILE_PREFIX))
    else:
        raise VocabularyNotFoundException()


def create_vocab(input_file: Path, target_vocab_size: int):
    with input_file.open("r") as input:
        return tfds.features.text.SubwordTextEncoder.build_from_corpus(
            corpus_generator=(line.strip() for line in input),
            target_vocab_size=target_vocab_size
        )


def get_or_create_vocab(input_file: Path, output_dir: Path, target_vocab_size):
    try:
        vocab = get_vocab(output_dir)
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
        return [vocab.vocab_size] + vocab.encode(text) + [vocab.vocab_size + 1]

    encoded_examples = []
    logging.info("Opening input file")
    with open(str(input_file), "r") as f:
        for i, text in enumerate(f):
            if i > 0 and i % 10000 == 0:
                logging.info("Processed {} lines from input file, {} encoded examples so far...".format(
                    i, len(encoded_examples)))
            encoded = encode(text)
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
    parser.add_argument("--target-vocab-size", type=int, default=2**15)
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