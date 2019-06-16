import tensorflow as tf
import logging
import os
import tensorflow_datasets as tfds
from pathlib import Path


VOCAB_FILE_PREFIX = "vocab"


def get_vocab(input_file: Path, output_dir: Path, target_vocab_size):
    if len(list(output_dir.glob("vocab*"))) > 0:
        logging.info("Loading existing vocabulary")
        vocab = tfds.features.text.SubwordTextEncoder.load_from_file(str(output_dir / VOCAB_FILE_PREFIX))
    else:
        logging.info("Started building vocabulary")
        with input_file.open("r") as input:
            vocab = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                corpus_generator=input,
                target_vocab_size=target_vocab_size
            )
        logging.info("Saving vocabulary")
        vocab.save_to_file(str(output_dir / VOCAB_FILE_PREFIX))
    return vocab


def preprocess(input_file: Path, output_dir: Path, output_name, target_vocab_size):
    if not output_dir.exists():
        os.mkdir(str(output_dir))

    vocab = get_vocab(input_file, output_dir, target_vocab_size)

    def encode_and_serialize(text):
        encoded = [vocab.vocab_size] + vocab.encode(text.numpy()) + [vocab.vocab_size + 1]
        proto = tf.train.Example(
            features=tf.train.Features(feature={
                "text": tf.train.Feature(int64_list=tf.train.Int64List(value=encoded))
            }))
        return proto.SerializeToString()

    def tf_encode_and_serialize(text):
        tf_string = tf.py_function(encode_and_serialize, [text], tf.string)
        return tf.reshape(tf_string, ())

    output_file_name = str(output_dir / output_name) + ".tfrecord"
    logging.info("Generating '{}'".format(output_file_name))
    ds = tf.data.TextLineDataset(str(input_file))
    ds = ds.map(tf_encode_and_serialize)
    writer = tf.data.experimental.TFRecordWriter(output_file_name)
    writer.write(ds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Using raw textual input, encodes and creates a TFRecord")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--target-vocab-size", type=int, default=2**15)
    parser.add_argument("--verbose", action="store_true")

    params = parser.parse_args()

    if params.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    preprocess(Path(params.input), Path(params.output_dir), params.output_name, params.target_vocab_size)