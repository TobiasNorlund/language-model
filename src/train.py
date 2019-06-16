import tensorflow as tf
from pathlib import Path


def get_dataset(dataset_path: Path, batch_size: int, shuffle_buffer: int, prefetch_buffer: int):
    feature_description = {
        'text': tf.io.VarLenFeature(tf.int64)
    }

    def _parse_function(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)
        return tf.sparse.to_dense(example["text"])

    ds = tf.data.TFRecordDataset(str(dataset_path))
    ds = ds.map(_parse_function)
    ds = ds.shuffle(buffer_size=shuffle_buffer)
    ds = ds.padded_batch(batch_size, padded_shapes=(-1,))
    ds = ds.repeat()
    ds = ds.prefetch(buffer_size=prefetch_buffer)

    return ds


def train(train_data: Path, batch_size: int, shuffle_buffer: int, prefetch_buffer: int):
    train_ds = get_dataset(train_data, batch_size, shuffle_buffer, prefetch_buffer)

    for step, batch in enumerate(train_ds):
        print("{}: {}".format(step, batch))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Train")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--shuffle-buffer", type=int, default=1000)
    parser.add_argument("--prefetch-buffer", type=int, default=1)
    params = parser.parse_args()

    train(Path(params.train_data), params.batch_size, params.shuffle_buffer, params.prefetch_buffer)