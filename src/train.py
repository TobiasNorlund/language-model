import tensorflow as tf
import numpy as np
import time
from model import transformer
from preprocess import get_vocab
from pathlib import Path
from utils import HParamSet
from absl import app, flags

# Training data params
hparams = HParamSet()
hparams.add("shuffle_buffer", 100, help="Shuffle buffer")
hparams.add("prefetch_buffer", 1, help="Prefetch buffer")

# Training params
hparams.add("batch_size", 1, help="Batch size")
hparams.add("learning_rate", 0.01, help="Learning rate")
hparams.add("checkpoint_every", 1000, help="Checkpoint every X step")

# Flags
flags.DEFINE_string("train_data", None, help="Training data tfrecord file")
flags.DEFINE_string("vocab", None, help="Vocab file")
flags.DEFINE_string("checkpoint_path", None, help="Checkpoint path")
flags.mark_flags_as_required(["train_data", "vocab", "checkpoint_path"])


def get_dataset(dataset_path: Path, batch_size: int, shuffle_buffer: int, prefetch_buffer: int):
    feature_description = {
        'text': tf.io.VarLenFeature(tf.int64)
    }

    def _parse_function(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)
        return tf.sparse.to_dense(example["text"])

    ds = tf.data.TFRecordDataset(str(dataset_path))
    ds = ds.map(_parse_function)
    ds = ds.padded_batch(batch_size, padded_shapes=(-1,))
    ds = ds.shuffle(buffer_size=shuffle_buffer)
    ds = ds.prefetch(buffer_size=prefetch_buffer)

    return ds


def create_masks(tar):
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = transformer.create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = transformer.create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return combined_mask


def calculate_loss(loss_obj, real, pred):
    # Masks padded tokens from loss_object
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_obj(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def main(argv):
    train_ds = get_dataset(Path(flags.FLAGS.train_data),
                           hparams.batch_size,
                           hparams.shuffle_buffer,
                           hparams.prefetch_buffer)
    vocab_size = get_vocab(Path(flags.FLAGS.vocab)).vocab_size + 2  # TODO: Add abstraction for the two special tokens?

    # Model
    transformer_decoder = transformer.TransformerOnlyDecoder(vocab_size)

    # Loss
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    # Metrics
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    # Optimizer
    optimizer = tf.optimizers.SGD(hparams.learning_rate)

    # Global step and epoch counters
    global_step = tf.Variable(0, name="global_step", trainable=False)
    epoch = tf.Variable(0, name="epoch", trainable=False)

    # Checkpointing
    checkpoint_path = Path(flags.FLAGS.checkpoint_path)
    ckpt = tf.train.Checkpoint(transformer_decoder=transformer_decoder, optimizer=optimizer,
                               global_step=global_step, epoch=epoch)
    ckpt_manager = tf.train.CheckpointManager(ckpt, str(checkpoint_path), max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored')

    # Tensorboard events
    train_log_dir = str(checkpoint_path / "events")
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    def train_step(tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        mask = create_masks(tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer_decoder(tar_inp, True, mask)
            loss = calculate_loss(loss_object, tar_real, predictions)

        gradients = tape.gradient(loss, transformer_decoder.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer_decoder.trainable_variables))

        train_accuracy(tar_real, predictions)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=global_step.numpy())
            tf.summary.scalar("gradient_norm", tf.linalg.global_norm(gradients), step=global_step.numpy())

        return loss

    try:
        while True:
            epoch_start = time.time()
            steps_start = time.time()

            # Reset metrics
            train_accuracy.reset_states()

            for batch in train_ds:
                global_step.assign_add(1)
                loss = train_step(batch)

                if global_step.numpy() == 1:
                    print("Number of trainable parameters: {}".format(
                        np.sum([np.prod(v.get_shape().as_list()) for v in transformer_decoder.trainable_variables])))

                # Print intermediate metrics
                if global_step.numpy() % 10 == 0:
                    print('Step: {} Loss: {:.4f} Accuracy: {:.4f} ({:.3f}s)'.format(
                        global_step.numpy(), loss, train_accuracy.result(), time.time() - steps_start))
                    steps_start = time.time()

                # Checkpoint every X step
                if global_step.numpy() % hparams.checkpoint_every == 0:
                    ckpt_save_path = ckpt_manager.save(checkpoint_number=global_step.numpy())
                    print("Saving checkpoint at '{}'".format(ckpt_save_path))

            # Update train accuracy metric
            with train_summary_writer.as_default():
                tf.summary.scalar('accuracy', train_accuracy.result(), step=global_step.numpy())

            print("Epoch {} finished in {} secs".format(epoch.numpy(), time.time() - epoch_start))
            epoch.assign_add(1)

            ckpt_save_path = ckpt_manager.save(checkpoint_number=global_step.numpy())
            print("Saving checkpoint at '{}'".format(ckpt_save_path))

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
