import tensorflow as tf
import time
from model import transformer
from preprocess import get_vocab
from model.learning_rate_schedule import CustomSchedule
from pathlib import Path


HPARAMS = {
    "num_layers": 1,
    "d_model": 128,
    "num_heads": 8,
    "dff": 512,
    "dropout_rate": 0.1,
    "learning_rate_constant": 1.0,
    "checkpoint_every": 1000
}


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


def train(train_data: Path, vocab_dir: Path, batch_size: int, shuffle_buffer: int, prefetch_buffer: int,
          num_layers: int, d_model: int, num_heads: int, dff: int, dropout_rate: 0.1, learning_rate_constant: float,
          checkpoint_path: Path, checkpoint_every: int):
    # Training data
    train_ds = get_dataset(train_data, batch_size, shuffle_buffer, prefetch_buffer)
    vocab_size = get_vocab(vocab_dir).vocab_size + 2  # TODO: Add abstraction for the two special tokens?

    # Model
    transformer_decoder = transformer.TransformerOnlyDecoder(num_layers, d_model, num_heads, dff,
                                                             vocab_size, dropout_rate)

    # Loss
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(real, pred):
        # Masks padded tokens from loss_object
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    # Metrics
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    # Optimizer
    learning_rate_schedule = CustomSchedule(d_model, constant=learning_rate_constant)
    optimizer = tf.keras.optimizers.Adam(learning_rate_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # Global step and epoch counters
    global_step = tf.Variable(0, name="global_step")
    epoch = tf.Variable(0, name="epoch")

    # Checkpointing
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
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer_decoder.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer_decoder.trainable_variables))

        train_accuracy(tar_real, predictions)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=global_step.numpy())
            tf.summary.scalar('learning_rate', learning_rate_schedule(float(global_step.numpy())),
                              step=global_step.numpy())
            tf.summary.scalar("gradient_norm", tf.linalg.global_norm(gradients), step=global_step.numpy())
            tf.summary.scalar('accuracy', train_accuracy.result(), step=global_step.numpy())

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

                # Print intermediate metrics
                if global_step.numpy() % 10 == 0:
                    print('Step: {} Loss: {:.4f} Accuracy: {:.4f} ({:.3f}s)'.format(
                        global_step.numpy(), loss, train_accuracy.result(), time.time() - steps_start))
                    steps_start = time.time()

                # Checkpoint every X step
                if global_step.numpy() % checkpoint_every == 0:
                    ckpt_save_path = ckpt_manager.save(checkpoint_number=global_step.numpy())
                    print("Saving checkpoint at '{}'".format(ckpt_save_path))

            print("Epoch {} finished in {} secs".format(epoch.numpy(), time.time() - epoch_start))
            epoch.assign_add(1)

            ckpt_save_path = ckpt_manager.save(checkpoint_number=global_step.numpy())
            print("Saving checkpoint at '{}'".format(ckpt_save_path))


    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Train")

    # Data params
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--vocab-dir", type=Path, required=True)
    parser.add_argument("--shuffle-buffer", type=int, default=100)
    parser.add_argument("--prefetch-buffer", type=int, default=1)

    # Model params
    parser.add_argument("--num-layers", default=HPARAMS["num_layers"], type=int)
    parser.add_argument("--d-model", default=HPARAMS["d_model"], type=int)
    parser.add_argument("--num_heads", default=HPARAMS["num_heads"], type=int)
    parser.add_argument("--dff", default=HPARAMS["dff"], type=int)

    # Training params
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--dropout_rate", default=HPARAMS["dropout_rate"], type=int)
    parser.add_argument("--learning-rate-constant", default=HPARAMS["learning_rate_constant"], type=float)
    parser.add_argument("--checkpoint-every", default=HPARAMS["checkpoint_every"], type=int)
    # TODO: Add params for learning rate schedule?

    params = parser.parse_args()

    train(params.train_data, params.vocab_dir, params.batch_size, params.shuffle_buffer, params.prefetch_buffer,
          params.num_layers, params.d_model, params.num_heads, params.dff, params.dropout_rate,
          params.learning_rate_constant, params.checkpoint_path, params.checkpoint_every)
