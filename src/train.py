import tensorflow as tf
from tensorflow.python.eager.profiler import start_profiler_server
start_profiler_server(6009)
num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", num_gpus)
#if num_gpus > 0:
#    for gpu_device in tf.config.experimental.list_physical_devices('GPU'):
#        tf.config.experimental.set_memory_growth(gpu_device, True)
import numpy as np
import time
import json
import hparams as hp
from model import transformer
from optimizer import get_optimizer
from preprocess import get_vocab
from pathlib import Path
from absl import app, flags

# Training hparams
hp.add("shuffle_buffer", 1, help="Shuffle buffer")
hp.add("batch_size", 100, help="batch_size")
hp.add("padding_length", 110, help="")
hp.add("dynamic_batching", False, help="Whether to use dynamic batching")
hp.add("max_tokens", 100, help="Max tokens")
hp.add("max_seq_len", 600, help="Max sequence len")


def get_dataset_dynamic(dataset_path: Path, shuffle_buffer: int, skip: int = 0):
    def parse_json(json_string_tensor):
        encoded = json.loads(json_string_tensor.numpy())["encoded"]
        return tf.constant(encoded, dtype=tf.int64, shape=[len(encoded)])

    def parse_json_fn(text):
        return tf.py_function(parse_json, inp=[text], Tout=tf.int64)

    #boundaries = np.arange(1, max_seq_len)
    #batch_sizes = [int(max_tokens / i) for i in np.arange(1, max_seq_len + 1)]

    # TODO: Manually optimized for schibsted-all on two GTX 1080
    boundaries = [30, 115, 120, 130, 140, 160, 180, 200, 240, 280, 350, 400, 500, 601]
    batch_sizes = [60, 40, 38, 36, 34, 32, 30, 28, 24, 20, 18, 16, 12, 10, 10]

    ds = tf.data.TextLineDataset(str(dataset_path))
    ds = ds.shuffle(buffer_size=shuffle_buffer, seed=42)
    ds = ds.repeat()
    ds = ds.skip(skip)
    ds = ds.map(parse_json_fn)
    ds = ds.apply(tf.data.experimental.bucket_by_sequence_length(lambda x: tf.shape(x),
                                                                 boundaries,
                                                                 batch_sizes,
                                                                 padded_shapes=[None],
                                                                 pad_to_bucket_boundary=True))
    ds = ds.prefetch(2)

    return ds


def get_dataset_static(dataset_path: Path, batch_size: int, shuffle_buffer: int, skip: int = 0):
    def parse_json(json_string_tensor):
        encoded = json.loads(json_string_tensor.numpy())["encoded"]
        return tf.constant(encoded, dtype=tf.int64, shape=[len(encoded)])

    def parse_json_fn(text):
        return tf.py_function(parse_json, inp=[text], Tout=tf.int64)

    ds = tf.data.TextLineDataset(str(dataset_path))
    ds = ds.shuffle(buffer_size=shuffle_buffer, seed=42)
    ds = ds.repeat()
    ds = ds.skip(skip)
    ds = ds.map(parse_json_fn)
    ds = ds.padded_batch(batch_size, padded_shapes=[None])
    ds = ds.prefetch(2)

    return ds


def train_loop(ds, transformer_decoder, global_step, num_examples_processed, ckpt_manager, optimizer, learning_rate,
               dist_strategy, checkpoint_every, summarize_every, continuous=True):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    # train_step_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int64)] - Not working with tf.distribute

    def calculate_loss(real, pred, batch_tokens):
        # Masks padded tokens from loss_object
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        return tf.reduce_sum(tf.boolean_mask(loss_, mask) / tf.cast(batch_tokens, tf.float32), keepdims=True)

    @tf.function(experimental_relax_shapes=True)
    def train_step(dist_batch):

        num_batch_tokens = dist_strategy.reduce(tf.distribute.ReduceOp.SUM,
                                       dist_strategy.experimental_run_v2(
                                           lambda x: tf.reduce_sum(
                                               tf.cast(tf.math.logical_not(tf.math.equal(x, 0)), tf.int64),
                                               keepdims=True),
                                           args=(dist_batch,)),
                                       axis=0)

        def _per_replica_step(batch):
            tar_inp = batch[:, :-1]
            tar_real = batch[:, 1:]

            mask = transformer.create_masks(tar_inp)
            should_summarize = tf.math.equal(tf.math.mod(global_step, summarize_every), 0)

            with tf.summary.record_if(should_summarize):
                with tf.GradientTape() as tape:
                    predictions, _ = transformer_decoder(tar_inp, True, mask)
                    loss = calculate_loss(tar_real, predictions, num_batch_tokens)

                vars = transformer_decoder.trainable_variables
                gradients = tape.gradient(loss, vars)
                optimizer.apply_gradients(zip(gradients, vars))
                num_examples_processed.assign_add(tf.cast(tf.shape(batch)[0], num_examples_processed.dtype))

                # Summarize individual vars and gradients
                #if should_summarize:
                #    for i in range(len(vars)):
                #        tf.summary.scalar("variable/" + vars[i].name, tf.linalg.norm(vars[i]))
                #        tf.summary.scalar("gradient/" + vars[i].name, tf.linalg.norm(gradients[i]))

            gradient_norm = tf.linalg.global_norm(gradients)[tf.newaxis]

            return loss , gradient_norm

        per_replica_losses, per_replica_grad_norms = \
            dist_strategy.experimental_run_v2(_per_replica_step, args=(dist_batch,))

        # Approximate global gradient norm by mean over replica grad norms
        mean_grad_norm = dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_grad_norms, axis=0) / \
                         dist_strategy.num_replicas_in_sync
        total_loss = dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=0)

        tf.summary.scalar("loss", total_loss)
        tf.summary.scalar("gradient_norm", mean_grad_norm)
        tf.summary.scalar("learning_rate",
                         learning_rate if type(learning_rate) is float else learning_rate(global_step))

        return total_loss

    steps_start = time.time()

    for batch in ds:
        global_step.assign_add(1)
        tf.summary.experimental.set_step(global_step)

        # Take a gradient step
        loss = train_step(batch)

        if global_step.numpy() == 1:
            print("Number of trainable parameters: {}".format(
                np.sum([np.prod(v.get_shape().as_list()) for v in transformer_decoder.trainable_variables])))

        # Print intermediate metrics
        if global_step.numpy() % 1 == 0:
           print('Step: {}\tLoss: {:.4f}\tNum examples: {}\tTime: {:.3f}s'.format(
               global_step.numpy(), loss, num_examples_processed.numpy(), time.time() - steps_start))
           steps_start = time.time()

        # Checkpoint every X step
        if global_step.numpy() % checkpoint_every == 0:
            ckpt_save_path = ckpt_manager.save(checkpoint_number=global_step)
            print("Saving checkpoint at '{}'".format(ckpt_save_path))

            if not continuous:
                break


def main(argv):
    vocab_size = get_vocab(Path(flags.FLAGS.vocab)).vocab_size

    dist_strategy = tf.distribute.MirroredStrategy()
    with dist_strategy.scope():

        # Model
        transformer_decoder = transformer.TransformerOnlyDecoder(vocab_size)

        # Optimizer
        optimizer, learning_rate = get_optimizer()

        # Counters
        global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64)
        num_examples_processed = tf.Variable(0, name="num_examples_processed", trainable=False, dtype=tf.int64,
                                             aggregation=tf.VariableAggregation.SUM)

        # Checkpointing
        checkpoint_path = Path(flags.FLAGS.checkpoint_path)
        ckpt = tf.train.Checkpoint(transformer_decoder=transformer_decoder, optimizer=optimizer,
                                   global_step=global_step, num_examples_processed=num_examples_processed)
        ckpt_manager = tf.train.CheckpointManager(ckpt, str(checkpoint_path), max_to_keep=5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored checkpoint from: {}".format(ckpt_manager.latest_checkpoint))

        # Tensorboard events
        train_log_dir = str(checkpoint_path / "events")
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        # Training dataset
        if hp.get("dynamic_batching") is True:
            ds = get_dataset_dynamic(Path(flags.FLAGS.data), hp.get("shuffle_buffer"),
                                     skip=num_examples_processed.numpy())
        else:
            ds = get_dataset_static(Path(flags.FLAGS.data), hp.get("batch_size"), hp.get("shuffle_buffer"),
                                    skip=num_examples_processed.numpy())

        ds = dist_strategy.experimental_distribute_dataset(ds)

        try:
            with train_summary_writer.as_default():
                train_loop(ds, transformer_decoder, global_step, num_examples_processed, ckpt_manager, optimizer,
                           learning_rate, dist_strategy, flags.FLAGS.checkpoint_every,
                           flags.FLAGS.summarize_every, flags.FLAGS.continuous)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    flags.DEFINE_string("data", None, help="Training data tfrecord file")
    flags.DEFINE_string("vocab", None, help="Vocab file")
    flags.DEFINE_string("checkpoint_path", None, help="Checkpoint path")
    flags.DEFINE_integer("checkpoint_every", 1000, help="Checkpoint every X step")
    flags.DEFINE_boolean("continuous", True, help="Whether to continue training after checkpointing")
    flags.DEFINE_integer("summarize_every", 50, help="Summarize model stats every X step")
    flags.mark_flags_as_required(["data", "vocab", "checkpoint_path"])

    app.run(main)
