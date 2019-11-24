import tensorflow as tf
from tensorflow.python.eager.profiler import start_profiler_server
import numpy as np
import time
import json
import hparams as hp
from model import transformer
from optimizer import get_optimizer
from preprocess import get_vocab
from pack import pack_dataset
from pathlib import Path
from absl import app, flags

# Training hparams
hp.add("shuffle_buffer", 1, help="Shuffle buffer")
hp.add("batch_size", 10, help="batch_size")
hp.add("packed_seq_len", 600, help="Packed sequence length")


def get_dataset(dataset_path: Path, batch_size: int, shuffle_buffer: int, packed_seq_len: int, skip: int = 0):
    def parse_json(json_string_tensor):
        encoded = json.loads(json_string_tensor.numpy())["encoded"]
        return tf.constant(encoded, dtype=tf.int32, shape=[len(encoded)])

    def parse_json_fn(text):
        return {"text": tf.py_function(parse_json, inp=[text], Tout=tf.int32)}

    ds = tf.data.TextLineDataset(str(dataset_path))
    ds = ds.shuffle(buffer_size=shuffle_buffer, seed=42)
    ds = ds.repeat()
    ds = ds.skip(skip)
    ds = ds.map(parse_json_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = pack_dataset(ds, length=packed_seq_len)  # Stack examples to fill out full seq len
    ds = ds.batch(batch_size)
    ds = ds.prefetch(2)

    return ds


def train_loop(ds, transformer_decoder, global_step, num_examples_processed, ckpt_manager, optimizer, learning_rate,
               dist_strategy, checkpoint_every, summarize_every, continuous=True):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def calculate_loss(ground_truth, predictions, ground_truth_pos, num_batch_tokens):
        # Masks non predictable tokens (padding & <START> tokens) = zeros in ground_truth_pos tensor
        # Note: Assumes all sequences start with a <START> token
        mask = tf.math.logical_not(tf.math.equal(ground_truth_pos, 0))
        token_losses = loss_object(ground_truth, predictions)

        return tf.reduce_sum(tf.boolean_mask(token_losses, mask) / tf.cast(num_batch_tokens, tf.float32), keepdims=True)

    @tf.function
    def train_step(dist_batch_text, dist_batch_pos, dist_batch_seg):

        # Calculate total number of predictable tokens (padding & <START> tokens) = zeros in batch_pos tensor
        # Note: Assumes all sequences start with a <START> token
        num_batch_tokens = dist_strategy.reduce(tf.distribute.ReduceOp.SUM,
                                                dist_strategy.experimental_run_v2(
                                                    lambda x: tf.reduce_sum(
                                                        tf.cast(tf.math.logical_not(tf.math.equal(x, 0)), tf.int32),
                                                        keepdims=True),
                                                    args=(dist_batch_pos,)),
                                                axis=0)

        def _per_replica_step(batch_text, batch_pos, batch_seg):
            model_input, model_input_pos = batch_text[:, :-1], batch_pos[:, :-1]
            ground_truth, ground_truth_pos = batch_text[:, 1:], batch_pos[:, 1:]

            attention_mask = transformer.create_packed_attention_masks(model_input)
            should_summarize = tf.math.equal(tf.math.mod(global_step, summarize_every), 0)

            with tf.summary.record_if(should_summarize):
                with tf.GradientTape() as tape:
                    predictions, _ = transformer_decoder(model_input, model_input_pos, True, attention_mask)
                    loss = calculate_loss(ground_truth, predictions, ground_truth_pos, num_batch_tokens)

                vars = transformer_decoder.trainable_variables
                gradients = tape.gradient(loss, vars)
                optimizer.apply_gradients(zip(gradients, vars))
                num_examples_processed.assign_add(tf.reduce_sum(tf.reduce_max(batch_seg, axis=1)))

                # Summarize individual vars and gradients
                # if should_summarize:
                #    for i in range(len(vars)):
                #        tf.summary.scalar("variable/" + vars[i].name, tf.linalg.norm(vars[i]))
                #        tf.summary.scalar("gradient/" + vars[i].name, tf.linalg.norm(gradients[i]))

            gradient_norm = tf.linalg.global_norm(gradients)[tf.newaxis]

            return loss, gradient_norm

        per_replica_losses, per_replica_grad_norms = \
            dist_strategy.experimental_run_v2(_per_replica_step, args=(dist_batch_text, dist_batch_pos, dist_batch_seg))

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
        loss = train_step(batch["text"], batch["text_position"], batch["text_segmentation"])

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
        num_examples_processed = tf.Variable(0, name="num_examples_processed", trainable=False, dtype=tf.int32,
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
        ds = get_dataset(Path(flags.FLAGS.data), hp.get("batch_size"), hp.get("shuffle_buffer"),
                         hp.get("packed_seq_len"), skip=num_examples_processed.numpy())
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

    num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    print("Num GPUs Available: ", num_gpus)
    start_profiler_server(6009)

    app.run(main)
