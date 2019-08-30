import tensorflow as tf
import time
import json
from absl import app, flags
from model import transformer
from preprocess import get_vocab
from pathlib import Path
from decode import decode_encoded, RandomSamplingStrategy, TopKSamplingStrategy

flags.DEFINE_boolean("wait_for_checkpoint", True, help="Whether to wait for next checkpoint when done")
flags.DEFINE_string("data", None, help="Data tfrecord file")
flags.DEFINE_string("vocab", None, help="Vocab path")
flags.DEFINE_string("checkpoint_path", None, help="Model checkpoint path")
flags.DEFINE_integer("batch_size", 1, help="Batch size")
flags.mark_flags_as_required(["data", "vocab", "checkpoint_path"])


def get_dataset(dataset_path: Path, batch_size: int):
    feature_description = {
        'text': tf.io.VarLenFeature(tf.int64)
    }

    def _parse_function(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)
        return tf.sparse.to_dense(example["text"])

    ds = tf.data.TFRecordDataset(str(dataset_path))
    ds = ds.map(_parse_function)
    ds = ds.padded_batch(batch_size, padded_shapes=(-1,))

    return ds


def render_markdown(gt_example, random_sampled, top_5):
    return """
    Ground Truth: {}
    Random sampled: {}
    Top-5: {}
    """.format(gt_example, random_sampled, top_5)


def evaluate(vocab_path: Path, checkpoint_path: Path, dataset_path: Path, batch_size: int):
    # Vocab
    vocab = get_vocab(str(vocab_path))

    # Load model
    transformer_decoder = transformer.TransformerOnlyDecoder(vocab.vocab_size,
                                                             transformer.hparams.num_layers,
                                                             transformer.hparams.d_model,
                                                             transformer.hparams.num_heads,
                                                             transformer.hparams.dff,
                                                             transformer.hparams.dropout_rate)

    # Global step and epoch counters
    global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64)

    # Restore from checkpoint
    ckpt = tf.train.Checkpoint(transformer_decoder=transformer_decoder, global_step=global_step)
    ckpt_manager = tf.train.CheckpointManager(ckpt, str(checkpoint_path), max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Restored checkpoint from: {}".format(ckpt_manager.latest_checkpoint))
    else:
        raise RuntimeError("Couldn't load from checkpoint")

    # Dataset
    ds = get_dataset(str(dataset_path), batch_size=batch_size)

    # Metrics
    token_accuracy = tf.keras.metrics.SparseCategoricalAccuracy("token_accuracy")
    log_ppl = tf.keras.metrics.Mean("log_perplexity")

    for batch in ds:
        batch_inp = batch[:, :-1]
        batch_tar = batch[:, 1:]

        # Apply model
        mask = transformer.create_masks(batch_inp)
        logits, _ = transformer_decoder(batch_inp, False, mask)  # TODO: Visualise attentions

        # Update metrics
        token_accuracy(batch_tar, logits)
        log_ppl(tf.nn.sparse_softmax_cross_entropy_with_logits(batch_tar, logits) / tf.math.log(2.0))

    # Decode some examples
    gt_examples = []
    random_sampling_examples = []
    top_5_sampling_examples = []
    for example in get_dataset(str(dataset_path), batch_size=1).shuffle(1000, seed=42).take(5):
        # Use the first 4 tokens as seed
        gt_examples.append(vocab.decode(example[0].numpy()))
        random_sampling_examples.append(
            vocab.decode(decode_encoded(example[0][:4].numpy(), transformer_decoder, vocab.end_idx,
                                        RandomSamplingStrategy())))
        top_5_sampling_examples.append(
            vocab.decode(decode_encoded(example[0][:4].numpy(), transformer_decoder, vocab.end_idx,
                                        TopKSamplingStrategy(5))))

    # Tensorboard events
    eval_log_dir = str(checkpoint_path / (dataset_path.stem + "_eval"))
    eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)

    with eval_summary_writer.as_default():
        tf.summary.scalar("token_accuracy", token_accuracy.result(), global_step.numpy())
        tf.summary.scalar("log_perplexity", log_ppl.result(), global_step.numpy())

        # Write decoded examples..
        for i, (gt_example, rand_ex, top_5_ex) in enumerate(zip(gt_examples,
                                                                random_sampling_examples,
                                                                top_5_sampling_examples)):
            tf.summary.text("decoded_example_{}".format(i + 1),
                            tf.convert_to_tensor(render_markdown(gt_example, rand_ex, top_5_ex)),
                            global_step.numpy())

    return {"token_accuracy": float(token_accuracy.result().numpy()), "log_perplexity": float(log_ppl.result().numpy())}


def main(argv):
    checkpoint_path = Path(flags.FLAGS.checkpoint_path)
    try:
        while True:
            latest_checkpoint = tf.train.latest_checkpoint(str(checkpoint_path))
            start_time = time.time()
            print("Starting evaluation of checkpoint '{}'".format(latest_checkpoint))
            res = evaluate(Path(flags.FLAGS.vocab), checkpoint_path, Path(flags.FLAGS.data),
                     flags.FLAGS.batch_size)
            print("Evaluation of checkpoint '{}' finished in {}s".format(latest_checkpoint, time.time() - start_time))
            print(json.dumps(res))

            if flags.FLAGS.wait_for_checkpoint:
                while latest_checkpoint == tf.train.latest_checkpoint(str(checkpoint_path)):
                    time.sleep(10)
            else:
                break
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app.run(main)
