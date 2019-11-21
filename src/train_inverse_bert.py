import tensorflow as tf
import numpy as np
from preprocess import get_vocab, Vocabulary
from model import transformer
from pathlib import Path
from optimizer import get_optimizer
from absl import flags, app
import hparams as hp

# Training hparams
hp.add("shuffle_buffer", 1, help="Shuffle buffer")
hp.add("batch_size", 100, help="batch_size")
hp.add("min_num_spans", 1, help="Minimum number of spans per paragraph")
hp.add("max_num_spans", 4, help="Maximum number of spans per paragraph")
hp.add("min_span_len", 1, help="Minimum length of spans")
hp.add("max_span_len", 5, help="Minimum length of spans")

SEPARATOR_TOKEN = "~"


def get_dataset(dataset_path: Path, vocab: Vocabulary, batch_size: int, shuffle_buffer: int, skip: int = 0,
                min_num_spans=1, max_num_spans=4, min_span_len=1, max_span_len=5):

    def split_article(article_string_tensor):
        paragraphs = article_string_tensor.numpy().decode().split("<p>")
        return tf.constant([p.strip().encode() for p in paragraphs], dtype=tf.string)

    def encode_with_spanned_prefix(paragraph_string_tensor):
        """
        Takes a string tensor, splits it up by paragraphs and randomizes non overlapping spans of the text to use for
        conditioning a generative language model.

        Example original text:
        "Den infekterade striden om hur polisen i Norrbotten skött sitt jobb när det gäller sexhandeln har nu nått
        justitiekanslern (JK). <p> En hög polischef har JK-anmälts för att han försökt tysta en anställd."

        Example output (but encoded):
        ["<SEP> infekterade striden <SEP> Norrbotten <SEP> sexhandeln <START> Den infekterade striden om hur polisen i
         Norrbotten skött sitt jobb när det gäller sexhandeln har nu nått justitiekanslern (JK). <END>",
         "<SEP> polischef <SEP> JK-anmälts <START> En hög polischef har JK-anmälts för att han försökt tysta en
         anställd. <END>"
        ]
        """
        encoded_paragraph = vocab.encode(paragraph_string_tensor.numpy(),
                                         include_start_token=False, include_end_token=False)

        while True:
            num_spans = np.random.randint(min_num_spans, max_num_spans)
            span_lengths = np.random.randint(min_span_len, max_span_len, num_spans)
            span_start = np.sort(np.random.choice(len(encoded_paragraph) - min_span_len - 1, num_spans))
            # Ensure non-overlapping and not overflow
            if all(span_start[1:] > (span_start + span_lengths)[:-1]) and \
                    all((span_start + span_lengths) < len(encoded_paragraph)):
                break

        encoded = []
        separator_encoding = vocab.encode(SEPARATOR_TOKEN)
        for i in range(num_spans):
            encoded += separator_encoding + encoded_paragraph[span_start[i]: span_start[i] + span_lengths[i]]
        encoded += [vocab.start_idx] + encoded_paragraph + [vocab.end_idx]

        return tf.constant(encoded, dtype=tf.int64, shape=[len(encoded)])

    ds = tf.data.TextLineDataset(str(dataset_path))
    ds = ds.flat_map(lambda article_text: tf.data.Dataset.from_tensor_slices(
        tf.py_function(split_article, inp=[article_text], Tout=tf.string)))
    ds = ds.filter(lambda paragraph_text: tf.strings.length(paragraph_text, unit="UTF8_CHAR") > 50)
    ds = ds.map(lambda paragraph_text: tf.py_function(encode_with_spanned_prefix, inp=[paragraph_text], Tout=tf.int64))

    ds = ds.repeat()
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.skip(skip)
    ds = ds.padded_batch(batch_size, padded_shapes=[-1])
    ds = ds.prefetch(2)

    return ds


def main(argv):

    vocab = get_vocab(Path(flags.FLAGS.vocab))
    train_ds = get_dataset(Path(flags.FLAGS.data), vocab, hp.get("batch_size"), hp.get("shuffle_buffer"),
                           min_num_spans=hp.get("min_num_spans"),
                           max_num_spans=hp.get("max_num_spans"),
                           min_span_len=hp.get("min_span_len"),
                           max_span_len=hp.get("max_span_len"))

    # Model
    transformer_decoder = transformer.TransformerOnlyDecoder(vocab.vocab_size)

    # Optimizer
    optimizer, learning_rate = get_optimizer()

    # Counters
    global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64)

    # Checkpointing
    checkpoint_path = Path(flags.FLAGS.checkpoint_path)
    ckpt = tf.train.Checkpoint(transformer_decoder=transformer_decoder, optimizer=optimizer,
                               global_step=global_step)
    ckpt_manager = tf.train.CheckpointManager(ckpt, str(checkpoint_path), max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Restored checkpoint from: {}".format(ckpt_manager.latest_checkpoint))

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def calculate_loss(gt, pred):
        # gt: [batch, len]
        # pred: [batch, len, vocab_size]
        loss = loss_object(gt, pred) # [batch, len]

        # Mask: 1. Paddings & "input"
        padding_mask = tf.math.logical_not(tf.math.equal(gt, 0))

        lengths = tf.where(tf.equal(gt, vocab.start_idx))[:, 1] + 1
        mask = tf.sequence_mask(lengths, tf.shape(gt)[1])
        input_mask = tf.logical_not(mask)

        final_mask = tf.logical_and(padding_mask, input_mask)

        return tf.reduce_mean(tf.boolean_mask(loss, final_mask))

    # @tf.function
    def train_step(batch):
        batch_input = batch[:, :-1]
        batch_target = batch[:, 1:]

        mask = transformer.create_masks(batch_input, vocab.start_idx)

        with tf.GradientTape() as tape:
            logits, _ = transformer_decoder(batch_input, training=True, look_ahead_mask=mask)  #  TODO: Prefix lm mask
            loss = calculate_loss(batch_target, logits)

        gradients = tape.gradient(loss, transformer_decoder.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer_decoder.trainable_variables))

        return loss

    for batch in train_ds:
        global_step.assign_add(1)

        loss = train_step(batch)

        print(loss)


if __name__ == "__main__":
    flags.DEFINE_string("data", None, help="Training data file")
    flags.DEFINE_string("vocab", None, help="Vocab file")
    flags.DEFINE_string("checkpoint_path", None, help="Checkpoint path")
    flags.mark_flags_as_required(["data", "vocab", "checkpoint_path"])

    num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    print("Num GPUs Available: ", num_gpus)

    app.run(main)