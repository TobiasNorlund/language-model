import tensorflow as tf
from pathlib import Path
from abc import ABCMeta, abstractmethod
from absl import app, flags
from model import transformer
from preprocess import get_vocab


class DecodingStrategy(metaclass=ABCMeta):
    @abstractmethod
    def select(self, logits):
        """
        Takes a tensor 1-d tensor "logits" and returns an index (scalar int tensor)
        :param logits:
        :return:
        """
        pass


class RandomSamplingStrategy(DecodingStrategy):
    
    def __init__(self, temperature=1.0):
        self.temperature = tf.convert_to_tensor(temperature)

    def select(self, logits):
        return tf.random.categorical(logits / self.temperature, num_samples=1)[0, 0]


class TopKSamplingStrategy(DecodingStrategy):

    def __init__(self, k=5, temperature=1.0):
        self.k = tf.convert_to_tensor(k)
        self.temperature = tf.convert_to_tensor(temperature)

    def select(self, logits):
        values, indices = tf.math.top_k(logits / self.temperature, k=self.k)
        selection = tf.random.categorical(values , num_samples=1)
        return indices[0, selection[0, 0]]


def decode_encoded(seed_encoded, model, end_token_idx, strategy, max_len=100):
    """

    :param seed_encoded:
    :param model:
    :param strategy:
    :return:
    """
    seed_encoded = list(seed_encoded)

    for i in range(max_len):
        seed_tensor = tf.convert_to_tensor(seed_encoded)[tf.newaxis, :]
        seed_pos = tf.range(tf.shape(seed_tensor)[1])[tf.newaxis, :]
        mask = transformer.create_look_ahead_mask(tf.shape(seed_tensor)[1])

        # Get logits for next token
        logits, _ = model(seed_tensor, seed_pos, training=False, attention_mask=mask)
        logits = logits[:, -1, :]

        new_token = strategy.select(logits)

        seed_encoded.append(new_token.numpy())
        if new_token == end_token_idx:
            break

    return seed_encoded


def decode(seed_text, vocab, model, strategy, max_len):
    """
    Decodes text from model, starting from seed_text using the given decoding strategy
    :param seed_text:
    :param vocab:
    :param model:
    :param strategy:
    :return: The decoded
    """
    return vocab.decode(decode_encoded(vocab.encode(seed_text, include_start_token=True),
                                       model,
                                       vocab.end_idx,
                                       strategy,
                                       max_len))


def main(argv):

    if FLAGS.strategy == "random":
        strategy = RandomSamplingStrategy(temperature=FLAGS.temperature)
    elif FLAGS.strategy == "top-k":
        strategy = TopKSamplingStrategy(k=FLAGS.k, temperature=FLAGS.temperature)
    else:
        raise RuntimeError("Unsupported strategy '{}'".format(FLAGS.strategy))

    # Vocab
    vocab = get_vocab(str(Path(FLAGS.vocab)))

    # Load model
    transformer_decoder = transformer.TransformerOnlyDecoder()

    # Global step and epoch counters
    global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64)

    # Restore from checkpoint
    checkpoint_path = Path(FLAGS.checkpoint_path)
    ckpt = tf.train.Checkpoint(transformer_decoder=transformer_decoder, global_step=global_step)
    ckpt_manager = tf.train.CheckpointManager(ckpt, str(checkpoint_path), max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print("Restored checkpoint from: {}".format(ckpt_manager.latest_checkpoint))
    else:
        raise RuntimeError("Couldn't load from checkpoint")

    while True:
        seed_text = input("Seed text:\n")
        decoded = decode(seed_text, vocab, transformer_decoder, strategy, max_len=FLAGS.max_len)
        print(decoded)


if __name__ == "__main__":
    flags.DEFINE_string("vocab", None, help="Vocab path")
    flags.DEFINE_string("checkpoint_path", None, help="Model checkpoint path")
    flags.DEFINE_enum("strategy", "random", ["random", "top-k"], help="Decoding strategy")
    flags.DEFINE_integer("max_len", 100, help="Max length of generated text (in tokens)")
    flags.DEFINE_float("temperature", 1.0, help="Sampling temperature")
    flags.DEFINE_integer("k", 5, help="Top k to resample from")

    flags.mark_flags_as_required(["checkpoint_path", "vocab"])

    FLAGS = flags.FLAGS

    app.run(main)
