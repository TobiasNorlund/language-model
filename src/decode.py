import tensorflow as tf
from pathlib import Path
from absl import app, flags
from model import transformer
from preprocess import get_vocab


flags.DEFINE_float("temperature", 1.0, help="Sampling temperature")
flags.DEFINE_integer("max_len", 100, help="Max length of generated text (in tokens)")

FLAGS = flags.FLAGS


def decode_random_sampling(seed_encoded, model, end_token_idx, temperature=1.0, max_len=100):
    seed_encoded = list(seed_encoded)
    temp = tf.convert_to_tensor(temperature)

    def _decode_step(seed):
        mask = transformer.create_look_ahead_mask(tf.shape(seed)[1])
        logits, _ = model(seed, training=False, look_ahead_mask=mask)
        return tf.random.categorical(logits[:,-1,:] / temp, num_samples=1)[0, 0].numpy()

    for i in range(max_len):
        new_token = _decode_step(tf.convert_to_tensor(seed_encoded)[tf.newaxis, :])
        seed_encoded.append(new_token)
        if new_token == end_token_idx:
            break

    return seed_encoded


def decode_encoded(seed_encoded, model, end_token_idx, strategy):
    """

    :param seed_encoded:
    :param model:
    :param strategy:
    :return:
    """
    if strategy == "random":
        return decode_random_sampling(seed_encoded, model, end_token_idx, temperature=FLAGS.temperature,
                                      max_len=FLAGS.max_len)
    else:
        raise RuntimeError("Unsupported strategy '{}'".format(strategy))


def decode(seed_text, vocab, model, strategy):
    """
    Decodes text from model, starting from seed_text using the given decoding stretegy
    :param seed_text:
    :param vocab:
    :param model:
    :param strategy:
    :return: The decoded
    """
    return vocab.decode(decode_encoded(vocab.encode(seed_text, include_start_token=True),
                                       model,
                                       vocab.end_idx,
                                       strategy))


def main(argv):
    # Vocab
    vocab = get_vocab(str(Path(flags.FLAGS.vocab)))

    # Load model
    transformer_decoder = transformer.TransformerOnlyDecoder()

    # Global step and epoch counters
    global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64)
    epoch = tf.Variable(0, name="epoch", trainable=False)

    # Restore from checkpoint
    checkpoint_path = Path(flags.FLAGS.checkpoint_path)
    ckpt = tf.train.Checkpoint(transformer_decoder=transformer_decoder, global_step=global_step, epoch=epoch)
    ckpt_manager = tf.train.CheckpointManager(ckpt, str(checkpoint_path), max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored')
    else:
        raise RuntimeError("Couldn't load from checkpoint")

    while True:
        seed_text = input("Seed text:\n")
        decoded = decode(seed_text, vocab, transformer_decoder, flags.FLAGS.strategy)
        print(decoded)


if __name__ == "__main__":
    flags.DEFINE_string("vocab", None, help="Vocab path")
    flags.DEFINE_string("checkpoint_path", None, help="Model checkpoint path")
    flags.DEFINE_enum("strategy", "random", ["random"], help="Decoding strategy")
    flags.mark_flags_as_required(["checkpoint_path"])

    app.run(main)
