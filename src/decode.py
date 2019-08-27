import tensorflow as tf
from pathlib import Path
from absl import app, flags
from model import transformer

flags.DEFINE_string("checkpoint_path", None, help="Model checkpoint path")


def decode(seed_text, model, strategy):
    """
    Decodes text from model, starting from seed_text using the given decoding stretegy
    :param seed_text:
    :param model:
    :param strategy:
    :return:
    """
    pass


def main(argv):
    # Load model
    transformer_decoder = transformer.TransformerOnlyDecoder()

    # Global step and epoch counters
    global_step = tf.Variable(0, name="global_step", trainable=False)
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

    # TODO: Request seed_text from stdin and run decode(...)


if __name__ == "__main__":
    app.run(main)
