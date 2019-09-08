import hparams as hp
import tensorflow as tf

hp.add("optimizer", "sgd", enum_values=["sgd", "adam", "rmsprop"], dtype=list, help="Optimizer")
hp.add("learning_rate", 0.01, help="Learning rate")
hp.add("momentum", 0.0, help="Momentum for optimizer")
hp.add("adam_beta_1", 0.9, help="Beta 1 for Adam optimizer")
hp.add("adam_beta_2", 0.999, help="Beta 2 for Adam optimizer")
hp.add("rmsprop_rho", 0.9, help="Rho value for RMSprop")


def get_optimizer():
    if hp.get("optimizer") == "sgd":
        return tf.keras.optimizers.SGD(hp.get("learning_rate"),
                                       hp.get("momentum"))
    elif hp.get("optimizer") == "adam":
        return tf.keras.optimizers.Adam(hp.get("learning_rate"),
                                        hp.get("adam_beta_1"),
                                        hp.get("adam_beta_2"))
    elif hp.get("optimizer") == "rmsprop":
        return tf.keras.optimizers.RMSprop(hp.get("learning_rate"),
                                           hp.get("rmsprop_rho"),
                                           hp.get("momentum"))
