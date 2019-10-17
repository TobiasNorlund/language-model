import hparams as hp
import tensorflow as tf
from model.learning_rate_schedule import CustomSchedule

hp.add("optimizer", "sgd", enum_values=["sgd", "adam", "rmsprop", "adagrad"], dtype=list, help="Optimizer")
hp.add("learning_rate", 0.01, help="Learning rate")
hp.add("learning_rate_schedule", False, help="Use learning rate schedule")
hp.add("learning_rate_schedule_constant", 1.0, help="Learning rate schedule constant")
hp.add("learning_rate_warmup_steps", 4000, help="Learning rate schedule warmup steps")
hp.add("momentum", 0.0, help="Momentum for optimizer")
hp.add("adam_beta_1", 0.9, help="Beta 1 for Adam optimizer")
hp.add("adam_beta_2", 0.999, help="Beta 2 for Adam optimizer")
hp.add("rmsprop_rho", 0.9, help="Rho value for RMSprop")


def get_optimizer():
    learning_rate = CustomSchedule(hp.get("d_model"), hp.get("learning_rate_schedule_constant"),
                                   hp.get("learning_rate_warmup_steps")) \
        if hp.get("learning_rate_schedule") else hp.get("learning_rate")
    if hp.get("optimizer") == "sgd":
        return tf.keras.optimizers.SGD(learning_rate,
                                       hp.get("momentum")), learning_rate
    elif hp.get("optimizer") == "adam":
        return tf.keras.optimizers.Adam(learning_rate,
                                        hp.get("adam_beta_1"),
                                        hp.get("adam_beta_2")), learning_rate
    elif hp.get("optimizer") == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate,
                                           hp.get("rmsprop_rho"),
                                           hp.get("momentum")), learning_rate
    elif hp.get("optimizer") == "adagrad":
        return tf.keras.optimizers.Adagrad(learning_rate), learning_rate