import tensorflow as tf
import numpy as np
import hparams as hp

hp.add("num_layers", 1, help="Num transformer layers")
hp.add("dropout_rate", 0.0, help="Dropout rate")
hp.add("d_model", 128, help="d-model")
hp.add("num_heads", 4, help="Num self attention heads")
hp.add("dff", 512, help="dff")

# TEMP
INITIALIZATION_SCALE = 1.0 / tf.math.sqrt(20.0)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # apply sin to even indices in the array; 2i
    sines = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions so that we can add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_packed_attention_masks(batch_pos):
    """
    Creates an attention mask for a packed batch of sequences.
    It masks both padded tokens (=0) and "look ahead"
    Used in the 1st attention block in the decoder.
    :param batch_pos: [batch, packed_seq_len]
    :return: float32 of shape [batch, 1, packed_seq_len, packed_seq_len]. 0. = ok to attend, 1. = not ok
    """
    def _get_mask(pos):
        mask = np.ones((pos.shape[0], 1, pos.shape[1], pos.shape[1]), dtype=np.float32)
        for batch_idx in range(pos.shape[0]):
            boundaries = np.where(np.pad(pos[batch_idx, :], pad_width=(0, 1))[1:] < pos[batch_idx, :])[0] + 1
            seq_start_idx = 0
            for seq_end_idx in boundaries:
                mask[batch_idx, 0, seq_start_idx:seq_end_idx, seq_start_idx:seq_end_idx] = \
                    np.triu(np.ones(seq_end_idx - seq_start_idx), 1)
                seq_start_idx = seq_end_idx
        return mask

    return tf.numpy_function(_get_mask, inp=[batch_pos], Tout=tf.float32)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_v, depth_v)

    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    initializer1 = tf.initializers.VarianceScaling(1.0 * INITIALIZATION_SCALE, mode="fan_in", distribution="normal", seed=42)  # Xavier
    initializer2 = tf.initializers.VarianceScaling(2.0 * INITIALIZATION_SCALE, mode="fan_in", distribution="normal", seed=42)  # He
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu', kernel_initializer=initializer1, name="ffn1"),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model, kernel_initializer=initializer2, name="ffn2")  # (batch_size, seq_len, d_model)
    ])


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        initializer = tf.initializers.VarianceScaling(1.0 * INITIALIZATION_SCALE, mode="fan_in", distribution="normal", seed=42)
        self.wq = tf.keras.layers.Dense(d_model, kernel_initializer=initializer, name="q")
        self.wk = tf.keras.layers.Dense(d_model, kernel_initializer=initializer, name="k")
        self.wv = tf.keras.layers.Dense(d_model, kernel_initializer=initializer, name="v")

        self.dense = tf.keras.layers.Dense(d_model, kernel_initializer=initializer, name="merge")

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_v, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_v, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_v, d_model)

        return output, attention_weights


# class EncoderLayer(tf.keras.layers.Layer):
#     def __init__(self, d_model, num_heads, dff, rate):
#         super(EncoderLayer, self).__init__()
#
#         self.mha = MultiHeadAttention(d_model, num_heads)
#         self.ffn = point_wise_feed_forward_network(d_model, dff)
#
#         self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#
#         self.dropout1 = tf.keras.layers.Dropout(rate)
#         self.dropout2 = tf.keras.layers.Dropout(rate)
#
#     def call(self, x, training, mask):
#         attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
#         attn_output = self.dropout1(attn_output, training=training)
#         out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
#
#         ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
#         ffn_output = self.dropout2(ffn_output, training=training)
#         out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
#
#         return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        # self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, center=False, scale=False)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, center=False, scale=False)
        # self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        # self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        # tf.summary.scalar("input_variance", tf.math.reduce_variance(x))

        x = self.layernorm1(x)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        # tf.summary.histogram("mha", attn1)

        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm2(attn1 + x)
        # tf.summary.histogram("mha_normed", out1)

        # if enc_output is not None:
        #    attn2, attn_weights_block2 = self.mha2(
        #        enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        #    attn2 = self.dropout2(attn2, training=training)
        #    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
        # else:
        attn_weights_block2 = None
        #    out2 = out1  # self.layernorm2(out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, target_seq_len, d_model)
        # tf.summary.histogram("ffn", ffn_output)

        ffn_output = self.dropout3(ffn_output, training=training)
        # out3 = self.layernorm3(ffn_output + out1)  # (batch_size, target_seq_len, d_model)
        # tf.summary.histogram("out", ffn_output)

        return ffn_output, attn_weights_block1, attn_weights_block2


# class Encoder(tf.keras.layers.Layer):
#     def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate):
#         super(Encoder, self).__init__()
#
#         self.d_model = d_model
#         self.num_layers = num_layers
#
#         self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
#         self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)
#
#         self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
#                            for _ in range(num_layers)]
#
#         self.dropout = tf.keras.layers.Dropout(rate)
#
#     def call(self, x, training, mask):
#         seq_len = tf.shape(x)[1]
#
#         # adding embedding and position encoding.
#         x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
#         x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
#         x += self.pos_encoding[:, :seq_len, :]
#
#         x = self.dropout(x, training=training)
#
#         for i in range(self.num_layers):
#             x = self.enc_layers[i](x, training, mask)
#
#         return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, rate, **kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers

        embedding_initializer = tf.initializers.VarianceScaling(1.0, mode="fan_out", distribution="normal", seed=42)
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model,
                                                   embeddings_initializer=embedding_initializer)
        self.pos_encoding = positional_encoding(1000, self.d_model)  # TODO: Max length

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate, name="layer_{}".format(i))
                           for i in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6, center=False, scale=False)

    def call(self, input, input_pos, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(input)[1]
        attention_weights = {}

        x = self.embedding(input)  # (batch_size, target_seq_len, d_model)
        # tf.summary.scalar("embedding_variance", tf.math.reduce_variance(x))
        # tf.summary.histogram("embeddings", x)

        # Scale embeddings so that variance = 1.0 (at start)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # tf.summary.scalar("scaled_embedding_variance", tf.math.reduce_variance(x))

        # Add positional encoding
        x += tf.gather(self.pos_encoding, input_pos)

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            #attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            #attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        x = self.layernorm(x)

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


# class Transformer(tf.keras.Model):
#
#     def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size,
#                  rate=hparams.dropout_rate):
#         super(Transformer, self).__init__()
#
#         self.encoder = Encoder(num_layers, d_model, num_heads, dff,
#                                input_vocab_size, rate)
#
#         self.decoder = Decoder(num_layers, d_model, num_heads, dff,
#                                target_vocab_size, rate)
#
#         self.final_layer = tf.keras.layers.Dense(target_vocab_size)
#
#     def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
#         enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
#
#         # dec_output.shape == (batch_size, tar_seq_len, d_model)
#         dec_output, attention_weights = self.decoder(
#             tar, enc_output, training, look_ahead_mask, dec_padding_mask)
#
#         final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
#
#         return final_output, attention_weights


class TransformerOnlyDecoder(tf.keras.Model):

    def __init__(self, target_vocab_size=None):
        super(TransformerOnlyDecoder, self).__init__()
        # Note: If target_vocab_size is None, a checkpoint needs to be restored to initialise embeddings
        self.decoder = Decoder(
            num_layers=hp.get("num_layers"),
            d_model=hp.get("d_model"),
            num_heads=hp.get("num_heads"),
            dff=hp.get("dff"),
            target_vocab_size=target_vocab_size,
            rate=hp.get("dropout_rate"))

    def call(self, input, input_pos, training, attention_mask):
        """
        Predicts next token logits for each token in each packed sequence in input
        :param input: int32 tensor of shape [batch, packed_seq_len]
        :param input_pos: int32 tensor of shape [batch, packed_seq_len] with positions for each token
        :param training: bool
        :param attention_mask: attention span for each token in input. [batch, 1, packed_seq_len, packed_seq_len]
        :return: [batch, packed_seq_len, vocab_size] logits, attention weights
        """
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(input, input_pos, None, training, attention_mask, None)

        # Final projection to vocabulary => logits, logits should have variance ~1.0 at start
        final_output = tf.matmul(dec_output, self.decoder.embedding.embeddings, transpose_b=True)
        # final_output += self.logits_bias
        # tf.summary.histogram("logits", final_output)
        # tf.summary.histogram("logits_bias_weights", self.logits_bias.value())

        return final_output, attention_weights
