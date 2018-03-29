import numpy as np
import tensorflow as tf


def tf_embedding_variable(n_tokens, emb_dim, np_file):
    weight_init = np.load(np_file)
    assert weight_init.shape == (n_tokens, emb_dim)
    return tf.Variable(tf.concat([tf.convert_to_tensor(weight_init), tf.zeros([1, emb_dim])], 0))

def tf_embedding(embeddings, x):
    return tf.nn.embedding_lookup(embeddings, x)

def tf_question_rnn(num_hid, x):
    rnn_cell = tf.nn.rnn_cell.GRUCell(num_hid)
    outputs, state = tf.nn.static_rnn(rnn_cell, tf.unstack(x, axis=1), dtype=tf.float32)
    return state
