from fc import FCNet, fc_net
import tensorflow as tf


def new_attention(v, q, v_dim, num_hid, dropout=0.2, training=True):
    v_proj = fc_net(tf.reshape(v, [-1, 36 * v_dim]), [tf.shape(v)[1], num_hid], training)
    q_proj = fc_net(q, [tf.shape(q)[1], num_hid], training)
    joint_repr = tf.layers.dropout(v_proj * q_proj, dropout, training=training)
    joint_repr = tf.layers.dense(joint_repr, 36)
    return tf.nn.softmax(joint_repr, 1)
