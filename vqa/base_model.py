from attention import new_attention
from language_model import tf_embedding, tf_embedding_variable, tf_question_rnn
from classifier import simple_classifier
from fc import fc_net
import tensorflow as tf


def tf_baseline(dataset, num_hid):
    def model_fn(v, b, q, training):
        w_emb = tf_embedding(tf_embedding_variable(dataset.dictionary.ntoken, 300, 'data/glove6b_init_300d.npy'), q)
        q_emb = tf_question_rnn(num_hid, w_emb)
        v_att = new_attention(v, q_emb, dataset.v_dim, num_hid, training=training)
        q_net = fc_net(q_emb, [num_hid, num_hid], training)
        v_net = fc_net(tf.reduce_sum(v * tf.expand_dims(v_att, -1), axis=1), [dataset.v_dim, num_hid], training)
        classifier = simple_classifier(q_net * v_net, dataset.num_ans_candidates, num_hid, 0.5, training)
        return classifier
    return model_fn
