import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import tensorflow as tf


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits

def simple_classifier(x, out_dim, hid_dim, dropout, training):
    x = tf.layers.dense(x, hid_dim)
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)
    x = tf.layers.dropout(x, dropout, training=training)
    x = tf.layers.dense(x, out_dim)
    return x

