import argparse
import os
import time
import utils
from tqdm import tqdm
from base_model import tf_baseline
from dataset import Dictionary, VQAFeatureDataset, tensorflow_generator
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()
    return args

def tf_eval(model_fn, save_path, eval_dataset, eval_gen, debug, batch_size):
    # Visual feature input: Batch size, obj per image, features per obj
    v = tf.placeholder(tf.float32, shape=[None, 36, eval_dataset.v_dim])
    # Bounding box input: batch size, boxes per image, coords per box
    b = tf.placeholder(tf.float32, shape=[None, 36, 6])
    # Question word input: Batch size, 14 words
    q = tf.placeholder(tf.int32, shape=[None, 14])
    # Labels input: Batch size, num answers
    a = tf.placeholder(tf.float32, shape=[None, eval_dataset.num_ans_candidates])
    # Training flag
    training = tf.placeholder(tf.bool)
    # Runs the model
    pred = model_fn(v, b, q, training)
    score = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(a, 1)), tf.float32))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, save_path)
        print("Model restored.")

        val_score = 0
        val_count = 0
        for (v_, b_, q_, a_) in tqdm(val_gen()(), total=eval_dataset.size/batch_size):
            if debug and val_count > 10:
                break
            val_count += 1
            val_score += sess.run(score, feed_dict={v: v_, b: b_, q: q_, a: a_, training: False})
        val_score /= val_count * batch_size

        print('Val score: {}%'.format(round(val_score * 100, 2)))

if __name__ == '__main__':
    args = parse_args()
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    batch_size = args.batch_size
    val_dset = VQAFeatureDataset('val', dictionary)
    val_gen = lambda: tensorflow_generator('val', dictionary, batch_size=batch_size)
    model_fn = tf_baseline(val_dset, args.num_hid)
    tf_eval(model_fn, args.path, val_dset, val_gen, args.debug, batch_size)
