import os
import time
import utils
from tqdm import tqdm
import tensorflow as tf


def tf_train(model_fn, train_dataset, train_gen, val_dataset, val_gen, num_epochs, batch_size, debug, output):
    best_val_score = 0
    optim = tf.train.AdamOptimizer(1e-4)
    # Visual feature input: Batch size, obj per image, features per obj
    v = tf.placeholder(tf.float32, shape=[None, 36, train_dataset.v_dim])
    # Bounding box input: batch size, boxes per image, coords per box
    b = tf.placeholder(tf.float32, shape=[None, 36, 6])
    # Question word input: Batch size, 14 words
    q = tf.placeholder(tf.int32, shape=[None, 14])
    # Labels input: Batch size, num answers
    a = tf.placeholder(tf.float32, shape=[None, train_dataset.num_ans_candidates])
    # Training flag
    training = tf.placeholder(tf.bool)
    # Runs the model
    pred = model_fn(v, b, q, training)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=a, logits=pred)
    loss_total = tf.reduce_sum(loss)
    train_op = optim.minimize(loss)
    score = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(a, 1)), tf.float32))
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)

        for epoch in range(num_epochs):
            total_loss = 0
            train_score = 0
            train_count = 0
            t = time.time()

            for (v_, b_, q_, a_) in tqdm(train_gen()(), total=train_dataset.size/batch_size):
                if debug and train_count > 20:
                    break
                train_count += 1
                temp_loss, temp_score, _ = sess.run([loss_total, score, train_op], feed_dict={v: v_, b: b_, q: q_, a: a_, training: True})
                total_loss += temp_loss
                train_score += temp_score
            total_loss /= train_count * batch_size
            train_score /= train_count * batch_size

            val_score = 0
            val_count = 0
            for (v_, b_, q_, a_) in tqdm(val_gen()(), total=val_dataset.size/batch_size):
                if debug and val_count > 10:
                    break
                val_count += 1
                val_score += sess.run(score, feed_dict={v: v_, b: b_, q: q_, a: a_, training: False})
            val_score /= val_count * batch_size

            print('Epoch {} - Time: {} seconds'.format(epoch+1, time.time()-t))
            print('Train loss: {}, score {}%'.format(total_loss, round(train_score * 100, 2)))
            print('Val score: {}%'.format(round(val_score * 100, 2)))

            if val_score > best_val_score:
                save_path = saver.save(sess, output)
                print("Model saved at {}".format(save_path))
