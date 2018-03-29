import argparse
import numpy as np

from dataset import Dictionary, VQAFeatureDataset, VQA_Dataset, tensorflow_generator
import base_model
from train import tf_train
import utils
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    batch_size = args.batch_size
    train_dset = VQAFeatureDataset('train', dictionary)
    train_gen = lambda: tensorflow_generator('train', dictionary, batch_size=batch_size)
    val_dset = VQAFeatureDataset('val', dictionary)
    val_gen = lambda: tensorflow_generator('val', dictionary, batch_size=batch_size)
    model_fn = base_model.tf_baseline(train_dset, args.num_hid)
    tf_train(model_fn, train_dset, train_gen, val_dset, val_gen, args.epochs, batch_size, args.debug, args.output)
