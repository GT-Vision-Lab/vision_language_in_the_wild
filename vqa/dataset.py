from __future__ import print_function
import os
import json
import sys
if sys.version_info[0] < 3:
    import cPickle
else:
    import _pickle as cPickle
import numpy as np
import utils
import h5py
import tensorflow as tf


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer}
    return entry


def _load_dataset(dataroot, name, img_id2val):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(
        dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        if img_id in img_id2val:
            yield _create_entry(img_id2val[img_id], question, answer)


class VQAFeatureDataset:
    def __init__(self, name, dictionary, dataroot='data'):
        assert name in ['train', 'val']
        self.name = name
        self.dataroot = dataroot

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary

        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '{}36_imgid2idx.pkl'.format(name)), 'rb'))
        print('loading features from h5 file')
        self.h5_path = os.path.join(dataroot, '{}36.hdf5'.format(name))
        #with h5py.File(self.h5_path, 'r') as hf:
        #    self.features = np.array(hf.get('image_features'))
        #    self.spatials = np.array(hf.get('spatial_features'))

        #self.features = torch.from_numpy(self.features)
        #self.spatials = torch.from_numpy(self.spatials)


        self.entries = _load_dataset(dataroot, name, self.img_id2idx)
        self.size = len(cPickle.load(open(os.path.join(dataroot, 'cache', '%s_target.pkl' % name), 'rb')))

        with h5py.File(self.h5_path, 'r') as hf:
            self.v_dim = np.array(hf['image_features'][0]).shape[1]
            self.s_dim = np.array(hf['spatial_features'][0]).shape[1]
            print("v_dim: " + str(self.v_dim))
            print("s_dim: " + str(self.s_dim))


    def tokenize(self, entry, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        tokens = self.dictionary.tokenize(entry['question'], False)
        tokens = tokens[:max_length]
        if len(tokens) < max_length:
            # Note here we pad in front of the sentence
            padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
            tokens = padding + tokens
        utils.assert_eq(len(tokens), max_length)
        entry['q_token'] = tokens
        return entry


def tensorflow_generator(name, dictionary, batch_size=1, dataroot='data'):
    vqa_dataset = VQAFeatureDataset(name, dictionary, dataroot)
    def tensorflowize(entry):
        if not entry:
            raise StopIteration
        entry = vqa_dataset.tokenize(entry)
        question = np.array(entry['q_token'])
        answer = entry['answer']
        labels = np.array(answer['labels'])
        scores = np.array(answer['scores'], dtype=np.float32)
        if len(labels):
            labels = np.expand_dims(labels, axis=0)
            target = np.zeros(vqa_dataset.num_ans_candidates)
            target[labels] = scores
        else:
            return None
        with h5py.File(vqa_dataset.h5_path, 'r') as hf:
            features = np.array(hf['image_features'][entry['image']])
            spatials = np.array(hf['spatial_features'][entry['image']])
            # Output: v, b, q, a
            ans = (features, spatials, question, target)
            return ans
    def gen():
        batch = []
        while True:
            result = tensorflowize(next(vqa_dataset.entries))
            if result:
                batch.append(result)
            if batch_size == len(batch):
                ans = tuple(np.stack([x[i] for x in batch]) for i in (0,1,2,3))
                batch = []
                yield ans
    return gen

class VQA_Dataset(tf.data.Dataset):
    def __init__(self, name, dictionary, dataroot='data'):
        self = tf.data.Dataset.from_generator(tensorflow_generator(name, dictionary, dataroot), (tf.int32, tf.float32, tf.float32, tf.float32, tf.float32), (tf.TensorShape([14]), tf.TensorShape([1]), tf.TensorShape([1]), tf.TensorShape([36, 2048]), tf.TensorShape([36, 6])))
        self.name = name
        self.dataroot = dataroot
