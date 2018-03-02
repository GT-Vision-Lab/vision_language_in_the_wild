from __future__ import print_function
import os
import json
import cPickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset


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

            

class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, dataroot='data'):
        super(VQAFeatureDataset, self).__init__()
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
            open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name)))
        print('loading features from h5 file')
        self.h5_path = os.path.join(dataroot, '%s36.hdf5' % name)
        #with h5py.File(self.h5_path, 'r') as hf:
        #    self.features = np.array(hf.get('image_features'))
        #    self.spatials = np.array(hf.get('spatial_features'))

        #self.features = torch.from_numpy(self.features)
        #self.spatials = torch.from_numpy(self.spatials)


        self.entries = _load_dataset(dataroot, name, self.img_id2idx)

        with h5py.File(self.h5_path, 'r') as hf:
            self.v_dim = np.array(hf['image_features'][0]).shape[1]
            self.s_dim = np.array(hf['spatial_features'][0]).shape[1]
            print("v_dim: " + str(self.v_dim))
            print("s_dim: " + str(self.s_dim))
        
        self.entries = _load_dataset(dataroot, name, self.img_id2idx)

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

    def tensorize(self, entry):
        question = torch.from_numpy(np.array(entry['q_token']))
        entry['q_token'] = question

        answer = entry['answer']
        labels = np.array(answer['labels'])
        scores = np.array(answer['scores'], dtype=np.float32)
        if len(labels):
            labels = torch.from_numpy(labels)
            scores = torch.from_numpy(scores)
            entry['answer']['labels'] = labels
            entry['answer']['scores'] = scores
        else:
            entry['answer']['labels'] = None
            entry['answer']['scores'] = None
        return entry

    def __getitem__(self, index):
        entry = next(self.entries)
        if not entry:
            self.entries = _load_dataset(self.dataroot, self.name, self.img_id2idx)
            entry = next(self.entries)
        entry = self.tensorize(self.tokenize(entry))

        with h5py.File(self.h5_path, 'r') as hf:
            features = torch.from_numpy(np.array(hf['image_features'][entry['image']]))
            spatials = torch.from_numpy(np.array(hf['spatial_features'][entry['image']]))
        #features = self.features[entry['image']]
        #spatials = self.spatials[entry['image']]

        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        return features, spatials, question, target

    def __len__(self):
        return len(self.img_id2idx)
