#! /usr/bin/python3
import json
import pickle
import random
import os
from tqdm import tqdm
from shutil import copyfile

voice = "Matthew_named"

with open('phrases.json.backup') as f:
    phrases = json.load(f)

with open('ques2vid.pkl', 'rb') as f:
    q2v = pickle.load(f)

for fn in tqdm(os.listdir("Matthew")):
    phrase = phrases[int(fn[:-4])]
    if phrase in q2v:
        copyfile("Matthew/{}".format(fn), "Matthew_named/{}.wav".format(q2v[phrase]))
