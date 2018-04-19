#! /usr/bin/python3
import json
import pickle

decoder = json.JSONDecoder()

with open('../bottom-up-attention-vqa/data/v2_OpenEnded_mscoco_train2014_questions.json', 'r') as f:
    trainQ = decoder.decode(f.read())
with open('../bottom-up-attention-vqa/data/v2_OpenEnded_mscoco_val2014_questions.json', 'r') as f:
    valQ = decoder.decode(f.read())
with open('../bottom-up-attention-vqa/data/v2_OpenEnded_mscoco_test-dev2015_questions.json', 'r') as f:
    devQ = decoder.decode(f.read())
with open('../bottom-up-attention-vqa/data/v2_OpenEnded_mscoco_test2015_questions.json', 'r') as f:
    testQ = decoder.decode(f.read())
with open('./answers/lookup.pkl', 'rb') as f:
    ansData = pickle.load(f)

questions = {}

for q in trainQ["questions"]:
    qq = q["question"]
    if qq in questions:
        if not questions[qq]:
            questions[qq] = []
        questions[qq] = questions[qq].append((q["question_id"], q["image_id"]))
    else:
        questions[qq] = [(q["question_id"], q["image_id"])]

trainQuestions = questions
questions = {}

for q in valQ["questions"]:
    qq = q["question"]
    if qq in questions:
        if not questions[qq]:
            questions[qq] = []
        questions[qq] = questions[qq].append((q["question_id"], q["image_id"]))
    else:
        questions[qq] = [(q["question_id"], q["image_id"])]

valQuestions = questions
questions = {}

for q in devQ["questions"]:
    qq = q["question"]
    if qq in questions:
        if not questions[qq]:
            questions[qq] = []
        questions[qq] = questions[qq].append((q["question_id"], q["image_id"]))
    else:
        questions[qq] = [(q["question_id"], q["image_id"])]

devQuestions = questions
questions = {}

for q in testQ["questions"]:
    qq = q["question"]
    if qq in questions:
        if not questions[qq]:
            questions[qq] = []
        questions[qq] = questions[qq].append((q["question_id"], q["image_id"]))
    else:
        questions[qq] = [(q["question_id"], q["image_id"])]

testQuestions = questions

questions = set(trainQuestions.keys()).union(set(valQuestions.keys()).union(set(devQuestions.keys()).union(set(testQuestions.keys()))))

trainQLen = len(trainQuestions)
valQLen = len(valQuestions)
devQLen = len(devQuestions)
testQLen = len(testQuestions)
trainQChars = sum(map(len, trainQuestions.keys()))
valQChars = sum(map(len, valQuestions.keys()))
devQChars = sum(map(len, devQuestions.keys()))
testQChars = sum(map(len, testQuestions.keys()))
totalQChars = sum(map(len, questions))

totalAChars = sum(map(len, ansData.keys()))

print("Train question count: " + str(trainQLen))
print("Train questions characters: " + str(trainQChars))
print("Val question count: " + str(valQLen))
print("Val questions characters: " + str(valQChars))
print("Dev question count: " + str(devQLen))
print("Dev questions characters: " + str(devQChars))
print("Test question count: " + str(testQLen))
print("Test questions characters: " + str(testQChars))
print("Total Question characters: " + str(trainQChars + valQChars + devQChars + testQChars))
print("Total Question characters merged: " + str(sum(map(len, list(questions)))))
print("Answer characters: " + str(totalAChars))
print("Total Question + Answer characters: " + str(totalQChars + totalAChars))

output = list(questions) + list(ansData.keys())

with open('phrases.json', 'w') as f:
    json.dump(output, f)