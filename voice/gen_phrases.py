#! /usr/bin/python3
from boto3 import Session
from botocore.exceptions import BotoCoreError, ClientError
from contextlib import closing
from tqdm import tqdm
import wave
import os
import sys
import json
import argparse

'''
input_path = './phrases.json'
voice = "Matthew"
api = "AWS"
sample_rate = 16000
output_path = None
audio_format = "wav"
emphasis = "none"
'''

parser = argparse.ArgumentParser(description="Use a remote TTS API to generate audio for prompts.")
parser.add_argument("--input_path", type=str, help="Path to json containing list of phrases.", default='./phrases.json')
parser.add_argument("--api", type=str, help="Choice of API", choices=["AWS"], default="AWS")
parser.add_argument("--voice", type=str, help="Name of speaker", default=None)
parser.add_argument("--sample_rate", type=int, help="Samples per second", choices=[8000, 16000], default=16000)
parser.add_argument("--audio_format", type=str, help="File format for audio files", choices=["wav"], default="wav")
parser.add_argument("--emphasis", type=str, help="Level of emphasis on voice", choices=["none", "some", "lots"], default="none")
args = parser.parse_args()

if not args.voice:
    print("Name of speaker ('--voice')")
    sys.exit(-1)

with open(args.input_path, 'rb') as f:
    phrases = json.load(f)

session = Session(profile_name="default")
polly = session.client("polly")

for idx, text in enumerate(tqdm(phrases[0:2], total=len(phrases), unit="phrases", desc=args.api)):
    try:
        if args.emphasis is "none":
            response = polly.synthesize_speech(Text=text, OutputFormat="pcm", VoiceId=args.voice, SampleRate=str(args.sample_rate))
        else:
            emphasis_tags = ("<emphasis>" if args.emphasis is "some" else "<emphasis level=\"strong\">", "</emphasis>")
            request_text = "<speech>" + emphasis_tags[0] + text + emphasis_tags[1] + "</speech>"
            response = polly.synthesize_speech(Text=request_text, OutputFormat="pcm", VoiceId=args.voice, SampleRate=args.sample_rate, Type="ssml")
    except (BotoCoreError, ClientError) as error:
        print(error)
        sys.exit(-1)
    
    if "AudioStream" in response:
        with closing(response["AudioStream"]) as stream:
            if not os.path.exists("questions"):
                os.makedirs("questions")
            if not os.path.exists("questions/{}".format(args.voice)):
                os.makedirs("questions/{}".format(args.voice))
            output = "./questions/{}/{}.{}".format(args.voice, idx, args.audio_format)
            try:
                with wave.open(output, 'wb') as wav:
                    wav.setparams((1, 2, args.sample_rate, 0, 'NONE', 'NONE'))
                    wav.writeframes(stream.read())
            except IOError as error:
                print(error)
                sys.exit(-1)
    else:
        print("No audio stream found.")
        sys.exit(-1)

print("Done")
print("Saved {} wav files to {}.".format(len(phrases), "./questions/{}".format(args.voice)))
print("Characters spent: " + str(sum(map(len, phrases))))
