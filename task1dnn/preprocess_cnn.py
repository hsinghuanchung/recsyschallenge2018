import json
import numpy as np
import os
from sys import argv
import pickle
import re
import csv
import gensim
import argparse
from util import DataManager
'''
This code is for preprocessing data for task1 of recsys challenge 2018.

if MODE is LOOKUP_XY:
    python3 cnn.py [path/to/recsys_spotify/data]
    input:
        path/to/recsys_spotify/data
    output:
        0samples.csv

if MODE is SPLIT:
    python3 cnn.py [path/to/save/token] [path/to/save/x] [path/to/save/y]
    e.g.
        python3 cnn.py /mnt/data/b04901058/recsys/token0_Xfull.pk /mnt/data/b04901058/recsys/0_X /mnt/data/b04901058/recsys/0_Y

'''

MODE = 'LOOKUP_XY'

if MODE == 'SPLIT':
    samples = []
    seed_num = 0 

def process_mpd(path):
    count = 1
    filenames = os.listdir(path)
    for filename in sorted(filenames):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            fullpath = os.sep.join((path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            for playlist in mpd_slice['playlists']:
                if MODE == 'DICT':
                    playlist_dict(playlist)
                if MODE == 'SPLIT':
                    playlist_split(playlist)
                    save_samples()
            
            print(str(count) + '/' + str(len(filenames)))
            count += 1


def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def playlist_split(playlist):
    global samples
    if playlist["num_tracks"] <= seed_num: return
    X = []
    Y = []
    X = X + normalize_name(playlist["name"]).split()
    for i in range(playlist["num_tracks"]):
        if i < seed_num: X.append(playlist["tracks"][i]["track_uri"][14:])
        else: Y.append(playlist["tracks"][i]["track_uri"][14:])
    samples.append([X,Y])

def save_samples():
    global samples
    with open('0samples.csv','a') as f:
        for sample in samples:
            for x in sample[0]:
                f.write(x + ' ')
            f.write(',')
            for y in sample[1]:
                f.write(y + ' ')
            f.write('\n')
    samples = []


def new_process_xy(tokenpath,path2x,path2y):
    dm = DataManager()
    dm.add_data('seed', '0samples.csv')
    dm.add_data('truth', '0samples.csv')
    dm.tokenize(230000) #vocab size
    dm.save_tokenizer(tokenpath)
    dm.to_sequence(1) #max length
    dm.save_sequence(path2x)
    dm.tosave_label(path2y)


def main():
    if MODE == 'SPLIT':
        path = argv[1]
        process_mpd(path)
    if MODE == 'LOOKUP_XY':
        tokenpath = argv[1]
        path2x = argv[2]
        path2y = argv[3]
        new_process_x(tokenpath,path2x,path2y)

if __name__ == '__main__':
    main()

