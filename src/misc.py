import os
import pickle
import re
from itertools import permutations

def load_pickle(filename, path):
    completeName = os.path.join(path, filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_as_pickle(filename, data, path):
    completeName = os.path.join(path, filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)