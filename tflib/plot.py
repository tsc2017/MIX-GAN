import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import time
import pickle
import os, shutil
_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})
first_flush=True

_iter = [0]
def tick():
    _iter[0] += 1

def plot(name, value):
    _since_last_flush[name][_iter[0]] = value

def flush(save_path):
    global _since_beginning, _since_last_flush, first_flush
    if first_flush and os.path.exists(os.path.join(save_path,'log.pkl')):
        try:
            pkl_file=open(os.path.join(save_path,'log.pkl'), 'rb')
            _since_beginning.update(pickle.load(pkl_file))
        except:
            pkl_file=open(os.path.join(save_path,'log.pkl.bak'), 'rb')
            _since_beginning.update(pickle.load(pkl_file))
    first_flush=False 
    pkl_file=os.path.join(save_path,'log.pkl')
    if os.path.exists(pkl_file):
        try:
            backup_pkl=shutil.copyfile(pkl_file, pkl_file+'.bak') # shutil.copy does not work well with gcsfuse and is thus not used here
        except:
            pass
    prints = []

    for name, vals in _since_last_flush.items():
        prints.append("{}={:0,.2f},".format(name, np.mean(list(vals.values()))))
        _since_beginning[name].update(vals)

        x_vals = np.sort(list(_since_beginning[name].keys()))
        y_vals = [_since_beginning[name][x] for x in x_vals]

        plt.clf()
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel(name)
        try:
            plt.savefig(save_path+name.replace(' ', '_')+'.jpg')
        except:
            pass

    print("iter {}\t{}".format(_iter[0], " ".join(prints)))
    _since_last_flush.clear()

    with open(pkl_file, 'wb') as f:
        pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)