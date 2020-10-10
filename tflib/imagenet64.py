import numpy as np

import os
import urllib
import gzip
import pickle
import gc
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo,encoding='latin1')
    fo.close()
    return dict['data'], dict['labels']

def cifar_generator(filenames, batch_size, data_dir):
    all_data = []
    all_labels = []
    for filename in filenames:        
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0).reshape([-1,3,64,64]).transpose([0,2,3,1]).reshape([-1,64*64*3])
    labels = np.concatenate(all_labels, axis=0)-1

    print('All training data loaded into memory.')# 1281167×64×64×3×8 bits = 15.74 GB
    del all_data, all_labels
    gc.collect()
    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in range(len(images) // batch_size):
            yield (images[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])

    return get_epoch


def load(mode, batch_size, data_dir):
    if mode=='TRAIN':
        return cifar_generator(['train_data_batch_%i'%(i+1) for i in range(100)], batch_size, data_dir)
    else:
        return cifar_generator(['val_data_batch_%i' % (i + 1) for i in range(10)], batch_size, data_dir)
