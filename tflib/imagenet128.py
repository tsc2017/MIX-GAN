import tflib as lib
import pickle
import numpy as np
import os
import urllib
import gzip
import time
import multiprocessing
import threading
import queue

epoch=[1]

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict['data'], dict['labels']

def enqueue_pickle(filename, que, reading_lock, size):
    reading_lock.acquire()
    try:
        data, labels = unpickle(filename)
        labels = np.array(labels)
        data = data.reshape([-1, 3, size, size]).transpose([0, 2, 3, 1]).reshape([-1, size * size * 3])
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        que.put((data[indices], labels[indices]))
    except:
        print('Cannot read %s' % filename)
    reading_lock.release()    
    
def cifar_generator(filenames, batch_size, data_dir):
    pickles_queue =queue.Queue(maxsize = 2)
    reading_lock = threading.Lock()
    def get_epoch():
        print('Epoch %i' % epoch[0])
        epoch[0]+=1
        np.random.shuffle(filenames)
        for filename in filenames:  
            threading.Thread(target=enqueue_pickle, args=(data_dir + filename, pickles_queue,reading_lock, 128)).start()
            #lib.read_pickle.read_pickle(data_dir + '/' + filename, pickles_queue)
        count = 0
        while 1:
            #print('queue length: ', pickles_queue.qsize())
            data, labels = pickles_queue.get()
            labels = labels -1
            count+=1
            for i in range(data.shape[0] // batch_size):
                yield data[i * batch_size:(i + 1) * batch_size], labels[i * batch_size:(i + 1) * batch_size]
            pickles_queue.task_done()
            if count == len(filenames)-1:
                #assert pickles_queue.qsize()==1
                #pool.close()
                #pool.join()
                break
    return get_epoch

def load(mode, batch_size, data_dir):
    if mode=='TRAIN':
        return cifar_generator(['train_data_batch_%i'%(i+1) for i in range(100)], batch_size, data_dir)
    else:
        return cifar_generator(['val_data_batch_%i' % (i + 1) for i in range(10)], batch_size, data_dir)
