import pickle
import numpy as np
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict['data'], dict['labels']

def enqueue_pickle(filename, que, size):
    try:
        print('trying to read', filename)
        data, labels = unpickle(filename)
        labels = np.array(labels)
        data = data.reshape([-1, 3, size, size]).transpose([0, 2, 3, 1]).reshape([-1, size * size * 3])
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        que.put((data[indices], labels[indices]))
    except:
        print('Cannot read %s' % filename)