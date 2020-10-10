import numpy
import scipy.misc
import os
import urllib
import gzip
import pickle
height=width=32
def mnist_generator(data, batch_size, n_labelled, limit=None):
    images, targets = data

    rng_state = numpy.random.get_state()
    numpy.random.shuffle(images)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(targets)
    if limit is not None:
        print("WARNING ONLY FIRST {} MNIST DIGITS".format(limit))
        images = images.astype('float32')[:limit]
        targets = targets.astype('int32')[:limit]
    if n_labelled is not None:
        labelled = numpy.zeros(len(images), dtype='int32')
        labelled[:n_labelled] = 1
    images=images.reshape(-1,28,28)
    reshape_images=numpy.zeros((images.shape[0],height,width))
    for i in range(images.shape[0]):
        reshape_images[i]=scipy.misc.imresize(images[i], [height, width])
    images=reshape_images/255. #scale changes from [0,1] to [0,255], so need to renormalize
    def get_epoch():
        rng_state = numpy.random.get_state()
        numpy.random.shuffle(images)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(targets)

        if n_labelled is not None:
            numpy.random.set_state(rng_state)
            numpy.random.shuffle(labelled)
        image_batches = images.reshape(-1, height*width)
        target_batches = targets.reshape(-1)

        if n_labelled is not None:
            labelled_batches = labelled.reshape(-1, batch_size)

            for i in range(len(image_batches)//batch_size):
                yield (numpy.copy(image_batches[i*batch_size:(i+1)*batch_size]), numpy.copy(target_batches[i*batch_size:(i+1)*batch_size]), numpy.copy(labelled))

        else:

            for i in range(len(image_batches)//batch_size):
                yield (numpy.copy(image_batches[i*batch_size:(i+1)*batch_size]), numpy.copy(target_batches[i*batch_size:(i+1)*batch_size]))

    return get_epoch

def load(batch_size, test_batch_size, n_labelled=None):
    filepath = os.environ['HOME']+'/data/mydata/mnist.pkl.gz'
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print("Couldn't find MNIST dataset in /tmp, downloading...")
        urllib.urlretrieve(url, filepath)

    with gzip.open(filepath, 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f)

    return (
        mnist_generator(train_data, batch_size, n_labelled), 
        mnist_generator(dev_data, test_batch_size, n_labelled), 
        mnist_generator(test_data, test_batch_size, n_labelled)
    )