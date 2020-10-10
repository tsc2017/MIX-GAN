import scipy.io as sio
import numpy as np

data_dir='/home/tsc/data/mydata/STL-10/stl10_matlab/'

def make_generator(mode, image_path, files):
    if mode=='TRAIN':
        train_data_dir =data_dir+'train.mat'
        load_data = sio.loadmat(train_data_dir)
        all_images_train=load_data['X'].reshape([-1,3,96,96]).transpose([0,1,3,2])
        all_labels_train=load_data['y'].reshape([-1])
        return all_images_train[np.ix_(files)], all_labels_train[np.ix_(files)]-1
    elif mode=='TEST':
        test_data_dir =data_dir+'test.mat'
        load_data = sio.loadmat(test_data_dir)
        all_images_test=load_data['X'].reshape([-1,3,96,96]).transpose([0,1,3,2])
        all_labels_test=load_data['y'].reshape([-1])
        return all_images_test[np.ix_(files)], all_labels_test[np.ix_(files)]-1
def load(mode, data_dir,files):
        return make_generator(mode, data_dir, files)
