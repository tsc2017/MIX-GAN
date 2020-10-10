import numpy as np
import scipy.misc
import time
import os
#Loading all images into memory all at once will take a long time at startup while not loading them into memory will slow down the program.
#So we adopt the following alternative:
#If an image is accessed for the first time, it will be loaded into memory. 
#Subsequent visits to it will go to memory directly, which results in a 2~3x acceration
data_dir=os.environ['HOME']+'/data/mydata/ut-zap50k'
train_size=40000
test_size=10025
height=width=32
loaded={'TRAIN':np.zeros(train_size),'TEST':np.zeros(test_size)}
all_images={'TRAIN':np.zeros((train_size, 3, height, width), dtype='int32'),'TEST':np.zeros((test_size, 3, height, width), dtype='int32')}
all_sketches={'TRAIN':np.zeros((train_size, 1, height, width), dtype='int32'),'TEST':np.zeros((test_size, 1, height, width), dtype='int32')}
def make_generator(mode, image_path,sketch_path, name_list,permute_channels=True):
        epoch_count = [1]
    #def get_epoch():
        images = np.zeros((len(name_list), 3, height, width), dtype='int32')
        sketches = np.zeros((len(name_list),1, height, width), dtype='int32')
        files = name_list
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        epoch_count[0] += 1
        perm=np.arange(3)
        for n, i in enumerate(files):
            if loaded[mode][i]==0:
                image = scipy.misc.imread("{}/{}.jpg".format(image_path, str(i+1).zfill(len(str(train_size)))),mode='RGB')
                image=scipy.misc.imresize(image, [height, width])
                sketch = scipy.misc.imread("{}/{}.jpg".format(sketch_path, str(i+1).zfill(len(str(train_size)))))
                #print "{}/{}.jpg".format(sketch_path, str(i+1).zfill(len(str(train_size))))
                sketch= scipy.misc.imresize(sketch, [height, width])
                if len(image.shape)!=3:
                    print "{}/{}.jpg".format(image_path, str(i+1).zfill(len(str(train_size))))
                    print image.shape
                all_images[mode][i] = image.transpose(2,0,1)
                all_sketches[mode][i][0] = sketch
                loaded[mode][i]=1
            #images[n] =all_images[mode][i]
            #sketches[n][0] =all_sketches[mode][i][0]
            #if n > 0 and n % batch_size == 0:
                #yield (images,sketches)
        #return images,sketches
            np.random.shuffle(perm)
            if mode=='TRAIN' and permute_channels:
                all_images[mode][i]=all_images[mode][i][perm]
        return all_images[mode][np.ix_(files)], all_sketches[mode][np.ix_(files)]
def load(mode, data_dir,sketch_dir,name_list,permute_channels=True):
    if mode=='TRAIN':
        return make_generator(mode, data_dir+'/train', sketch_dir+'/train',name_list,permute_channels)
    elif mode=='TEST':
        return make_generator(mode, data_dir+'/test', sketch_dir+'/test',name_list,permute_channels)


if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        if i == 1000:
            break
        t0 = time.time()