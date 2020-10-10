import numpy as np
import scipy.misc
import time
import os
#Loading all images into memory all at once will take a long time at startup while not loading them into memory will slow down the program.
#So we adopt the following alternative:
#If an image is accessed for the first time, it will be loaded into memory. 
#Subsequent visits to it will go to memory directly, which results in a 2~3x acceration
data_dir=os.environ['HOME']+'/data/mydata/CelebA_aligned/CelebA_crop'
train_size=200000 # real training set size, should not be modified
test_size=2599 # real test set size, should not be modified
height=width=32 # can be modified
loaded=np.zeros(train_size+test_size)
all_images=[]
all_labels=[]
indices=[]
attr_file=os.environ['HOME']+'/data/mydata/CelebA_aligned/list_attr_celeba.txt'
with open(attr_file, 'r') as csvfile:
    line = csvfile.readline().strip()
    num_attr_records=int(line)
    line = csvfile.readline().strip()
    attr_names=line.split(' ')
    attrs=np.loadtxt(attr_file,dtype=int,skiprows=2,usecols=np.arange(40)+1)

def set_height(HEIGHT):
    global height,width,all_images
    height=width=HEIGHT
    all_images=np.zeros((train_size+test_size, 3, height, width), dtype='uint8')

def set_chosen_attr(chosen_attr_name):
    assert(len(chosen_attr_name)>0)
    global all_labels,indices,attr_names
    chosen_attr_idx=np.zeros(len(chosen_attr_name),dtype=int)
    for i in range(len(chosen_attr_name)):
        chosen_attr_idx[i]=attr_names.index(chosen_attr_name[i])
    all_labels=attrs[:,chosen_attr_idx]
    all_labels[all_labels==-1]=0
    print('Balancing training set attributes...')
    balanced_indices={}
    num_attrs=len(chosen_attr_name)
    for i in range(2**num_attrs):
        balanced_indices[bin(i)[2:].zfill(num_attrs)]=[]
    print( balanced_indices)
    for i in range(len(all_labels)):
        balanced_indices[''.join(all_labels[i].astype(str)).zfill(num_attrs)].append(i)
    min_length=len(all_labels)
    for i in range(2**num_attrs):
        min_length=min(min_length,len(balanced_indices[bin(i)[2:].zfill(num_attrs)]))
    for j in range(min_length):
        for i in range(2**num_attrs):
            indices+=[balanced_indices[bin(i)[2:].zfill(num_attrs)][j]]
    
    indices=np.array(indices)

    #np.random.shuffle(indices)
    print('Training set size=%i'%len(indices))
    return len(indices)

def set_chosen_attr(chosen_attr_idx):
    print('chosen attrs:', chosen_attr_idx)
    global all_labels,indices
    all_labels=attrs#[:,chosen_attr_idx]
    all_labels[all_labels==-1]=0
    #all_labels[:,list(set(np.arange(40))-set(chosen_attr_idx))]=0
    #indices=np.array(indices)
    indices=np.array(range(train_size))
    #np.random.shuffle(indices)
    print('Training set size=%i'%len(indices))
    return len(indices)

def make_generator(mode, image_path, name_list,permute_channels=False):
        global height,width,all_images        
        epoch_count = [1]
    #def get_epoch():
        images = np.zeros((len(name_list), 3, height, width), dtype='uint8')
        files = indices[name_list] #if mode=='TRAIN' else name_list
        #random_state = np.random.RandomState(epoch_count[0])
        #random_state.shuffle(files)
        epoch_count[0] += 1
        for n, i in enumerate(files):
            if loaded[i]==0:
                image = scipy.misc.imread("{}/{}.jpg".format(image_path, str(i+1).zfill(len(str(train_size+test_size)))),mode='RGB')
                image=scipy.misc.imresize(image, [height, width])
                all_images[i] = image.transpose(2,0,1)
                loaded[i]=1
        return all_images[np.ix_(files)], all_labels[np.ix_(files)]
def load(mode, data_dir,name_list,permute_channels=False):
        return make_generator(mode, data_dir, name_list,permute_channels)


def get_prior():
    return all_labels.sum(axis=0)/np.float32(all_labels.shape[0])

if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print ("{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0]))
        if i == 1000:
            break
        t0 = time.time()