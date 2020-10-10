import tensorflow.compat.v1 as tf, tensorflow.keras.backend as K
import os, sys
import numpy as np
from tensorflow.python.ops import array_ops
import functools
TF_VERSION=float('.'.join(tf.__version__.split('.')[:2]))
################################################
def rand_flip_left_right(images):
    return tf.image.random_flip_left_right(images)
    B, H, W, C = images.shape.as_list()
    toss=tf.random_uniform([B])<p
    flipped=array_ops.reverse_v2(x, [2])
    return tf.where(toss, flipped, images)

#Based on https://www.kaggle.com/yihdarshieh/make-chris-deotte-s-data-augmentation-faster
def get_rot_mat(rotation):
    batch_size=rotation.shape[0]
    one = tf.ones([batch_size], dtype=tf.float32)
    zero = tf.zeros([batch_size], dtype=tf.float32)
    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    # Intermediate matrix for rotation, shape = (9, batch_size) 
    rotation_matrix_temp = tf.stack([c1, s1, zero, -s1, c1, zero, zero, zero, one], axis=0)
    # shape = (batch_size, 9)
    rotation_matrix_temp = tf.transpose(rotation_matrix_temp)
    # Fianl rotation matrix, shape = (batch_size, 3, 3)
    rotation_matrix = tf.reshape(rotation_matrix_temp, shape=(batch_size, 3, 3))
    return rotation_matrix
def get_scale_mat(height_scale, width_scale):
    batch_size=height_scale.shape[0]
    one = tf.ones([batch_size], dtype=tf.float32)
    zero = tf.zeros([batch_size], dtype=tf.float32)
    # Intermediate matrix for zoom, shape = (9, batch_size) 
    zoom_matrix_temp = tf.stack([height_scale , zero, zero, zero, width_scale, zero, zero, zero, one], axis=0)
    # shape = (batch_size, 9)
    zoom_matrix_temp = tf.transpose(zoom_matrix_temp)
    # Fianl zoom matrix, shape = (batch_size, 3, 3)
    zoom_matrix = tf.reshape(zoom_matrix_temp, shape=(batch_size, 3, 3))
    return   zoom_matrix
def get_shift_mat(height_shift, width_shift): 
    batch_size=height_shift.shape[0]
    one = tf.ones([batch_size], dtype=tf.float32)
    zero = tf.zeros([batch_size], dtype=tf.float32)
    # Intermediate matrix for shift, shape = (9, batch_size) 
    shift_matrix_temp = tf.stack([one, zero, height_shift, zero, one, width_shift, zero, zero, one], axis=0)
    # shape = (batch_size, 9)
    shift_matrix_temp = tf.transpose(shift_matrix_temp)
    # Final shift matrix, shape = (batch_size, 3, 3)
    shift_matrix = tf.reshape(shift_matrix_temp, shape=(batch_size, 3, 3))      
    return shift_matrix

def upsample(x):
    bs, h, w, c = tf.unstack(tf.shape(x))
    x = tf.reshape(x, [bs, h, 1, w, 1, c]) * tf.ones(
       [1, 1, 2, 1, 2, 1])
    return tf.reshape(x, [bs, h * 2, w * 2, c])
def geometry_augment(images,p=1, PADDING="REFLECT"):
    B, real_H, real_W, C = images.shape.as_list()
    #pad_size=real_H//2
    #mirror_pad_images=tf.pad(images, [[0,0],[pad_size, pad_size],[pad_size, pad_size], [0, 0]],'REFLECT') #gradient wrt the padded region is unimplemented on TPUs. Constant padding can result in (possibly non-square) cutout regions at the margins of the final output, which may be desirable.
    #constant_pad_images=tf.pad(images, [[0,0],[pad_size, pad_size],[pad_size, pad_size], [0, 0]],'CONSTANT')
    #images=constant_pad_images#tf.stop_gradient(mirror_pad_images-constant_pad_images)+constant_pad_images
    #images=tf.compat.v1.image.resize(images, [real_H*4,real_W*4])
    #images=tf.stop_gradient(mirror_pad_images-constant_pad_images)+constant_pad_images
    images=upsample(images)
    B, H, W, C=images.shape.as_list()
    DIM = H
    XDIM = DIM%2 #fix for size 331
    m=tf.reshape(tf.eye(3),[1,3,3])
    #Transform the doubled coordinates back to the original coordinates (multiply by S^-1)
    m=get_scale_mat(tf.ones([B])/2,tf.ones([B])/2)@m
    #Horizontal flip
    toss=tf.cast(tf.random.uniform([B])<p,tf.float32)
    m=get_scale_mat(tf.ones([B]),tf.ones([B])-2*toss)@m
    #90-degree rotation
    toss=tf.reshape(tf.random.categorical(tf.tile(tf.math.log([[1-p+p/4, p/4, p/4, p/4]]),[B,1]),1),[B])
    rad1=tf.cast(toss,tf.float32)*np.pi/2
    m = get_rot_mat(rad1)@m
    #Integer translation
    toss=tf.cast(tf.random.uniform([B])<p,tf.float32)
    x_offset=tf.cast(tf.round(toss*tf.random.uniform([B],-.125,.125)*real_W),tf.float32)
    y_offset=tf.cast(tf.round(toss*tf.random.uniform([B],-.125,.125)*real_H),tf.float32)
    m = get_shift_mat(y_offset, x_offset)@m
    #Isotropic scaling
    toss=tf.cast(tf.random.uniform([B])<p,tf.float32)
    scale=tf.exp(toss*tf.random.normal([B],0,0.2*tf.math.log(2.)))
    m = get_scale_mat(scale, scale)@m
    #Pre-rotation  
    p_rot=1.-tf.sqrt(1.- p)
    toss=tf.cast(tf.random.uniform([B])<p_rot,tf.float32)
    rad2=toss*tf.random.uniform([B],-np.pi,np.pi)
    m = get_rot_mat(rad2)@m
    #Anisotropic scaling
    toss=tf.cast(tf.random.uniform([B])<p,tf.float32)
    x_scale=tf.exp(toss*tf.random.normal([B],0,0.2*tf.math.log(2.)))
    y_scale=1./x_scale
    m = get_scale_mat(y_scale, x_scale)@m
    #Post-rotation  
    p_rot=1.-tf.sqrt(1.- p)
    toss=tf.cast(tf.reshape(tf.random.uniform([B])<p_rot,[B]),tf.float32)
    rad3=toss*tf.random.uniform([B],-np.pi,np.pi)
    m = get_rot_mat(rad3)@m
    #Fractional translation
    toss=tf.cast(tf.random.uniform([B])<p,tf.float32)
    x_offset=toss*tf.random.normal([B],0,.125)*real_W
    y_offset=toss*tf.random.normal([B],0,.125)*real_H
    m = get_shift_mat(y_offset, x_offset)@m
    # Transform to the coordinates of the upsampled images (multiply by S)
    m=get_scale_mat(tf.ones([B])*2,tf.ones([B])*2)@m
    
    # LIST DESTINATION PIXEL INDICES
    #x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM ) # TF1.15 or above
    x=tf.reshape(tf.transpose([tf.range(DIM//2,-DIM//2,-1)]*DIM),[-1])
    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )
    z = tf.ones([DIM*DIM],dtype='int32')
    idx = tf.stack( [x,y,z] )
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = tf.linalg.matmul(m, tf.cast(idx, dtype='float32'))  # shape = (batch_size, 3, DIM ** 2)
    idx2 = K.cast(idx2, dtype='int32')  # shape = (batch_size, 3, DIM ** 2)
    if PADDING == "REFLECT": #TPU compatible reflection padding, based on https://www.kaggle.com/psaikko/augmentations-with-reflect-padding
        idx2 = tf.stack([DIM // 2 - idx2[:, 0, ], DIM // 2 + idx2[:, 1, ]], axis=1)
        # Identify out-of-bounds positions
        bounds_mask = tf.math.logical_or(tf.math.less(idx2, 0), tf.math.greater(idx2, DIM-1))
        # Compute mirrored positions
        mirror_idxs = tf.math.subtract(DIM-1, tf.math.floormod(idx2, DIM-1))
        idx2 = tf.where(bounds_mask, mirror_idxs, idx2)
        idx3 = tf.stack([idx2[:, 0,], idx2[:, 1,]], axis=1)
    else:
        #idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)
        idx3 = tf.stack([DIM // 2 - idx2[:, 0, ], DIM // 2 + idx2[:, 1, ]], axis=1)
    # shape = (batch_size, DIM ** 2, 3)
    d = tf.gather_nd(images, tf.transpose(idx3, perm=[0, 2, 1]), batch_dims=1)
    d=tf.reshape(d, (B, DIM, DIM, 3))
    #d=tf.compat.v1.image.resize(d, [real_H*2,real_W*2])
    d=tf.nn.avg_pool(d,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',data_format='NHWC')
    # shape = (batch_size, DIM, DIM, 3)
    images = tf.reshape(d, (B, DIM//2, DIM//2, 3))
    #images=images[:,(real_H)//2:(3*real_H)//2,(real_W)//2:(3*real_W)//2,:]
    return images

########################## Color Transformations ######################################################
def brightness_matrix(B, p):
    one = tf.ones([B], dtype=tf.float32)
    zero = tf.zeros([B], dtype=tf.float32)
    toss=tf.cast(tf.random.uniform([B])<p,tf.float32)
    b=toss*tf.random.normal([B], 0, .2)
    shift_matrix_temp = tf.stack([one,zero,zero,zero, zero,one,zero,zero,  zero,zero,one,zero, b,b,b,one], axis=0)
    shift_matrix_temp = tf.transpose(shift_matrix_temp)
    return tf.reshape(shift_matrix_temp, shape=(B, 4, 4))     
    
def contrast_matrix(B, p):
    one = tf.ones([B], dtype=tf.float32)
    zero = tf.zeros([B], dtype=tf.float32)
    toss=tf.cast(tf.random.uniform([B])<p,tf.float32)
    c=tf.exp(toss*tf.random.normal([B],0,0.5*tf.math.log(2.)))
    scale_matrix_temp=tf.stack([c,zero,zero,zero, zero,c,zero,zero,  zero,zero,c,zero, zero,zero,zero,one], axis=0)
    scale_matrix_temp = tf.transpose(scale_matrix_temp)
    return tf.reshape(scale_matrix_temp, shape=(B, 4, 4))  

def luma_flip_matrix(B, p, v):
    toss=tf.cast(tf.random.uniform([B])<p,tf.float32)
    i=toss*tf.cast(tf.random.uniform([B],0,2,dtype=tf.int32),tf.float32)
    return tf.reshape(tf.eye(4),[1,4,4])-2*tf.reshape(tf.transpose(v)@v,[1,4,4])*tf.reshape(i,[B,1,1])

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians. Based on https://stackoverflow.com/a/6802723
    """
    B=int(theta.shape[0])
    one = tf.ones([B], dtype=tf.float32)
    zero = tf.zeros([B], dtype=tf.float32)
    axis = tf.math.l2_normalize(axis)
    a = tf.math.cos(theta / 2.0)
    b, c, d, dummy = tf.split(-tf.reshape(axis,[1,4]) * tf.reshape(tf.math.sin(theta / 2.0),[B,1]),4, axis=-1)
    b, c, d = tf.reshape(b,[B]), tf.reshape(c,[B]), tf.reshape(d,[B])
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return tf.transpose(tf.stack([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), zero],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), zero],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, zero],
                    [zero, zero, zero, one]]),[2,0,1])
def hue_rotation_matrix(B, p, v):
    toss=tf.cast(tf.random.uniform([B])<p,tf.float32)
    theta=toss*tf.random.uniform([B],-np.pi,np.pi,dtype=tf.float32)
    return rotation_matrix(v,theta)
def saturation_matrix(B, p, v):
    toss=tf.cast(tf.random.uniform([B])<p,tf.float32)
    s=tf.exp(toss*tf.random.normal([B],0,tf.math.log(2.)))
    vv_t=tf.reshape(tf.transpose(v)@v,[1,4,4])
    return vv_t+(tf.reshape(tf.eye(4),[1,4,4])-vv_t)*tf.reshape(s,[B,1,1])
def color_augment(images,p=1):
    B,H,W,_=images.shape.as_list()
    images=images
    ones=tf.ones([B,H,W,1],dtype=tf.float32)
    images=tf.concat([images, ones], -1)
    images=tf.reshape(images,[B,H*W,4])
    C=tf.eye(4)
    v=([[1,1,1,0]]/np.sqrt(3)).astype('float32') #Luma axis
    C@=brightness_matrix(B, p) # Python >= 3.5 required to use "@" for matrix mulplication
    C@=contrast_matrix(B, p)
    C@=luma_flip_matrix(B, p, v)
    C@=hue_rotation_matrix(B, p, v)
    C@=saturation_matrix(B, p, v)
    images=tf.reshape(images@C,[B,H,W,4])
    images=images[:,:,:,:3]#tf.slice(images, [0,0,0,0],[B,H,W,3])
    return images
##################################################################
#Cutout is based on https://github.com/mit-han-lab/data-efficient-gans/blob/7a1ea3d0a1e467b0c74f3bdb79ef9ace5e41c321/DiffAugment_tf.py#L51-L64
def cutout(x, toss, ratio=[1, 2]):
    batch_size = tf.shape(x)[0]
    image_size = tf.shape(x)[1:3]
    cutout_size = image_size * ratio[0] // ratio[1]
    offset_x = tf.random.uniform([tf.shape(x)[0], 1, 1], maxval=image_size[0] + (1 - cutout_size[0] % 2), dtype=tf.int32)
    offset_y = tf.random.uniform([tf.shape(x)[0], 1, 1], maxval=image_size[1] + (1 - cutout_size[1] % 2), dtype=tf.int32)
    grid_batch, grid_x, grid_y = tf.meshgrid(tf.range(batch_size, dtype=tf.int32), tf.range(cutout_size[0], dtype=tf.int32), tf.range(cutout_size[1], dtype=tf.int32), indexing='ij')
    cutout_grid = tf.stack([grid_batch, grid_x + offset_x - cutout_size[0] // 2, grid_y + offset_y - cutout_size[1] // 2], axis=-1)
    mask_shape = tf.stack([batch_size, image_size[0], image_size[1]])
    cutout_grid = tf.maximum(cutout_grid, 0)
    cutout_grid = tf.minimum(cutout_grid, tf.reshape(mask_shape - 1, [1, 1, 1, 3]))
    mask = tf.maximum(1 - tf.reshape(toss,[-1,1,1])*tf.scatter_nd(cutout_grid, tf.ones([batch_size, cutout_size[0], cutout_size[1]], dtype=tf.float32), mask_shape), 0)
    x = x * tf.expand_dims(mask, axis=3)
    return x
def rand_cutout(images,p=1):
    B,H,W,C=images.shape.as_list()
    toss=tf.cast(tf.random.uniform([B])<p,tf.float32)
    images=cutout(images, toss)
    return images
def add_rgb_noise(images,p=1):
    B,H,W,C=images.shape.as_list()
    toss=tf.cast(tf.random.uniform([B,1,1,1])<p,tf.float32)
    sigma=tf.abs(tf.random.normal([B,1,1,1],0,0.1))
    mu=tf.zeros([1,1,1,1])
    n_rgb=tf.random.normal(images.shape,mu,sigma)
    return images+toss*n_rgb
def imagespace_augment(images,p=1, add_noise=True, cutout=True, dropout=True):
    if add_noise:
        images=add_rgb_noise(images,p)
    if cutout:
        images=rand_cutout(images,p)
    if dropout:
        images=tf.nn.dropout(images, rate=p)*(1-p)
    return images
###################################################################
def augment(images, p=0., flip=False, clip=False, geo_aug=True, col_aug=True, imsp_aug=True, add_noise=True, cutout=True, dropout=False):
    p=tf.nn.relu(p)
    if flip:#Dataset x-flip
        images=rand_flip_left_right(images)
    if geo_aug:
        images=geometry_augment(images,p)  
    if col_aug:
        images=color_augment(images,p)
    if imsp_aug:
        images=imagespace_augment(images,p,add_noise,cutout,dropout)
    if clip:
        return tf.clip_by_value(images,-1.,1.)
    else:
        return images