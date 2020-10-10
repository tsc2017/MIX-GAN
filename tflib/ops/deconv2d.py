import tflib as lib

import numpy as np
import tensorflow as tf
from tflib.utils import orthogonal_regularizer
_default_weightnorm = False
_default_spectralnorm =0
dtype='float32'
norm_weight_names=[]
weight_init = tf.compat.v1.truncated_normal_initializer(mean=0.0, stddev=0.02)
weight_init = tf.compat.v1.orthogonal_initializer()
weight_regularizer = orthogonal_regularizer(0.0001)
def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

_weights_stdev = None
def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev

def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None

def Deconv2D(
    name, 
    input_dim, 
    output_dim, 
    filter_size, 
    inputs, 
    stride=2,
    he_init=True,
    weightnorm=None,
    spectralnorm=None,
    biases=True,
    gain=1.,
    mask_type=None,
    ):
    """
    inputs: tensor of shape (batch size, height, width, input_dim)
    returns: tensor of shape (batch size, 2*height, 2*width, output_dim)
    """
    with tf.name_scope(name) as scope:

        if mask_type != None:
            raise Exception('Unsupported configuration')

        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')
        if stride==None:
            stride = 2
        fan_in = input_dim * filter_size**2 / (stride**2)
        fan_out = output_dim * filter_size**2

        if he_init:
            filters_stdev = np.sqrt(4./(fan_in+fan_out))
        else: # Normalized init (Glorot & Bengio)
            filters_stdev = np.sqrt(2./(fan_in+fan_out))


        if _weights_stdev is not None:
            filter_values = uniform(
                _weights_stdev,
                (filter_size, filter_size, output_dim, input_dim)
            )
        else:
            filter_values = uniform(
                filters_stdev,
                (filter_size, filter_size, output_dim, input_dim)
            )

        filter_values *= gain

        filters = lib.get_param(name+'.Filters',filter_values.shape, filter_values.dtype, weight_init, weight_regularizer)
        #filters = lib.param(name+'.Filters', filter_values)
        #tf.add_to_collection('G_deconv',orthogonal_regularizer(0.0001)(filters))
        
        if weightnorm==None:
            weightnorm = _default_weightnorm
        if weightnorm:
            norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0,1,3)))
            target_norms = lib.param(
                name + '.g',
                norm_values
            )
            with tf.name_scope('weightnorm') as scope:
                norms = tf.sqrt(tf.reduce_sum(tf.square(filters), reduction_indices=[0,1,3]))
                filters = filters * tf.expand_dims(target_norms / norms, 1)

        # spectral normalization      
        power_method_update=tf.zeros([])
        t_update=tf.zeros([])
        if spectralnorm==None:
            spectralnorm = _default_spectralnorm
        if spectralnorm:
                if name not in norm_weight_names:
                    norm_weight_names.append(name)
                v=lib.param(name + '.sn.v',np.random.randn(1, input_dim),dtype=dtype,trainable=False)

                t=tf.Variable(10.,dtype=dtype, trainable=False)
                W=tf.reshape(filters,[filter_size*filter_size*output_dim, input_dim])
                new_u =  tf.nn.l2_normalize(tf.matmul(v, tf.transpose(W)))
                new_v =  tf.nn.l2_normalize(tf.matmul(new_u, W))
                new_u = tf.stop_gradient(new_u)
                new_v = tf.stop_gradient(new_v)
                #new_u=tf.random_normal(new_u.shape)
                #new_v=tf.random_normal(new_v.shape)
                #new_u = tf.nn.l2_normalize((new_u),1)
                #new_v = tf.nn.l2_normalize((new_v),1)
                
                spectral_norm = tf.matmul(tf.matmul(new_u, W),tf.transpose(new_v))
                #spectral_norm=tf.svd(W, compute_uv=False)[0]
                #spectral_norm=tf.stop_gradient(spectral_norm)
                
                #filters/=tf.norm(filters)
                t_update=tf.assign(t,tf.maximum(1.,t-0.01))
                power_method_update = tf.assign(v, new_v)
                with tf.control_dependencies([power_method_update,t_update]):
                    filters=tf.reshape(W/spectral_norm, filters.shape)#*target_norm
                
        #inputs = tf.transpose(inputs, [0,2,3,1], name='NCHW_to_NHWC')

        input_shape = tf.shape(inputs)
        try: # tf pre-1.0 (top) vs 1.0 (bottom)
            output_shape = tf.pack([input_shape[0], stride*input_shape[1], stride*input_shape[2], output_dim])
        except Exception as e:
            output_shape = tf.stack([input_shape[0], stride*input_shape[1], stride*input_shape[2], output_dim])

        result = tf.nn.conv2d_transpose(
            value=inputs, 
            filter=filters,
            output_shape=output_shape, 
            strides=[1,stride,stride, 1],
            padding='SAME'
         )

        if biases:
            _biases = lib.param(
                name+'.Biases',
                np.zeros(output_dim, dtype='float32')
            )
            result = tf.nn.bias_add(result, _biases)

        #result = tf.transpose(result, [0,3,1,2], name='NHWC_to_NCHW')


        return result
