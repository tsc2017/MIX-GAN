import tflib as lib
import numpy as np
import tensorflow.compat.v1 as tf
from tflib.utils import orthogonal_regularizer
from tflib.utils import spectral_norm
_default_weightnorm = False
_default_spectralnorm = 0
group=True
dtype='float32'
norm_weight_names=[]

weight_init = tf.compat.v1.orthogonal_initializer()
weight_regularizer =None # orthogonal_regularizer(0.0001)
def _l2normalize(v, eps=1e-12):
    return tf.math.l2_normalize(v, None, eps)

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

def Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=True, mask_type=None, stride=1, weightnorm=None,spectralnorm=None, biases=True, gain=1., update_collection=None, depthwise=False,channel_multiplier =1,labels=None,weight_init=weight_init, update_sn=None,    weight_regularizer=weight_regularizer):
        """
         inputs: tensor of shape (batch size, num channels, height, width)
         mask_type: one of None, 'a', 'b'

         returns: tensor of shape (batch size, num channels, height, width)
        """
        #with tf.name_scope(name) as scope:

        if mask_type is not None:
            mask_type, mask_n_channels = mask_type

            mask = np.ones(
                (filter_size, filter_size, input_dim, output_dim), 
                dtype=dtype
            )
            center = filter_size // 2

            # Mask out future locations
            # filter shape is (height, width, input channels, output channels)
            mask[center+1:, :, :, :] = 0.
            mask[center, center+1:, :, :] = 0.

            # Mask out future channels
            for i in xrange(mask_n_channels):
                for j in xrange(mask_n_channels):
                    if (mask_type=='a' and i >= j) or (mask_type=='b' and i > j):
                        mask[
                            center,
                            center,
                            i::mask_n_channels,
                            j::mask_n_channels
                        ] = 0.


        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype(dtype)
        

        fan_in = input_dim * filter_size**2
        fan_out = output_dim * filter_size**2 / (stride**2)
        if depthwise:
            fan_in = filter_size**2
            fan_out = channel_multiplier * filter_size**2 / (stride**2)
        if mask_type is not None: # only approximately correct
            fan_in /= 2.
            fan_out /= 2.

        if he_init:
            filters_stdev = np.sqrt(4./(fan_in+fan_out))
        else: # Normalized init (Glorot & Bengio)
            filters_stdev = np.sqrt(2./(fan_in+fan_out))

        if _weights_stdev is not None:
            filter_values = uniform(
                _weights_stdev,
                (filter_size, filter_size, input_dim, output_dim)
            )
        else:
            filter_values = uniform(
                filters_stdev,
                (filter_size, filter_size, input_dim, output_dim)
            )
        if depthwise:
            filter_values = uniform(
            filters_stdev,
            (filter_size, filter_size, input_dim, channel_multiplier)
                 )
     
        # print "WARNING IGNORING GAIN"
        filter_values *= gain

        filters = lib.get_param(name+'.Filters', filter_values.shape, dtype, weight_init, weight_regularizer)
        #filters = lib.param(name+'.Filters', filter_values)
        
        #tf.add_to_collection('G_conv' if 'Generator' in name else 'D_conv',orthogonal_regularizer(0.0001)(filters))
        if depthwise and group:
            filters=tf.reshape(filters,[filter_size, filter_size, 1, input_dim*channel_multiplier])
            #filters=tf.reshape(filters,[filter_size, filter_size, input_dim,channel_multiplier])
        if weightnorm==None:
            weightnorm = _default_weightnorm
        if weightnorm:
            norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0,1,2)))
            target_norms = lib.param(name + '.g',norm_values)
            with tf.name_scope('weightnorm') as scope:
                norms = tf.sqrt(tf.reduce_sum(tf.square(filters), reduction_indices=[0,1,2]))
                filters = filters * (target_norms / (norms+1e-12))

        # spectral normalization      
        power_method_update=tf.zeros([])
        t_update=tf.zeros([])
        if spectralnorm==None:
            spectralnorm = _default_spectralnorm
        if spectralnorm and not depthwise:
            filters=spectral_norm(filters,update_sn=update_sn)
            '''
                v=lib.param(name + '.sn.v',np.random.randn(1, output_dim),dtype=dtype,trainable=False)
                W=tf.reshape(filters,[filter_size*filter_size*input_dim, output_dim])
                new_u =  tf.nn.l2_normalize(tf.matmul(v, tf.transpose(W)))
                new_v =  tf.nn.l2_normalize(tf.matmul(new_u, W))
                new_u = tf.stop_gradient(new_u)
                new_v = tf.stop_gradient(new_v)
                #new_u=tf.random_normal(new_u.shape)
                #new_v=tf.random_normal(new_v.shape)
                #new_u = tf.nn.l2_normalize((new_u),1)
                #new_v = tf.nn.l2_normalize((new_v),1)
                
                spectral_norm = tf.matmul(tf.matmul(new_u, W),tf.transpose(new_v))
                #spectral_norm=tf.math.reduce_logsumexp(tf.abs(W))
                #spectral_norm=tf.svd(W, compute_uv=False)[0]
                #spectral_norm=tf.stop_gradient(spectral_norm)
                
                #filters/=tf.norm(filters)
                if name not in norm_weight_names:
                    print('first of', name)
                    norm_weight_names.append(name)
                    power_method_update = tf.assign(v, new_v)
                    with tf.control_dependencies([power_method_update]):
                        filters=tf.reshape(W/spectral_norm, filters.shape)#*target_norm
                else:
                    print('not first of', name)
                    filters=tf.reshape(W/spectral_norm, filters.shape)
             '''   
                
        if mask_type is not None:
            with tf.name_scope('filter_mask'):
                filters = filters * mask

        if not depthwise or group:
            result = tf.nn.conv2d(
            input=inputs, 
            filter=filters, 
            strides=[1, stride, stride, 1],
            padding='SAME',
            data_format='NHWC')
            if 'Generator' in name:
                rec = tf.nn.conv2d_transpose(
                value=result, 
                filter=filters,
                output_shape=inputs.shape.as_list(), 
                strides=[1,stride,stride, 1],
                padding='SAME'
                 )
                assert inputs.shape==rec.shape
                tf.add_to_collection('REC_LOSS', tf.reduce_mean((tf.stop_gradient(inputs)-rec)**2))
        else:
            result = tf.nn.depthwise_conv2d(
            input=inputs, 
            filter=filters, 
            strides=[1, stride, stride, 1],
            padding='SAME',
            data_format='NHWC')
        if biases:
            _biases = lib.param(
                name+'.Biases',
                np.zeros(output_dim, dtype=dtype)
             )
            result = tf.nn.bias_add(result, _biases, data_format='NHWC')

        return result
