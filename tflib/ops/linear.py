import tflib as lib
import numpy as np
import tensorflow.compat.v1 as tf
from tflib.utils import orthogonal_regularizer_fully
from tflib.utils import spectral_norm
_default_weightnorm = False
_default_spectralnorm = 0
dtype='float32'
norm_weight_names=[]

weight_init = tf.compat.v1.orthogonal_initializer()
weight_regularizer = None#orthogonal_regularizer_fully(0.0001)
def _l2normalize(v, eps=1e-12):
    return tf.math.l2_normalize(v, None, eps)

def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

def disable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = False

_weights_stdev = None
def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev

def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None

def Linear(
        name, 
        input_dim, 
        output_dim, 
        inputs,
        biases=True,
        initialization=None,
        weightnorm=None,
        spectralnorm=None,
        gain=1.,
        weight_init=weight_init,
        weight_regularizer=weight_regularizer,
        update_sn=None
        ):
        """
            initialization: None, `lecun`, 'glorot', `he`, 'glorot_he', `orthogonal`, `("uniform", range)`
        """
        #with tf.name_scope(name) as scope:

        def uniform(stdev, size):
            if _weights_stdev is not None:
                stdev = _weights_stdev
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype(dtype)

        if initialization == 'lecun':# and input_dim != output_dim):
            # disabling orth. init for now because it's too slow
            weight_values = uniform(
                np.sqrt(1./input_dim),
                (input_dim, output_dim)
            )

        elif initialization == 'glorot' or (initialization == None):
            weight_values = uniform(
                np.sqrt(2./(input_dim+output_dim)),
                (input_dim, output_dim)
            )

        elif initialization == 'he':

            weight_values = uniform(
                np.sqrt(2./input_dim),
                (input_dim, output_dim)
            )

        elif initialization == 'glorot_he':

            weight_values = uniform(
                np.sqrt(4./(input_dim+output_dim)),
                (input_dim, output_dim)
            )

        elif initialization == 'orthogonal' or \
            (initialization == None and input_dim == output_dim):
            
            # From lasagne
            def sample(shape):
                if len(shape) < 2:
                    raise RuntimeError("Only shapes of length 2 or more are "
                                       "supported.")
                flat_shape = (shape[0], np.prod(shape[1:]))
                 # TODO: why normal and not uniform?
                a = np.random.normal(0.0, 1.0, flat_shape)
                u, _, v = np.linalg.svd(a, full_matrices=False)
                # pick the one with the correct shape
                q = u if u.shape == flat_shape else v
                q = q.reshape(shape)
                return q.astype(dtype)
            weight_values = sample((input_dim, output_dim))
        
        elif initialization[0] == 'uniform':
        
            weight_values = np.random.uniform(
                low=-initialization[1],
                high=initialization[1],
                size=(input_dim, output_dim)
            ).astype(dtype)

        else:
            raise Exception('Invalid initialization!')


        weight_values *= gain

        weight = lib.get_param(name + '.W',weight_values.shape, dtype, weight_init, weight_regularizer)
        #weight = lib.param(name + '.W',weight_values)
        #tf.add_to_collection('G_linear' if 'Generator' in name else 'D_linear',orthogonal_regularizer_fully(0.0001)(weight))
        if weightnorm==None:
            weightnorm = _default_weightnorm
        if weightnorm:
            norm_values = np.sqrt(np.sum(np.square(weight_values), axis=0))
            # norm_values = np.linalg.norm(weight_values, axis=0)

            target_norms = lib.param(
                name + '.g',
                norm_values
            )

            with tf.name_scope('weightnorm') as scope:
                norms = tf.sqrt(tf.reduce_sum(tf.square(weight), reduction_indices=[0]))
                weight = weight * (target_norms / (norms+1e-12))

        # spectral normalization      
        power_method_update=tf.zeros([])
        t_update=tf.zeros([])
        if spectralnorm==None:
            spectralnorm = _default_spectralnorm
        if spectralnorm:
            weight=spectral_norm(weight,update_sn=update_sn)
            '''
                v=lib.param(name + '.sn.v',np.random.randn(1, output_dim),dtype=dtype,trainable=False)
                #t=tf.Variable(10.,dtype=dtype, trainable=False)
                W=tf.reshape(weight,[input_dim, output_dim])
                new_u = _l2normalize(tf.matmul(v, tf.transpose(W)))
                new_v = _l2normalize(tf.matmul(new_u, W))
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
                #t_update=tf.assign(t,tf.maximum(1.,t-0.01))
                
                if name not in norm_weight_names:
                    norm_weight_names.append(name)
                    power_method_update = tf.assign(v, new_v)
                    with tf.control_dependencies([power_method_update]):
                        weight=tf.reshape(W/spectral_norm, weight.shape)#*target_norm
                else:
                    weight=tf.reshape(W/spectral_norm, weight.shape)
          '''          
        # if 'Discriminator' in name:
        #     print "WARNING weight constraint on {}".format(name)
        #     weight = tf.nn.softsign(10.*weight)*.1

        if inputs.get_shape().ndims == 2:
            result = tf.matmul(inputs, weight)
        else:
            reshaped_inputs = tf.reshape(inputs, [-1, input_dim])
            result = tf.matmul(reshaped_inputs, weight)
            result = tf.reshape(result, tf.pack(tf.unpack(tf.shape(inputs))[:-1] + [output_dim]))
        if 'Generator' in name:
                rec = tf.matmul(result, tf.transpose(weight))
                assert inputs.shape==rec.shape
                tf.add_to_collection('REC_LOSS', tf.reduce_mean((tf.stop_gradient(inputs)-rec)**2))

        if biases:
            result = tf.nn.bias_add(
                result,
                lib.param(
                    name + '.b',
                    np.zeros((output_dim,), dtype=dtype)
                )
            )

        return result