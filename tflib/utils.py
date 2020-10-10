import tensorflow as tf
import tflib as lib
dtype='float32'
def orthogonal_regularizer(scale) :
    """ Defining the Orthogonal regularizer and return the function at last to be used in Conv layer as kernel regularizer"""

    def ortho_reg(w) :
        """ Reshaping the matrxi in to 2D tensor for enforcing orthogonality"""
        w = tf.reshape(w, [-1, int(w.shape[-1])])
        w=tf.transpose(w) if w.shape[0]<w.shape[1] else w
        """ Declaring a Identity Tensor of appropriate size"""
        
        identity = tf.eye(int(w.shape[-1]), dtype=dtype)
        
        """ Regularizer Wt*W - I """
        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        reg = w_mul*(1-identity)

        """Calculating the Loss Obtained"""
        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return ortho_reg

def orthogonal_regularizer_fully(scale) :
    """ Defining the Orthogonal regularizer and return the function at last to be used in Fully Connected Layer """

    def ortho_reg_fully(w) :
        """ Reshaping the matrix in to 2D tensor for enforcing orthogonality"""
        
        w=tf.transpose(w) if w.shape[0]<w.shape[1] else w

        """Declaring a Identity Tensor of appropriate size"""
        identity = tf.eye(int(w.shape[-1]), dtype=dtype)
        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        reg = w_mul*(1-identity)

        """ Calculating the Loss """
        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return ortho_reg_fully

def spectral_norm(inputs, epsilon=1e-6, singular_value="auto", normalize_G=1 , update_sn=None):
  """Performs Spectral Normalization on a weight tensor.
  Details of why this is helpful for GAN's can be found in "Spectral
  Normalization for Generative Adversarial Networks", Miyato T. et al., 2018.
  [https://arxiv.org/abs/1802.05957].
  Args:
    inputs: The weight tensor to normalize.
    epsilon: Epsilon for L2 normalization.
    singular_value: Which first singular value to store (left or right). Use
      "auto" to automatically choose the one that has fewer dimensions.
  Returns:
    The normalized weight tensor.
  """
  if not normalize_G and 'Generator' in inputs.name:
        return inputs
  if len(inputs.shape) < 2:
    raise ValueError(
        "Spectral norm can only be applied to multi-dimensional tensors")

  # The paper says to flatten convnet kernel weights from (C_out, C_in, KH, KW)
  # to (C_out, C_in * KH * KW). Our Conv2D kernel shape is (KH, KW, C_in, C_out)
  # so it should be reshaped to (KH * KW * C_in, C_out), and similarly for other
  # layers that put output channels as last dimension. This implies that w
  # here is equivalent to w.T in the paper.
  w = tf.reshape(inputs, (-1, inputs.shape[-1]))

  # Choose whether to persist the first left or first right singular vector.
  # As the underlying matrix is PSD, this should be equivalent, but in practice
  # the shape of the persisted vector is different. Here one can choose whether
  # to maintain the left or right one, or pick the one which has the smaller
  # dimension. We use the same variable for the singular vector if we switch
  # from normal weights to EMA weights.
  var_name = inputs.name.replace("/ExponentialMovingAverage", "").split("/")[-1]
  var_name = var_name.split(":")[0] + ".u_var"
  if singular_value == "auto":
    singular_value = "left" if w.shape[0] <= w.shape[1] else "right"
  u_shape = (w.shape[0], 1) if singular_value == "left" else (1, w.shape[-1])
  u_var = lib.get_param(
      var_name,
      shape=u_shape,
      dtype=w.dtype,
      initializer=tf.random_normal_initializer(),
      trainable=False, aggregation=tf.VariableAggregation.MEAN)
  u = u_var

  # Use power iteration method to approximate the spectral norm.
  # The authors suggest that one round of power iteration was sufficient in the
  # actual experiment to achieve satisfactory performance.
  power_iteration_rounds = 1
  for _ in range(power_iteration_rounds):
    if singular_value == "left":
      # `v` approximates the first right singular vector of matrix `w`.
      v = tf.math.l2_normalize(
          tf.matmul(tf.transpose(w), u), axis=None, epsilon=epsilon)
      u = tf.math.l2_normalize(tf.matmul(w, v), axis=None, epsilon=epsilon)
    else:
      v = tf.math.l2_normalize(tf.matmul(u, w, transpose_b=True),
                               epsilon=epsilon)
      u = tf.math.l2_normalize(tf.matmul(v, w), epsilon=epsilon)

  # Update the approximation.
  def get_u_with_update():
      with tf.control_dependencies([tf.compat.v1.assign(u_var, u, name="update_u")]):
        return tf.identity(u)
  def get_u_without_update():
        return tf.identity(u)
  if update_sn is None:
    update_sn=tf.constant(True)
  u = tf.cond(update_sn, get_u_with_update, get_u_without_update)

  # The authors of SN-GAN chose to stop gradient propagating through u and v
  # and we maintain that option.
  u = tf.stop_gradient(u)
  v = tf.stop_gradient(v)

  if singular_value == "left":
    norm_value = tf.matmul(tf.matmul(tf.transpose(u), w), v)
  else:
    norm_value = tf.matmul(tf.matmul(v, w), u, transpose_b=True)
  norm_value.shape.assert_is_fully_defined()
  norm_value.shape.assert_is_compatible_with([1, 1])

  w_normalized = w / norm_value

  # Deflate normalized weights to match the unnormalized tensor.
  w_tensor_normalized = tf.reshape(w_normalized, inputs.shape)
  return w_tensor_normalized
