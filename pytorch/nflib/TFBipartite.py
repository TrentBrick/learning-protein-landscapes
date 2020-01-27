class DiscreteBipartiteFlow(tf.keras.layers.Layer):
  """A discrete reversible layer.
  The flow takes as input a one-hot Tensor of shape `[..., length, vocab_size]`.
  The flow returns a Tensor of same shape and dtype. (To enable gradients, the
  input must have float dtype.)
  For the forward pass, the flow computes:
  ```none
  net = layer(mask * inputs)
  loc, scale = tf.split(net, 2, axis=-1)
  loc = tf.argmax(loc, axis=-1)
  scale = tf.argmax(scale, axis=-1)
  outputs = ((inputs - (1-mask) * loc) * (1-mask) * inverse(scale)) % vocab_size
  ```
  For the reverse pass, the flow computes:
  ```none
  net = layer(mask * inputs)
  loc, scale = tf.split(net, 2, axis=-1)
  loc = tf.argmax(loc, axis=-1)
  scale = tf.argmax(scale, axis=-1)
  outputs = ((1-mask) * loc + (1-mask) * scale * inputs) % vocab_size
  ```
  The modular arithmetic happens in one-hot space.
  If `x` is a discrete random variable, the induced probability mass function on
  the outputs `y = flow(x)` is
  ```none
  p(y) = p(flow.reverse(y)).
  ```
  The location-only transform is always invertible ([integers modulo
  `vocab_size` form an additive group](
  https://en.wikipedia.org/wiki/Modular_arithmetic)). The transform with a scale
  is invertible if the scale and `vocab_size` are coprime (see
  [prime fields](https://en.wikipedia.org/wiki/Finite_field)).
  """

  def __init__(self, layer, mask, temperature, **kwargs):
    """Constructs flow.
    Args:
      layer: Two-headed masked network taking the inputs and returning a
        real-valued Tensor of shape `[..., length, 2*vocab_size]`.
        Alternatively, `layer` may return a Tensor of shape
        `[..., length, vocab_size]` to be used as the location transform; the
        scale transform will be hard-coded to 1.
      mask: binary Tensor of shape `[length]` forming the bipartite assignment.
      temperature: Positive value determining bias of gradient estimator.
      **kwargs: kwargs of parent class.
    """
    super(DiscreteBipartiteFlow, self).__init__(**kwargs)
    self.layer = layer
    self.mask = mask
    self.temperature = temperature

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    self.vocab_size = input_shape[-1]
    if isinstance(self.vocab_size, tf1.Dimension):
      self.vocab_size = self.vocab_size.value
    if self.vocab_size is None:
      raise ValueError('The last dimension of the inputs to '
                       '`DiscreteBipartiteFlow` should be defined. Found '
                       '`None`.')
    self.built = True

  def __call__(self, inputs, *args, **kwargs):
    if not isinstance(inputs, random_variable.RandomVariable):
      return super(DiscreteBipartiteFlow, self).__call__(
          inputs, *args, **kwargs)
    return transformed_random_variable.TransformedRandomVariable(inputs, self)

  def call(self, inputs, **kwargs):
    """Forward pass for bipartite generation."""
    inputs = tf.convert_to_tensor(inputs)
    batch_ndims = inputs.shape.ndims - 2
    mask = tf.reshape(tf.cast(self.mask, inputs.dtype),
                      [1] * batch_ndims + [-1, 1])
    masked_inputs = mask * inputs
    net = self.layer(masked_inputs, **kwargs)
    if net.shape[-1] == 2 * self.vocab_size:
      loc, scale = tf.split(net, 2, axis=-1)
      loc = tf.cast(utils.one_hot_argmax(loc, self.temperature), inputs.dtype)
      scale = tf.cast(utils.one_hot_argmax(scale, self.temperature),
                      inputs.dtype)
      inverse_scale = utils.multiplicative_inverse(scale, self.vocab_size)
      shifted_inputs = utils.one_hot_minus(inputs, loc)
      masked_outputs = (1. - mask) * utils.one_hot_multiply(shifted_inputs,
                                                            inverse_scale)
    elif net.shape[-1] == self.vocab_size:
      loc = net
      loc = tf.cast(utils.one_hot_argmax(loc, self.temperature), inputs.dtype)
      masked_outputs = (1. - mask) * utils.one_hot_minus(inputs, loc)
    else:
      raise ValueError('Output of layer does not have compatible dimensions.')
    outputs = masked_inputs + masked_outputs
    return outputs

  def reverse(self, inputs, **kwargs):
    """Reverse pass for the inverse bipartite transformation."""
    if not self.built:
      self._maybe_build(inputs)

    inputs = tf.convert_to_tensor(inputs)
    batch_ndims = inputs.shape.ndims - 2
    mask = tf.reshape(tf.cast(self.mask, inputs.dtype),
                      [1] * batch_ndims + [-1, 1])
    masked_inputs = mask * inputs
    net = self.layer(masked_inputs, **kwargs)
    if net.shape[-1] == 2 * self.vocab_size:
      loc, scale = tf.split(net, 2, axis=-1)
      scale = tf.cast(utils.one_hot_argmax(scale, self.temperature),
                      inputs.dtype)
      scaled_inputs = utils.one_hot_multiply(inputs, scale)
    elif net.shape[-1] == self.vocab_size:
      loc = net
      scaled_inputs = inputs
    else:
      raise ValueError('Output of layer does not have compatible dimensions.')
    loc = tf.cast(utils.one_hot_argmax(loc, self.temperature), inputs.dtype)
    masked_outputs = (1. - mask) * utils.one_hot_add(loc, scaled_inputs)
    outputs = masked_inputs + masked_outputs
    return outputs

  def log_det_jacobian(self, inputs):
    return tf.cast(0, inputs.dtype)d