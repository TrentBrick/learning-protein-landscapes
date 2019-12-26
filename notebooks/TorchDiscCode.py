"""
Questions: 
What do forward and reverse mean here? 
what is tf.split?

coprime means that the largest divisor for two numbers is 1. 

A finite field of order q exists if and only 
if the order q is a prime power pk (where p is a prime number and k is a positive integer
why does MADE have the output being 2x the input dimension? the neural network returns a location and scale transform!


for the autoregressive the forward direction is slow. backwards is in parallel. 
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import disc_utils


# TODO(trandustin): Move Reverse to another module(?).
class Reverse(nn.Module):
  """Swaps the forward and reverse transformations of a layer."""

  def __init__(self, reversible_layer, **kwargs):
    super(Reverse, self).__init__(**kwargs)
    if not hasattr(reversible_layer, 'reverse'):
      raise ValueError('Layer passed-in has not implemented "reverse" method: '
                       '{}'.format(reversible_layer))
    self.call = reversible_layer.reverse
    self.reverse = reversible_layer.call


class DiscreteAutoregressiveFlow(nn.Module):
  """A discrete reversible layer.
  The flow takes as input a one-hot Tensor of shape `[..., length, vocab_size]`.
  The flow returns a Tensor of same shape and dtype. (To enable gradients, the
  input must have float dtype.)
  For the forward pass, the flow computes in serial:
  ```none
  outputs = []
  for t in range(length):
    new_inputs = [outputs, inputs[..., t, :]]
    net = layer(new_inputs)
    loc, scale = tf.split(net, 2, axis=-1)
    loc = tf.argmax(loc, axis=-1)
    scale = tf.argmax(scale, axis=-1)
    new_outputs = (((inputs - loc) * inverse(scale)) % vocab_size)[..., -1, :]
    outputs.append(new_outputs)
  ```
  For the reverse pass, the flow computes in parallel:
  ```none
  net = layer(inputs)
  loc, scale = tf.split(net, 2, axis=-1)
  loc = tf.argmax(loc, axis=-1)
  scale = tf.argmax(scale, axis=-1)
  outputs = (loc + scale * inputs) % vocab_size
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

  def __init__(self, layer, temperature, vocab_size, **kwargs):
    """Constructs flow.
    Args:
      layer: Two-headed masked network taking the inputs and returning a
        real-valued Tensor of shape `[..., length, 2*vocab_size]`.
        Alternatively, `layer` may return a Tensor of shape
        `[..., length, vocab_size]` to be used as the location transform; the
        scale transform will be hard-coded to 1.
      temperature: Positive value determining bias of gradient estimator.
      **kwargs: kwargs of parent class.
    """
    super(DiscreteAutoregressiveFlow, self).__init__(**kwargs)
    self.layer = layer
    self.temperature = temperature
    self.vocab_size
    '''def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    self.vocab_size = input_shape[-1]
    if isinstance(self.vocab_size, tf1.Dimension):
      self.vocab_size = self.vocab_size.value
    if self.vocab_size is None:
      raise ValueError('The last dimension of the inputs to '
                       '`DiscreteAutoregressiveFlow` should be defined. Found '
                       '`None`.')
    self.built = True'''

    '''def __call__(self, inputs, *args, **kwargs):
    if not isinstance(inputs, random_variable.RandomVariable):
      return super(DiscreteAutoregressiveFlow, self).__call__(
          inputs, *args, **kwargs)
    return transformed_random_variable.TransformedRandomVariable(inputs, self)
    '''

  def forward(self, inputs, **kwargs):
    """Forward pass for left-to-right autoregressive generation.
    Expects to recieve a onehot."""
    #inputs = torch.Tensor(inputs)
    length = inputs.shape[-2]
    if length is None:
      raise NotImplementedError('length dimension must be known. Ensure input is a onehot with 3 dimensions (batch, length, onehot)')
    # Slowly go down the length of the sequence. 
    # Form initial sequence tensor of shape [..., 1, vocab_size]. In a loop, we
    # incrementally build a Tensor of shape [..., t, vocab_size] as t grows.
    outputs = self._initial_call(inputs[:, 0, :], length, **kwargs)
    # TODO(trandustin): Use tf.while_loop. Unrolling is memory-expensive for big
    # models and not valid for variable lengths.
    for t in range(1, length):
        outputs = self._per_timestep_call(outputs,
                                        inputs[..., t, :],
                                        length,
                                        t,
                                        **kwargs)
    return outputs

  def _initial_call(self, new_inputs, length, **kwargs):
    """Returns Tensor of shape [..., 1, vocab_size].
    Args:
      new_inputs: Tensor of shape [..., vocab_size], the new input to generate
        its output.
      length: Length of final desired sequence.
      **kwargs: Optional keyword arguments to layer.
    """
    inputs = new_inputs.unsqueeze(1) #new_inputs[..., tf.newaxis, :] # batch x 1 x onehots
    # TODO(trandustin): To handle variable lengths, extend MADE to subset its
    # input and output layer weights rather than pad inputs.
    batch_ndims = 1 #inputs.shape.ndims - 2
    padded_inputs = F.pad(
        inputs, (0,0,0, length - 1) ) # (padding_left,padding_right, padding_top,padding_bottom)
    
    """
    All this is doing is filling the input up to its length with 0s. 
    [[0, 0]] * 2 + [[0, 50 - 1], [0, 0]] -> [[0, 0], [0, 0], [0, 49], [0, 0]]
    what this means is, dont add any padding to the 0th dimension on the front or back. 
    same for the 2nd dimension (here we assume two tensors are for batches), for the length dimension, 
    add length -1 0s after. 
    
    """
    net = self.layer(padded_inputs, **kwargs) # feeding this into the MADE network. store these as net.
    if net.shape[-1] == 2 * self.vocab_size: # if the network outputted both a location and scale.
      loc, scale = torch.split(net, self.vocab_size, dim=-1) #tf.split(net, 2, axis=-1) # split in two into these variables
      loc = loc[:, 0, :] #
      loc = disc_utils.one_hot_argmax(loc, self.temperature).unsqueeze(1).type(inputs.dtype)
      scale = scale[:, 0, :]
      scale = disc_utils.one_hot_argmax(scale, self.temperature).unsqueeze(1).type(inputs.dtype)
      inverse_scale = disc_utils.multiplicative_inverse(scale, self.vocab_size)
      shifted_inputs = disc_utils.one_hot_minus(inputs, loc)
      outputs = disc_utils.one_hot_multiply(shifted_inputs, inverse_scale)
    elif net.shape[-1] == self.vocab_size:
      loc = net
      loc = loc[..., 0:1, :]
      loc = tf.cast(disc_utils.one_hot_argmax(loc, self.temperature), inputs.dtype)
      outputs = disc_utils.one_hot_minus(inputs, loc)
    else:
      raise ValueError('Output of layer does not have compatible dimensions.')
    return outputs

  def _per_timestep_call(self,
                         current_outputs,
                         new_inputs,
                         length,
                         timestep,
                         **kwargs):
    """Returns Tensor of shape [..., timestep+1, vocab_size].
    Args:
      current_outputs: Tensor of shape [..., timestep, vocab_size], the so-far
        generated sequence Tensor.
      new_inputs: Tensor of shape [..., vocab_size], the new input to generate
        its output given current_outputs.
      length: Length of final desired sequence.
      timestep: Current timestep.
      **kwargs: Optional keyword arguments to layer.
    """
    inputs = tf.concat([current_outputs,
                        new_inputs[..., tf.newaxis, :]], axis=-2)
    # TODO(trandustin): To handle variable lengths, extend MADE to subset its
    # input and output layer weights rather than pad inputs.
    batch_ndims = inputs.shape.ndims - 2
    padded_inputs = tf.pad(
        inputs,
        paddings=[[0, 0]] * batch_ndims + [[0, length - timestep - 1], [0, 0]])
    net = self.layer(padded_inputs, **kwargs)
    if net.shape[-1] == 2 * self.vocab_size:
      loc, scale = tf.split(net, 2, axis=-1)
      loc = loc[..., :(timestep+1), :]
      loc = tf.cast(disc_utils.one_hot_argmax(loc, self.temperature), inputs.dtype)
      scale = scale[..., :(timestep+1), :]
      scale = tf.cast(disc_utils.one_hot_argmax(scale, self.temperature),
                      inputs.dtype)
      inverse_scale = disc_utils.multiplicative_inverse(scale, self.vocab_size)
      shifted_inputs = disc_utils.one_hot_minus(inputs, loc)
      new_outputs = disc_utils.one_hot_multiply(shifted_inputs, inverse_scale)
    elif net.shape[-1] == self.vocab_size:
      loc = net
      loc = loc[..., :(timestep+1), :]
      loc = tf.cast(disc_utils.one_hot_argmax(loc, self.temperature), inputs.dtype)
      new_outputs = disc_utils.one_hot_minus(inputs, loc)
    else:
      raise ValueError('Output of layer does not have compatible dimensions.')
    outputs = tf.concat([current_outputs, new_outputs[..., -1:, :]], axis=-2)
    if not tf.executing_eagerly():
      outputs.set_shape([None] * batch_ndims + [timestep+1, self.vocab_size])
    return outputs

  def reverse(self, inputs, **kwargs):
    """Reverse pass returning the inverse autoregressive transformation."""
    if not self.built:
      self._maybe_build(inputs)

    net = self.layer(inputs, **kwargs)
    if net.shape[-1] == 2 * self.vocab_size:
      loc, scale = tf.split(net, 2, axis=-1)
      scale = tf.cast(disc_utils.one_hot_argmax(scale, self.temperature),
                      inputs.dtype)
      scaled_inputs = disc_utils.one_hot_multiply(inputs, scale)
    elif net.shape[-1] == self.vocab_size:
      loc = net
      scaled_inputs = inputs
    else:
      raise ValueError('Output of layer does not have compatible dimensions.')
    loc = tf.cast(disc_utils.one_hot_argmax(loc, self.temperature), inputs.dtype)
    outputs = disc_utils.one_hot_add(loc, scaled_inputs)
    return outputs

  def log_det_jacobian(self, inputs):
    return tf.cast(0, inputs.dtype)