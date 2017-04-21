from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

_layer_norm = tf.contrib.layers.layer_norm


class LayerNormGRUCell(tf.contrib.rnn.RNNCell):
  

  def __init__(self, num_units, w_initializer, u_initializer, b_initializer, activation=tf.nn.tanh):
    
    self._num_units = num_units
    self._w_initializer = w_initializer
    self._u_initializer = u_initializer
    self._b_initializer = b_initializer
    self._activation = activation

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def _w_h_initializer(self):

    def _initializer(shape, dtype=tf.float32, partition_info=None):
      num_units = self._num_units
      assert shape == [num_units, 2 * num_units]
      u_z = self._u_initializer([num_units, num_units], dtype, partition_info)
      u_r = self._u_initializer([num_units, num_units], dtype, partition_info)
      return tf.concat([u_z, u_r], 1)

    return _initializer

  def _w_x_initializer(self, input_dim):
    

    def _initializer(shape, dtype=tf.float32, partition_info=None):
      num_units = self._num_units
      assert shape == [input_dim, 2 * num_units]
      w_z = self._w_initializer([input_dim, num_units], dtype, partition_info)
      w_r = self._w_initializer([input_dim, num_units], dtype, partition_info)
      return tf.concat([w_z, w_r], 1)

    return _initializer

  def __call__(self, inputs, state, scope=None):
    """GRU cell with layer normalization."""
    input_dim = inputs.get_shape().as_list()[1]
    num_units = self._num_units

    with tf.variable_scope(scope or "gru_cell"):
      with tf.variable_scope("gates"):
        w_h = tf.get_variable(
            "w_h", [num_units, 2 * num_units],
            initializer=self._w_h_initializer())
        w_x = tf.get_variable(
            "w_x", [input_dim, 2 * num_units],
            initializer=self._w_x_initializer(input_dim))
        z_and_r = (_layer_norm(tf.matmul(state, w_h), scope="layer_norm/w_h") +
                   _layer_norm(tf.matmul(inputs, w_x), scope="layer_norm/w_x"))
        z, r = tf.split(tf.sigmoid(z_and_r), 2, 1)
      with tf.variable_scope("candidate"):
        w = tf.get_variable(
            "w", [input_dim, num_units], initializer=self._w_initializer)
        u = tf.get_variable(
            "u", [num_units, num_units], initializer=self._u_initializer)
        h_hat = (r * _layer_norm(tf.matmul(state, u), scope="layer_norm/u") +
                 _layer_norm(tf.matmul(inputs, w), scope="layer_norm/w"))
      new_h = (1 - z) * state + z * self._activation(h_hat)
    return new_h, new_h
