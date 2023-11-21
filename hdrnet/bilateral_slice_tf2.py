
# import jax
# import jax.numpy as jnp
import tensorflow as tf
from numerics_tf2 import lerp_weight
from numerics_tf2 import smoothed_lerp_weight
from numerics_tf2 import smoothed_lerp_weight_grad
import numpy as np


def bilateral_slice(grid, guide):
  ii, jj = tf.meshgrid(
      tf.range(guide.shape[0],dtype=tf.float32), tf.range(guide.shape[1],dtype=tf.float32), indexing='ij')

  scale_i = grid.shape[0] / guide.shape[0]
  scale_j = grid.shape[1] / guide.shape[1]

  gif = (ii + 0.5) * scale_i
  gjf = (jj + 0.5) * scale_j
  gkf = guide * grid.shape[2]

  # Compute trilinear interpolation weights without clamping.
  gi0 = tf.floor(gif - 0.5)
  gj0 = tf.floor(gjf - 0.5)
  gk0 = tf.floor(gkf - 0.5)
  gi1 = gi0 + 1
  gj1 = gj0 + 1
  gk1 = gk0 + 1

  wi0 = lerp_weight(gi0 + 0.5, gif)
  wi1 = lerp_weight(gi1 + 0.5, gif)
  wj0 = lerp_weight(gj0 + 0.5, gjf)
  wj1 = lerp_weight(gj1 + 0.5, gjf)
  wk0 = smoothed_lerp_weight(gk0 + 0.5, gkf)
  wk1 = smoothed_lerp_weight(gk1 + 0.5, gkf)

  w_000 = wi0 * wj0 * wk0
  w_001 = wi0 * wj0 * wk1
  w_010 = wi0 * wj1 * wk0
  w_011 = wi0 * wj1 * wk1
  w_100 = wi1 * wj0 * wk0
  w_101 = wi1 * wj0 * wk1
  w_110 = wi1 * wj1 * wk0
  w_111 = wi1 * wj1 * wk1


  # But clip when indexing into `grid`.
#   gi0c = tf.clip_by_value(gi0, 0, grid.shape[0] - 1)
#   gj0c = tf.clip_by_value(gj0, 0, grid.shape[1] - 1)
#   gk0c = tf.clip_by_value(gk0, 0, grid.shape[2] - 1)

#   gi1c = tf.clip_by_value(gi0 + 1, 0, grid.shape[0] - 1)
#   gj1c = tf.clip_by_value(gj0 + 1, 0, grid.shape[1] - 1)
#   gk1c = tf.clip_by_value(gk0 + 1, 0, grid.shape[2] - 1)
  gi0c = tf.cast(tf.clip_by_value(gi0, 0, grid.shape[0] - 1), dtype=tf.int32)
  gj0c = tf.cast(tf.clip_by_value(gj0, 0, grid.shape[1] - 1), dtype=tf.int32)
  gk0c = tf.cast(tf.clip_by_value(gk0, 0, grid.shape[2] - 1), dtype=tf.int32)

  gi1c = tf.cast(tf.clip_by_value(gi0 + 1, 0, grid.shape[0] - 1), dtype=tf.int32)
  gj1c = tf.cast(tf.clip_by_value(gj0 + 1, 0, grid.shape[1] - 1), dtype=tf.int32)
  gk1c = tf.cast(tf.clip_by_value(gk0 + 1, 0, grid.shape[2] - 1), dtype=tf.int32)
  print(grid.shape)
  print(w_000.shape)

  #        ijk: 0 means floor, 1 means ceil.
  # grid_val_000 = tf.gather(grid, [gi0c, gj0c, gk0c], axis=0)
  # grid_val_001 = tf.gather(grid, [gi0c, gj0c, gk1c], axis=0)
  # grid_val_010 = tf.gather(grid, [gi0c, gj1c, gk0c], axis=0)
  # grid_val_011 = tf.gather(grid, [gi0c, gj1c, gk1c], axis=0)
  # grid_val_100 = tf.gather(grid, [gi1c, gj0c, gk0c], axis=0)
  # grid_val_101 = tf.gather(grid, [gi1c, gj0c, gk1c], axis=0)
  # grid_val_110 = tf.gather(grid, [gi1c, gj1c, gk0c], axis=0)
  # grid_val_111 = tf.gather(grid, [gi1c, gj1c, gk1c], axis=0)
  grid_val_000 = grid[gi0c, gj0c, gk0c, : ,: ]
  grid_val_001 = grid[gi0c, gj0c, gk1c, :]
  grid_val_010 = grid[gi0c, gj1c, gk0c, :]
  grid_val_011 = grid[gi0c, gj1c, gk1c, :]
  grid_val_100 = grid[gi1c, gj0c, gk0c, :]
  grid_val_101 = grid[gi1c, gj0c, gk1c, :]
  grid_val_110 = grid[gi1c, gj1c, gk0c, :]
  grid_val_111 = grid[gi1c, gj1c, gk1c, :] 

  # Append a singleton "channels" dimension.
  w_000, w_001, w_010, w_011, w_100, w_101, w_110, w_111 = tf.expand_dims(
      w_000, axis=-1), tf.expand_dims(w_001, axis=-1), tf.expand_dims(w_010, axis=-1), \
      tf.expand_dims(w_011, axis=-1), tf.expand_dims(w_100, axis=-1), tf.expand_dims(w_101, axis=-1), \
      tf.expand_dims(w_110, axis=-1), tf.expand_dims(w_111, axis=-1)


  return (tf.matmul(w_000, grid_val_000) +
        tf.matmul(w_001, grid_val_001) +
        tf.matmul(w_010, grid_val_010) +
        tf.matmul(w_011, grid_val_011) +
        tf.matmul(w_100, grid_val_100) +
        tf.matmul(w_101, grid_val_101) +
        tf.matmul(w_110, grid_val_110) +
        tf.matmul(w_111, grid_val_111))


def apply(sliced, input_image, has_affine_term=True, name=None):
    """Applies a sliced affined model to the input image.

    Args:
      sliced: (Tensor) [batch_size, h, w, n_output, n_input+1] affine coefficients
      input_image: (Tensor) [batch_size, h, w, n_input] input data onto which to
        apply the affine transform.
      name: (string) name for the operation.
    Returns:
      ret: (Tensor) [batch_size, h, w, n_output] the transformed data.
    Raises:
      ValueError: if the input is not properly dimensioned.
      ValueError: if the affine model parameter dimensions do not match the input.
    """
    
    if len(input_image.get_shape().as_list()) != 4:
        raise ValueError('input image should have dims [b,h,w,n_in].')
    in_shape = input_image.get_shape().as_list()
    sliced_shape = sliced.get_shape().as_list()
    if (in_shape[:-1] != sliced_shape[:-2]):
        raise ValueError('input image and affine coefficients'
                            ' dimensions do not match: {} and {}'.format(
                                in_shape, sliced_shape))
    _, _, _, n_out, n_in = sliced.get_shape().as_list()
    if has_affine_term:
        n_in -= 1

    scale = sliced[:, :, :, :, :n_in]

    if has_affine_term:
        offset = sliced[:, :, :, :, n_in]

    # foreach chanel:
    #     a*x[0] + b*x[1] + c*x[2] + d = (h,w)
    #   res [ch1]   (x [ch1]   [aff11]           )   (x [ch1]   [aff21]           )
    #       [ch2] = (  [ch2] * [aff12] + [aff14] ) + (  [ch2] * [aff22] + [aff24] ) + same for aff3[1-4]
    #       [ch3]   (  [ch3]   [aff13]           )   (  [ch3]   [aff23]           )
    #

    out_channels = []
    for chan in range(n_out):
        ret = scale[:, :, :, chan, 0] * input_image[:, :, :, 0]
        for chan_i in range(1, n_in):
            ret += scale[:, :, :, chan, chan_i] * input_image[:, :, :, chan_i]
        if has_affine_term:
            ret += offset[:, :, :, chan]
        ret = tf.expand_dims(ret, 3)
        out_channels.append(ret)

    ret = tf.concat(out_channels, 3)

    return ret
# pylint: enable=redefined-builtin