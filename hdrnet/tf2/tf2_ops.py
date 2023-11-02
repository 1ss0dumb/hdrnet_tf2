import tensorflow as tf

def lerp_weight(x, xs):
    """Linear interpolation weight from a sample at x to xs.
    Returns the linear interpolation weight of a "query point" at coordinate `x`
    with respect to a "sample" at coordinate `xs`.
    The integer coordinates `x` are at pixel centers.
    The floating point coordinates `xs` are at pixel edges.
    (OpenGL convention).
    Args:
        x: "Query" point position.
        xs: "Sample" position.
    Returns:
        - 1 when x = xs.
        - 0 when |x - xs| > 1.
    """
    dx = x - xs
    abs_dx = tf.abs(dx)
    return tf.maximum(1.0 - abs_dx, 0.0)


def smoothed_abs(x, eps):
    """A smoothed version of |x| with improved numerical stability."""
    return tf.sqrt(tf.multiply(x, x) + eps)


def smoothed_lerp_weight(x, xs):
    """Smoothed version of `LerpWeight` with gradients more suitable for backprop.
    Let f(x, xs) = LerpWeight(x, xs)
               = max(1 - |x - xs|, 0)
               = max(1 - |dx|, 0)
    f is not smooth when:
    - |dx| is close to 0. We smooth this by replacing |dx| with
      SmoothedAbs(dx, eps) = sqrt(dx * dx + eps), which has derivative
      dx / sqrt(dx * dx + eps).
    - |dx| = 1. When smoothed, this happens when dx = sqrt(1 - eps). Like ReLU,
      We just ignore this (in the implementation below, when the floats are
      exactly equal, we choose the SmoothedAbsGrad path since it is more useful
      than returning a 0 gradient).
    Args:
        x: "Query" point position.
        xs: "Sample" position.
        eps: a small number.
    Returns:
        max(1 - |dx|, 0) where |dx| is smoothed_abs(dx).
    """
    eps = tf.constant(1e-8, dtype=tf.float32)
    dx = x - xs
    abs_dx = smoothed_abs(dx, eps)
    return tf.maximum(1.0 - abs_dx, 0.0)

def bilateral_slice(grid, guide):
    """Slices a bilateral grid using the a guide image.
    Args:
        grid: The bilateral grid with shape (gh, gw, gd, gc).
        guide: A guide image with shape (h, w). Values must be in the range [0, 1].
    Returns:
        sliced: An image with shape (h, w, gc), computed by trilinearly
        interpolating for each grid channel c the grid at 3D position
        [(i + 0.5) * gh / h,
        (j + 0.5) * gw / w,
        guide(i, j) * gd]
    """
    ii, jj = tf.meshgrid(
        tf.range(tf.shape(guide)[0]), tf.range(tf.shape(guide)[1]), indexing='ij')

    scale_i = tf.shape(grid)[0] / tf.shape(guide)[0]
    scale_j = tf.shape(grid)[1] / tf.shape(guide)[1]

    ii = tf.cast(ii, dtype=tf.float32)
    jj = tf.cast(ii, dtype=tf.float32)

    gif = (ii + 0.5) * scale_i
    gjf = (jj + 0.5) * scale_j
    gkf = tf.multiply(guide,tf.shape(grid)[2])

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

    # Clip when indexing into `grid`.
    gi0c = tf.clip_by_value(tf.cast(gi0, tf.int32), 0, tf.shape(grid)[0] - 1)
    gj0c = tf.clip_by_value(tf.cast(gj0, tf.int32), 0, tf.shape(grid)[1] - 1)
    gk0c = tf.clip_by_value(tf.cast(gk0, tf.int32), 0, tf.shape(grid)[2] - 1)

    gi1c = tf.clip_by_value(tf.cast(gi1, tf.int32), 0, tf.shape(grid)[0] - 1)
    gj1c = tf.clip_by_value(tf.cast(gj1, tf.int32), 0, tf.shape(grid)[1] - 1)
    gk1c = tf.clip_by_value(tf.cast(gk1, tf.int32), 0, tf.shape(grid)[2] - 1)

    # ijk: 0 means floor, 1 means ceil.
    grid_val_000 = tf.gather(grid, [gi0c, gj0c, gk0c], axis=[0, 1, 2])
    grid_val_001 = tf.gather(grid, [gi0c, gj0c, gk1c], axis=[0, 1, 2])
    grid_val_010 = tf.gather(grid, [gi0c, gj1c, gk0c], axis=[0, 1, 2])
    grid_val_011 = tf.gather(grid, [gi0c, gj1c, gk1c], axis=[0, 1, 2])
    grid_val_100 = tf.gather(grid, [gi1c, gj0c, gk0c], axis=[0, 1, 2])
    grid_val_101 = tf.gather(grid, [gi1c, gj0c, gk1c], axis=[0, 1, 2])
    grid_val_110 = tf.gather(grid, [gi1c, gj1c, gk0c], axis=[0, 1, 2])
    grid_val_111 = tf.gather(grid, [gi1c, gj1c, gk1c], axis=[0, 1, 2])

    # Append a singleton "channels" dimension.
    w_000, w_001, w_010, w_011, w_100, w_101, w_110, w_111 = tf.expand_dims(
        w_000, axis=-1), tf.expand_dims(w_001, axis=-1), tf.expand_dims(w_010, axis=-1), tf.expand_dims(w_011, axis=-1), \
        tf.expand_dims(w_100, axis=-1), tf.expand_dims(w_101, axis=-1), tf.expand_dims(w_110, axis=-1), tf.expand_dims(w_111, axis=-1)

    return tf.reduce_sum(
        tf.multiply(w_000, grid_val_000) +
        tf.multiply(w_001, grid_val_001) +
        tf.multiply(w_010, grid_val_010) +
        tf.multiply(w_011, grid_val_011) +
        tf.multiply(w_100, grid_val_100) +
        tf.multiply(w_101, grid_val_101) +
        tf.multiply(w_110, grid_val_110) +
        tf.multiply(w_111, grid_val_111), axis=-2)

