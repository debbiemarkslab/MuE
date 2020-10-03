
"""map_values, copied from tensorflow_nightly, to avoid having to to install
tensorflow_nightly.
copied from: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/sparse_ops.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import check_ops
from tensorflow.python.ops.gen_sparse_ops import *
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


def _replace_sparse_with_values(value, sparse_list):
    """Replace `SparseTensor`s with their values in `value`
  Each `SparseTensor` in `value` is replaced by its `values` tensor, and
  collects all `SparseTensor`s in `sparse_list`.
  Args:
    value: A structure of `Tensor`s and `SparseTensor`s
    sparse_list: A list. Output parameter that collects all `SparseTensor`s in
      `value`.
  Returns:
    `value` with each SparseTensor replaced by its `.value` attribute.
  """
    flat_vals = nest.flatten(value, expand_composites=False)
    new_vals = []
    for v in flat_vals:
        if isinstance(v, sparse_tensor.SparseTensor):
            sparse_list.append(v)
            new_vals.append(v.values)
        else:
            new_vals.append(v)
    return nest.pack_sequence_as(value, new_vals, expand_composites=False)


def _assert_sparse_compatible(sparse_tensors):
    """Check that all of `sparse_tensors` have same `indices` and `dense_shape`.
  Args:
    sparse_tensors: A list of sparse tensors.
  Returns:
    An op to be used as a control dependency.
  """
    checks = []
    first = sparse_tensors[0]
    for t in sparse_tensors[1:]:
        checks.append(
            check_ops.assert_equal(
             first.dense_shape, t.dense_shape, message="Mismatched shapes!"))
        checks.append(
            check_ops.assert_equal(
             first.indices, t.indices, message="Mismatched indices!"))
    return checks


@tf_export("sparse.map_values", v1=[])
@dispatch.add_dispatch_support
def map_values(op, *args, **kwargs):
    """Applies `op` to the `.values` tensor of one or more `SparseTensor`s.
  Replaces any `SparseTensor` in `args` or `kwargs` with its `values`
  tensor (which contains the non-default values for the SparseTensor),
  and then calls `op`.  Returns a `SparseTensor` that is constructed
  from the input `SparseTensor`s' `indices`, `dense_shape`, and the
  value returned by the `op`.
  If the input arguments contain multiple `SparseTensor`s, then they must have
  equal `indices` and dense shapes.
  Examples:
  >>> s = tf.sparse.from_dense([[1, 2, 0],
  ...                           [0, 4, 0],
  ...                           [1, 0, 0]])
  >>> tf.sparse.to_dense(tf.sparse.map_values(tf.ones_like, s)).numpy()
  array([[1, 1, 0],
         [0, 1, 0],
         [1, 0, 0]], dtype=int32)
  >>> tf.sparse.to_dense(tf.sparse.map_values(tf.multiply, s, s)).numpy()
  array([[ 1,  4,  0],
         [ 0, 16,  0],
         [ 1,  0,  0]], dtype=int32)
  >>> tf.sparse.to_dense(tf.sparse.map_values(tf.add, s, 5)).numpy()
  array([[6, 7, 0],
         [0, 9, 0],
         [6, 0, 0]], dtype=int32)
  Note: even though `tf.add(0, 5) != 0`, implicit zeros
  will remain unchanged. However, if the sparse tensor contains any explict
  zeros, these will be affected by the mapping!
  Args:
    op: The operation that should be applied to the SparseTensor `values`. `op`
      is typically an element-wise operation (such as math_ops.add), but any
      operation that preserves the shape can be used.
    *args: Arguments for `op`.
    **kwargs: Keyword arguments for `op`.
  Returns:
    A `SparseTensor` whose `indices` and `dense_shape` matches the `indices`
    and `dense_shape` of all input `SparseTensor`s.
  Raises:
    ValueError: If args contains no `SparseTensor`, or if the `indices`
      or `dense_shape`s of the input `SparseTensor`s are not equal.
  """
    sparse_list = []
    inner_args = _replace_sparse_with_values(args, sparse_list)
    inner_kwargs = _replace_sparse_with_values(kwargs, sparse_list)
    if not sparse_list:
        raise ValueError("No SparseTensor in argument list of map_values")

    with ops.control_dependencies(_assert_sparse_compatible(sparse_list)):
        # Delegate to op, and then compose the result from the transformed values
        # and the known indices/dense shape. Since we ensure that indices and shape
        # are identical, we can just use the first one.
        return sparse_tensor.SparseTensor(sparse_list[0].indices,
                                          op(*inner_args, **inner_kwargs),
                                          sparse_list[0].dense_shape)
