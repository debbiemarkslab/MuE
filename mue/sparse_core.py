import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability import distributions as tfpd
from tensorflow_probability.python.internal import reparameterization
from edward2.tensorflow.generated_random_variables import make_random_variable

from mue import sparse_util

import pdb


def mg2k(m, g):
    """Reindex from (m, g) to state k."""
    return 2*m + 1 - g


def flati(m, g):
    """Reindex from (m, g) to flattened position."""
    return 2*m + g


def get_eps(dtype):
    """Get small float value such that log(eps) is still non infinite."""
    if dtype == tf.float64:
        eps = tf.convert_to_tensor(1e-307, dtype=dtype)
    elif dtype == tf.float32:
        eps = tf.convert_to_tensor(1e-45, dtype=dtype)
    return eps


def make_transfer(M, Lsparse, dtype=tf.float32):
    """Transfer matrices, from r, u, s and c to HMM trans. and emit. matrix."""

    # Transfer matrix for initial transition vector.
    K = 2*(M+1)
    r_transf_0_indx = []
    u_transf_0_indx = []
    null_transf_0 = np.zeros((K,))
    for mp in range(M+1):
        for gp in range(2):
            kp = mg2k(mp, gp)
            if (0 == mp and gp == 0):
                r_transf_0_indx.append([flati(0, 0), kp])
                if mp < M:
                    u_transf_0_indx.append([flati(0, 0), kp])

            elif (0 < mp and mp <= M and gp == 0 and
                  mp <= Lsparse):
                r_transf_0_indx.append([flati(0, 0), kp])
                u_transf_0_indx.append([flati(0, 1), kp])
                for mpp in range(1, mp):
                    r_transf_0_indx.append([flati(mpp, 0), kp])
                    u_transf_0_indx.append([flati(mpp, 1), kp])
                r_transf_0_indx.append([flati(mp, 0), kp])
                if mp < Lsparse and mp < M:
                    u_transf_0_indx.append([flati(mp, 0), kp])

            elif (0 == mp and gp == 1):
                r_transf_0_indx.append([flati(0, 1), kp])

            elif (0 < mp and mp <= M and gp == 1 and
                  mp <= Lsparse):
                r_transf_0_indx.append([flati(0, 0), kp])
                u_transf_0_indx.append([flati(0, 1), kp])
                for mpp in range(1, mp):
                    r_transf_0_indx.append([flati(mpp, 0), kp])
                    u_transf_0_indx.append([flati(mpp, 1), kp])
                r_transf_0_indx.append([flati(mp, 1), kp])

            else:
                null_transf_0[kp] = 1.

    r_transf_0 = tf.sparse.SparseTensor(
            indices=r_transf_0_indx,
            values=tf.ones(len(r_transf_0_indx), dtype=dtype),
            dense_shape=[K, K])
    u_transf_0 = tf.sparse.SparseTensor(
            indices=u_transf_0_indx,
            values=tf.ones(len(u_transf_0_indx), dtype=dtype),
            dense_shape=[K, K])
    null_transf_0 = tf.convert_to_tensor(null_transf_0, dtype=dtype)

    # Transfer matrix for transition matrix.
    a_indx = []
    r_transf_indx = []
    u_transf_indx = []
    i = 0
    for m in range(M+1):
        for g in range(2):
            for mp in range(M+1):
                for gp in range(2):

                    k, kp = mg2k(m, g), mg2k(mp, gp)
                    a_indx.append([k, kp])

                    if (m+1-g == mp and gp == 0):
                        r_transf_indx.append([flati(m+1-g, 0), i])
                        if mp < M:
                            u_transf_indx.append([flati(m+1-g, 0), i])

                    elif (m+1-g < mp and mp <= M and gp == 0 and
                          mp-m-1+g <= Lsparse):
                        r_transf_indx.append([flati(m+1-g, 0), i])
                        u_transf_indx.append([flati(m+1-g, 1), i])
                        for mpp in range(m+2-g, mp):
                            r_transf_indx.append([flati(mpp, 0), i])
                            u_transf_indx.append([flati(mpp, 1), i])
                        r_transf_indx.append([flati(mp, 0), i])
                        if mp-m-1+g < Lsparse and mp < M:
                            u_transf_indx.append([flati(mp, 0), i])

                    elif (m+1-g == mp and gp == 1):
                        r_transf_indx.append([flati(m+1-g, 1), i])

                    elif (m+1-g < mp and mp <= M and gp == 1 and
                          mp-m-1+g <= Lsparse):
                        r_transf_indx.append([flati(m+1-g, 0), i])
                        u_transf_indx.append([flati(m+1-g, 1), i])
                        for mpp in range(m+2-g, mp):
                            r_transf_indx.append([flati(mpp, 0), i])
                            u_transf_indx.append([flati(mpp, 1), i])
                        r_transf_indx.append([flati(mp, 1), i])
                    else:
                        i -= 1
                        a_indx.pop()
                    i += 1
    tot_entries = i

    r_transf = tf.sparse.SparseTensor(
            indices=r_transf_indx,
            values=tf.ones(len(r_transf_indx), dtype=dtype),
            dense_shape=[K, tot_entries])
    u_transf = tf.sparse.SparseTensor(
            indices=u_transf_indx,
            values=tf.ones(len(u_transf_indx), dtype=dtype),
            dense_shape=[K, tot_entries])

    # Transfer matrix for emission matrix.
    vx_transf_indx = []
    vc_transf_indx = []
    for m in range(M+1):
        for g in range(2):
            k = mg2k(m, g)
            if g == 0:
                vx_transf_indx.append([m, k])
            elif g == 1:
                vc_transf_indx.append([m, k])

    vx_transf = tf.sparse.SparseTensor(
            indices=vx_transf_indx,
            values=tf.ones(len(vx_transf_indx), dtype=dtype),
            dense_shape=[M+1, K])
    vc_transf = tf.sparse.SparseTensor(
            indices=vc_transf_indx,
            values=tf.ones(len(vc_transf_indx), dtype=dtype),
            dense_shape=[M+1, K])

    return {'rt0': r_transf_0, 'ut0': u_transf_0, 'nt0': null_transf_0,
            'rt': r_transf, 'ut': u_transf, 'ai': a_indx,
            'vxt': vx_transf, 'vct': vc_transf}


def make_sparse_hmm_params(vxln, vcln, uln, rln, lln, transfer_mats,
                           dtype=tf.float32):
    """Assemble the HMM parameters based on the s, u and r parameters."""
    M = rln.shape[0]-1
    K = 2*(M+1)

    # Initial transition.
    a0 = (tf.sparse.sparse_dense_matmul(
                transfer_mats['rt0'], tf.reshape(rln, [K])[:, None],
                adjoint_a=True) +
          tf.sparse.sparse_dense_matmul(
                transfer_mats['ut0'], tf.reshape(uln, [K])[:, None],
                adjoint_a=True)
          )[:, 0] + (-1/get_eps(rln.dtype))*transfer_mats['nt0']

    # Sparse transition matrix.
    a_vals = (tf.sparse.sparse_dense_matmul(
                transfer_mats['rt'], tf.reshape(rln, [K])[:, None],
                adjoint_a=True) +
              tf.sparse.sparse_dense_matmul(
                transfer_mats['ut'], tf.reshape(uln, [K])[:, None],
                adjoint_a=True))[:, 0]
    a = tf.sparse.SparseTensor(
            indices=transfer_mats['ai'], values=a_vals, dense_shape=[K, K])

    # Emission matrix.
    seqln = (tf.sparse.sparse_dense_matmul(
                 transfer_mats['vxt'], vxln, adjoint_a=True) +
             tf.sparse.sparse_dense_matmul(
                 transfer_mats['vct'], vcln, adjoint_a=True))
    if lln is not None:
        # Substitution matrix.
        e = tf.reduce_logsumexp(
                seqln[:, :, None] + lln[None, :, :], axis=1)
    else:
        e = seqln

    return a0, a, e


@tf.function
def _sparse_matrix_logsumexp_dot(atilde, logabar, logv):
    """Multiply the sparse matrix (atilde*exp(logabar)[None, :]).T by logv."""
    logvbar = tf.stop_gradient(tf.reduce_max(logv))
    vtilde = tf.exp(logv - logvbar)
    matmul = tf.sparse.sparse_dense_matmul(atilde, vtilde[:, None],
                                           adjoint_a=True)[:, 0]
    eps = get_eps(logv.dtype)
    return (tf.math.log(matmul + eps) + logvbar + logabar)


def _sparse_reduce_max(aln, axis=0):
    """A version of tf.sparse.reduce_max that allows gradients."""
    vals = []
    for j in range(aln.dense_shape[1-axis]):
        bool_find = aln.indices[:, 1-axis] == j
        if tf.reduce_any(bool_find):
            vals.append(tf.reduce_max(
                tf.where(bool_find, aln.values, aln.dtype.min)))
        else:
            vals.append(tf.convert_to_tensor(0., dtype=aln.dtype))
    return tf.stack(vals)


def _sparse_argmax(aln, axis=0):
    """A version of argmax for sparse matrices."""
    vals = []
    for j in range(aln.dense_shape[1-axis]):
        bool_find = aln.indices[:, 1-axis] == j
        if tf.reduce_any(bool_find):
            vals.append(aln.indices[tf.argmax(
                tf.where(bool_find, aln.values, aln.dtype.min)), axis])
        else:
            vals.append(tf.convert_to_tensor(0, dtype=tf.int64))
    return tf.stack(vals)


def _decompose_transition_mat(a):
    """Break up transition matrix following logsumexp trick."""
    # Row maximums.
    logabar = tf.stop_gradient(tf.sparse.reduce_max(a, axis=0))
    # Divide each row by maximums.
    negabar_broadcast = tf.sparse.SparseTensor(
            a.indices, tf.gather(-logabar, a.indices[:, 1]), a.dense_shape)
    # Remaining matrix elements exponentiated.
    atilde = sparse_util.map_values(tf.exp,
                                    tf.sparse.add(a, negabar_broadcast))

    return atilde, logabar


class tfpSparseHiddenMarkovModel(tfpd.distribution.Distribution):
    """Hidden Markov model distribution with sparse transition.
    Note that as a consequence of the limitations on
    tf.sparse.sparse_dense_matmul, the distribution can't handle batching."""

    def __init__(self,
                 initial_distribution,
                 transition_distribution,
                 observation_distribution,
                 num_steps,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='SparseHiddenMarkovModel'):
        """Initialize hidden Markov model."""

        parameters = dict(locals())
        with tf.name_scope(name) as name:

            self._initial_distribution = initial_distribution
            self._transition_distribution = transition_distribution
            self._observation_distribution = observation_distribution
            self._num_steps = num_steps

            super(tfpSparseHiddenMarkovModel, self).__init__(
              dtype=self._observation_distribution.dtype,
              reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
              validate_args=validate_args,
              allow_nan_stats=allow_nan_stats,
              parameters=parameters,
              name=name)

    def _sample_n(self, n, seed=None, dummy=True):
        """Sample n sequences from the distribution."""

        if dummy:
            # Sampling from an HMM is expensive, but Edward2 always initializes
            # distributions with a sample, even when the output is observed.
            # We therefore default to a "dummy" sample of just zeros to avoid
            # this cost.
            return tf.zeros((n, self._num_steps,
                             self._observation_distribution.shape[1]),
                            dtype=tf.int32)
        else:

            def sample_initial_state():
                return tfpd.Categorical(
                    logits=self._initial_distribution).sample()

            def sample_next_state(state):
                transit_row = tf.sparse.slice(
                        self._transition_distribution,
                        [state, 0],
                        [state+1,
                         self._transition_distribution.dense_shape[1]])
                transit = tf.sparse.to_dense(sparse_util.map_values(
                                tf.exp, transit_row))[0, :]
                return tfpd.Categorical(probs=transit).sample()

            def sample_observation(state):
                return tfpd.OneHotCategorical(
                            logits=self._observation_distribution[state, :]
                            ).sample()

            samples = []
            # Note -- can't batch with sparse_dense_matmul.
            for i in range(n):
                sample = []
                state = sample_initial_state()
                sample.append(sample_observation(state))
                for j in range(1, self._num_steps):
                    state = sample_next_state(state)
                    sample.append(sample_observation(state))
                samples.append(tf.stack(sample, axis=0))
                print(samples[-1])
            return tf.stack(samples, axis=0)

    def _log_prob(self, value):
        """Log probability of a sequence."""
        # Emission matrix for given observations.
        emit_lp = tf.matmul(self._observation_distribution, value,
                            transpose_b=True)

        # Break up transition matrix following logsumexp trick.
        atilde, logabar = _decompose_transition_mat(
                                self._transition_distribution)

        # Initial vector.
        logv = self._initial_distribution + emit_lp[:, 0]

        # Iterate forward algorithm, following logsumexp trick.
        for j in range(1, self._num_steps):
            logv = (emit_lp[:, j] +
                    _sparse_matrix_logsumexp_dot(atilde, logabar, logv))

        log_p = tf.reduce_logsumexp(logv)

        return log_p

    def _mean(self):
        """Expected value at each position of the generated sequence."""
        # Break up transition matrix following logsumexp trick.
        atilde, logabar = _decompose_transition_mat(
                                self._transition_distribution)

        # Initial vector.
        logv = self._initial_distribution

        log_mean = [tf.reduce_logsumexp(
                logv[:, None] + self._observation_distribution, axis=0)]
        for j in range(1, self._num_steps):
            # Iterate forward algorithm, following logsumexp trick.
            logv = _sparse_matrix_logsumexp_dot(atilde, logabar, logv)
            # Compute log mean at current position.
            log_mean.append(tf.reduce_logsumexp(
                    logv[:, None] + self._observation_distribution, axis=0))

        # Exponentiate to get mean.
        return tf.exp(tf.stack(log_mean, axis=0))

    def posterior_mode(self, observations, mask=None, name='posterior_mode'):
        """Compute posterior mode with Viterbi."""
        # Observation probabilities.
        emit_lp = tf.matmul(self._observation_distribution, observations,
                            transpose_b=True)

        # Initialize joint probability.
        logv = self._initial_distribution + emit_lp[:, 0]

        # Forward pass.
        a = self._transition_distribution
        trellis = []
        for j in range(1, self._num_steps):
            joint_ln = tf.SparseTensor(
                        self._transition_distribution.indices,
                        self._transition_distribution.values
                        + tf.gather(logv, a.indices[:, 0])
                        + tf.gather(emit_lp[:, j], a.indices[:, 1]),
                        self._transition_distribution.dense_shape)
            logv = tf.sparse.reduce_max(joint_ln, axis=0)
            trellis.append(_sparse_argmax(joint_ln, axis=0))

        # Retrace.
        states = [tf.argmax(logv)]
        for j in range(self._num_steps-2, -1, -1):
            states.append(trellis[j][states[-1]])

        return tf.stack(states[::-1])


"""Make Edward distribution."""
SparseHiddenMarkovModel = make_random_variable(tfpSparseHiddenMarkovModel)


def encode(x, uln0, rln0, lln0,
           latent_length, latent_alphabet_size, alphabet_size,
           padded_data_length, transfer_mats, dtype=tf.float32):
    """First layer of encoder, using the MuE mean."""
    eps = get_eps(dtype)

    # Set initial sequence (replace inf with large number)
    vxln = tf.maximum(tf.math.log(x), tf.math.log(eps))

    # Set insert biases to uniform distribution.
    vcln = -np.log(alphabet_size)*tf.ones_like(vxln)

    # Set deletion and insertion parameters.
    uln = tf.ones((padded_data_length, 2), dtype=dtype) * (
            uln0 - tf.reduce_logsumexp(uln0))[None, :]
    rln = tf.ones((padded_data_length, 2), dtype=dtype) * (
            rln0 - tf.reduce_logsumexp(rln0))[None, :]
    lln = lln0 - tf.reduce_logsumexp(lln0, axis=1, keepdims=True)

    # Build HiddenMarkovModel, with one-hot encoded output.
    a0_enc, a_enc, e_enc = make_sparse_hmm_params(
            vxln, vcln, uln, rln, lln, transfer_mats, dtype=dtype)
    hmm_enc = tfpSparseHiddenMarkovModel(
            a0_enc, a_enc, e_enc, latent_length)

    # Compute mean.
    return hmm_enc._mean()


def get_most_common_tuple(lst):
    """Get the mode of a list of tuples."""

    return max(set(lst), key=lst.count)
