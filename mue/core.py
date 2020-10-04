import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability import distributions as tfpd


def mg2k(m, g):

    return 2*m + 1 - g


def make_transfer(M, dtype=tf.float32):
    """Transfer matrices, from r, u, s and c to HMM trans. and emit. matrix."""

    K = 2*(M+1)

    # r and u indices: m in {0, ..., M} and j in {0, 1, 2}, and (1-r), r
    r_transf_0 = np.zeros((M+1, 3, 2, K))
    u_transf_0 = np.zeros((M+1, 3, 2, K))
    null_transf_0 = np.zeros((K,))
    m, g = -1, 0
    for mp in range(M+1):
        for gp in range(2):
            kp = mg2k(mp, gp)
            if m + 1 - g == mp and gp == 0:
                r_transf_0[m+1-g, g, 0, kp] = 1
                u_transf_0[m+1-g, g, 0, kp] = 1

            elif m + 1 - g < mp and mp <= M and gp == 0:
                r_transf_0[m+1-g, g, 0, kp] = 1
                u_transf_0[m+1-g, g, 1, kp] = 1
                for mpp in range(m+2-g, mp):
                    r_transf_0[mpp, 2, 0, kp] = 1
                    u_transf_0[mpp, 2, 1, kp] = 1
                r_transf_0[mp, 2, 0, kp] = 1
                u_transf_0[mp, 2, 0, kp] = 1

            elif m + 1 - g == mp and gp == 1:
                r_transf_0[m+1-g, g, 1, kp] = 1

            elif m + 1 - g < mp and mp <= M and gp == 1:
                r_transf_0[m+1-g, g, 0, kp] = 1
                u_transf_0[m+1-g, g, 1, kp] = 1
                for mpp in range(m+2-g, mp):
                    r_transf_0[mpp, 2, 0, kp] = 1
                    u_transf_0[mpp, 2, 1, kp] = 1
                r_transf_0[mp, 2, 1, kp] = 1

            else:
                null_transf_0[kp] = 1
    u_transf_0[-1, :, :, :] = 0.

    r_transf = np.zeros((M+1, 3, 2, K, K))
    u_transf = np.zeros((M+1, 3, 2, K, K))
    null_transf = np.zeros((K, K))
    for m in range(M+1):
        for g in range(2):
            for mp in range(M+1):
                for gp in range(2):
                    k, kp = mg2k(m, g), mg2k(mp, gp)
                    if m + 1 - g == mp and gp == 0:
                        r_transf[m+1-g, g, 0, k, kp] = 1
                        u_transf[m+1-g, g, 0, k, kp] = 1

                    elif m + 1 - g < mp and mp <= M and gp == 0:
                        r_transf[m+1-g, g, 0, k, kp] = 1
                        u_transf[m+1-g, g, 1, k, kp] = 1
                        for mpp in range(m+2-g, mp):
                            r_transf[mpp, 2, 0, k, kp] = 1
                            u_transf[mpp, 2, 1, k, kp] = 1
                        r_transf[mp, 2, 0, k, kp] = 1
                        u_transf[mp, 2, 0, k, kp] = 1

                    elif m + 1 - g == mp and gp == 1:
                        r_transf[m+1-g, g, 1, k, kp] = 1

                    elif m + 1 - g < mp and mp <= M and gp == 1:
                        r_transf[m+1-g, g, 0, k, kp] = 1
                        u_transf[m+1-g, g, 1, k, kp] = 1
                        for mpp in range(m+2-g, mp):
                            r_transf[mpp, 2, 0, k, kp] = 1
                            u_transf[mpp, 2, 1, k, kp] = 1
                        r_transf[mp, 2, 1, k, kp] = 1

                    elif not (m == M and mp == M and g == 0 and gp == 0):
                        null_transf[k, kp] = 1
    u_transf[-1, :, :, :, :] = 0.

    vx_transf = np.zeros((M+1, K))
    vc_transf = np.zeros((M+1, K))
    for m in range(M+1):
        for g in range(2):
            k = mg2k(m, g)
            if g == 0:
                vx_transf[m, k] = 1
            elif g == 1:
                vc_transf[m, k] = 1

    return {'nt0': tf.convert_to_tensor(null_transf_0, dtype=dtype),
            'rt0': tf.convert_to_tensor(r_transf_0, dtype=dtype),
            'ut0': tf.convert_to_tensor(u_transf_0, dtype=dtype),
            'rt': tf.convert_to_tensor(r_transf, dtype=dtype),
            'ut': tf.convert_to_tensor(u_transf, dtype=dtype),
            'nt': tf.convert_to_tensor(null_transf, dtype=dtype),
            'vxt': tf.convert_to_tensor(vx_transf, dtype=dtype),
            'vct': tf.convert_to_tensor(vc_transf, dtype=dtype)}


def make_hmm_params(vxln, vcln, uln, rln, lln, transfer_mats, eps=1e-32,
                    dtype=tf.float32):
    """Assemble the HMM parameters based on the s, u and r parameters."""

    # Assemble transition matrices.
    ulnfull = uln[:, None, :] * tf.ones((1, 3, 1), dtype=dtype)
    rlnfull = rln[:, None, :] * tf.ones((1, 3, 1), dtype=dtype)
    a0 = (tf.einsum('ijk,ijkl->l', ulnfull, transfer_mats['ut0']) +
          tf.einsum('ijk,ijkl->l', rlnfull, transfer_mats['rt0']) +
          (-1/eps)*transfer_mats['nt0'])
    a = (tf.einsum('ijk,ijklf->lf', ulnfull, transfer_mats['ut']) +
         tf.einsum('ijk,ijklf->lf', rlnfull, transfer_mats['rt']) +
         (-1/eps)*transfer_mats['nt'])

    # Assemble emission matrix.
    seqln = (tf.einsum('ij,ik->kj', vxln, transfer_mats['vxt']) +
             tf.einsum('ij,ik->kj', vcln, transfer_mats['vct']))
    if lln is not None:
        # Substitution matrix.
        e = tf.reduce_logsumexp(
                seqln[:, :, None] + lln[None, :, :], axis=1)
    else:
        e = seqln

    return a0, a, e


@tf.function
def forward(a0, a, e, x, num):
    """Markov chain forward algorithm, with tf graph construction."""
    emit_lp = tf.matmul(e, x, transpose_b=True)
    f = a0 + emit_lp[:, 0]
    for j in range(1, num):
        f = tf.reduce_logsumexp(f[:, None] + a, axis=0) + emit_lp[:, j]

    return tf.reduce_logsumexp(f)


@tf.function
def forward_mean(a0, a, e, num):
    """Mean of HMM (by individual location)."""
    f = []
    f.append(a0[None, :])
    for j in range(1, num):
        f.append(tf.reduce_logsumexp(f[-1][0, :, None] + a, axis=0,
                                     keepdims=True))
    fm = tf.concat(f, axis=0)
    hmean = tf.reduce_logsumexp(fm[:, :, None] + e[None, :, :], axis=1)

    return tf.exp(hmean)


def hmm_log_prob(distr, x, xlen):
    """Fast log prob. calculation."""
    a0 = distr.initial_distribution.logits
    a = distr.transition_distribution.logits
    e = distr.observation_distribution.logits
    return forward(a0, a, e, x, xlen)


def hmm_mean(distr, xlen):
    """Fast mean calculation."""
    a0 = distr.initial_distribution.logits
    a = distr.transition_distribution.logits
    e = distr.observation_distribution.logits
    return forward_mean(a0, a, e, xlen)


def log_onehot_encoder(x, alphabet_size, eps, dtype):
    """Log space one hot encode, w/o boolean ops, w/padding."""
    # Log transform without NaN.
    xoh = (1. - x) * (-1/eps)
    # Pad.
    pad = -np.log(alphabet_size)*tf.ones(alphabet_size, dtype=dtype)[None, :]

    return tf.concat((xoh, pad), axis=0)


def encode(x, uln0, rln0, lln0,
           latent_length, latent_alphabet_size, alphabet_size,
           padded_data_length, transfer_mats, dtype=tf.float64, eps=1e-32):
    """First layer of encoder, using the MuE mean."""

    # Set initial sequence (replace inf with large number)
    vxln = tf.maximum(tf.math.log(x), -1e32)

    # Set insert biases to uniform distribution.
    vcln = -np.log(alphabet_size)*tf.ones_like(vxln)

    # Set deletion and insertion parameters.
    uln = tf.ones((padded_data_length, 2), dtype=dtype) * (
            uln0 - tf.reduce_logsumexp(uln0))[None, :]
    rln = tf.ones((padded_data_length, 2), dtype=dtype) * (
            rln0 - tf.reduce_logsumexp(rln0))[None, :]
    lln = lln0 - tf.reduce_logsumexp(lln0, axis=1, keepdims=True)

    # Build HiddenMarkovModel, with one-hot encoded output.
    a0_enc, a_enc, e_enc = make_hmm_params(
            vxln, vcln, uln, rln, lln, transfer_mats, eps=eps, dtype=dtype)

    hmm_enc = tfpd.HiddenMarkovModel(
            tfpd.Categorical(logits=a0_enc), tfpd.Categorical(logits=a_enc),
            tfpd.OneHotCategorical(logits=e_enc), latent_length)

    return hmm_mean(hmm_enc, latent_length)


def get_most_common_tuple(lst):
    """Get the mode of a list of tuples."""

    return max(set(lst), key=lst.count)
