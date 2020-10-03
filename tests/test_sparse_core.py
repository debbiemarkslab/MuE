import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability import distributions as tfpd

import sparse_core as mue
import sparse_util

import pdb


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
            kp = mue.mg2k(mp, gp)
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
                    k, kp = mue.mg2k(m, g), mue.mg2k(mp, gp)
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
            k = mue.mg2k(m, g)
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


def test_make_sparse_hmm_params():

    np.random.seed(10)
    dtype = tf.float64
    M = 6
    Lsparse = 3

    transfer_mats = mue.make_transfer(M, Lsparse, dtype=dtype)

    u1 = np.random.rand(M+1)
    u = tf.convert_to_tensor(
            np.concatenate([(1-u1)[:, None], u1[:, None]], axis=1),
            dtype=tf.float64)
    r1 = np.random.rand(M+1)
    r = tf.convert_to_tensor(
            np.concatenate([(1-r1)[:, None], r1[:, None]], axis=1),
            dtype=tf.float64)
    x = tf.random.uniform((M+1, 4), dtype=dtype)
    x = x/tf.reduce_sum(x, axis=1, keepdims=True)
    c = tf.random.uniform((M+1, 4), dtype=dtype)
    c = c/tf.reduce_sum(c, axis=1, keepdims=True)

    a0, a, e = mue.make_sparse_hmm_params(tf.math.log(x), tf.math.log(c),
                                          tf.math.log(u), tf.math.log(r),
                                          None, transfer_mats, dtype=dtype)

    a_exp = sparse_util.map_values(tf.exp, a)
    a_sum = tf.sparse.reduce_sum(a_exp, axis=1)[:-1].numpy()
    assert np.allclose(a_sum, np.ones(2*(M+1)-1))

    a0_sum = tf.reduce_sum(tf.exp(a0)).numpy()
    assert np.allclose(a0_sum, 1.)

    e_sum = tf.reduce_sum(tf.exp(e), axis=1)[:-1].numpy()
    assert np.allclose(e_sum, np.ones(2*(M+1)-1))

    Lsparse = 2*M
    transfer_mats = mue.make_transfer(M, Lsparse, dtype=dtype)
    l = tf.random.uniform((4, 3), dtype=dtype)
    l = l/tf.reduce_sum(l, axis=1, keepdims=True)
    chk_a0, a, chk_e = mue.make_sparse_hmm_params(
                                          tf.math.log(x), tf.math.log(c),
                                          tf.math.log(u), tf.math.log(r),
                                          tf.math.log(l), transfer_mats,
                                          dtype=dtype)

    transfer_mats = make_transfer(M, dtype=dtype)
    eps = 1e-32
    tst_a0, tst_a, tst_e = make_hmm_params(
            tf.math.log(x), tf.math.log(c), tf.math.log(u), tf.math.log(r),
            tf.math.log(l), transfer_mats, dtype=dtype, eps=eps)

    print(chk_a0)
    print(tst_a0)
    assert np.allclose(chk_a0.numpy(), tst_a0.numpy())
    srt_chk_a = tf.sparse.reorder(a)
    mask = 1-tf.sparse.to_dense(
                sparse_util.map_values(tf.ones_like, srt_chk_a))
    zeroed_a = (-1/eps)*mask
    chk_a = (tf.sparse.to_dense(srt_chk_a) + zeroed_a).numpy()
    chk_a[-1, -1] = 0.
    assert np.allclose(chk_a, tst_a.numpy())
    assert np.allclose(chk_e.numpy(), tst_e.numpy())


def forward(a0, a, e, x, num):
    """Markov chain forward algorithm, with tf graph construction."""
    emit_lp = tf.matmul(e, x, transpose_b=True)
    f = a0 + emit_lp[:, 0]
    for j in range(1, num):
        f = tf.reduce_logsumexp(f[:, None] + a, axis=0) + emit_lp[:, j]

    return tf.reduce_logsumexp(f)


def log_sparse_to_dense(s, eps=1e-32):
    rs = tf.sparse.reorder(s)
    return (tf.sparse.to_dense(rs) +
            (1 - tf.sparse.to_dense(sparse_util.map_values(tf.ones_like, rs)))
            * (-1/eps))


def test_sparse_reduce_max():

    st = tf.sparse.from_dense(tf.convert_to_tensor(
            [[-2, 0, -1], [1, 0, -3]], dtype=tf.float64))
    chk_rs = mue._sparse_reduce_max(st, axis=0)
    tst_rs = tf.sparse.reduce_max(st, axis=0)

    assert np.allclose(chk_rs.numpy(), tst_rs.numpy())

    # check gradients work.
    vals = tf.Variable(tf.convert_to_tensor([-2., -1., 1., -3.],
                       dtype=tf.float64))
    with tf.GradientTape() as gtape:
        st = tf.sparse.SparseTensor([[0, 0], [0, 2], [1, 0], [1, 2]],
                                    vals, [2, 3])
        chk_rs = mue._sparse_reduce_max(st, axis=0)
        loss = tf.reduce_sum(chk_rs)
        gradients = gtape.gradient(loss, vals)
    assert not tf.reduce_any(tf.math.is_nan(gradients))


def test_sparse_argmax():

    st = tf.sparse.from_dense(tf.convert_to_tensor(
            [[-2, 0, -1], [1, 0, -3]], dtype=tf.float64))
    chk_rs = mue._sparse_argmax(st, axis=0)
    tst_rs = tf.argmax(tf.sparse.to_dense(st), axis=0)

    assert np.allclose(chk_rs.numpy(), tst_rs.numpy())


def test_sample_n():

    tf.random.set_seed(20)
    dtype = tf.float64
    initial_distribution = tf.math.log(
            tf.convert_to_tensor([0.999, 0.0005, 0.0005], dtype=dtype))
    transition_distribution = tf.sparse.SparseTensor(
            indices=[[0, 0], [0, 1], [1, 2], [2, 0]],
            values=tf.math.log(tf.convert_to_tensor([0.001, 0.999, 1., 1.],
                                                    dtype=dtype)),
            dense_shape=[3, 3])
    observation_distribution = tf.math.log(
            tf.convert_to_tensor([[0.001, 0.999], [0.001, 0.999],
                                  [0.999, 0.001]], dtype=dtype))
    num_steps = 3

    shmm = mue.tfpSparseHiddenMarkovModel(
            initial_distribution, transition_distribution,
            observation_distribution, num_steps)

    n_samples = 4
    sample_dummy = shmm._sample_n(n_samples, dummy=True)
    assert np.allclose(sample_dummy.numpy(), np.zeros((
        n_samples, num_steps, observation_distribution.shape[1])))

    chk_sample = shmm._sample_n(n_samples, dummy=False)
    tst_sample = (np.ones(n_samples)[:, None, None] *
                  np.array([[0., 1], [0, 1], [1, 0]])[None, :, :])
    assert np.allclose(chk_sample.numpy(), tst_sample)


def test_forward():

    np.random.seed(10)
    dtype = tf.float64
    M = 20
    Lsparse = 4

    transfer_mats = mue.make_transfer(M, Lsparse, dtype=dtype)

    u1 = np.random.rand(M+1)
    u = tf.convert_to_tensor(
            np.concatenate([(1-u1)[:, None], u1[:, None]], axis=1),
            dtype=tf.float64)
    r1 = np.random.rand(M+1)
    r = tf.convert_to_tensor(
            np.concatenate([(1-r1)[:, None], r1[:, None]], axis=1),
            dtype=tf.float64)
    x = tf.random.uniform((M+1, 3), dtype=dtype)
    x = x/tf.reduce_sum(x, axis=1, keepdims=True)
    c = tf.random.uniform((M+1, 3), dtype=dtype)
    c = c/tf.reduce_sum(c, axis=1, keepdims=True)

    a0, a, e = mue.make_sparse_hmm_params(tf.math.log(x), tf.math.log(c),
                                          tf.math.log(u), tf.math.log(r),
                                          None, transfer_mats, dtype=dtype)

    x = tf.convert_to_tensor(np.array([[0., 1., 0.],
                                       [1., 0., 0.],
                                       [0., 1., 0.],
                                       [0., 1., 0.],
                                       [1., 0., 0.],
                                       [0.33, 0.33, 0.33]]))
    xlen = tf.convert_to_tensor(5)

    distr = mue.SparseHiddenMarkovModel(a0, a, e, xlen)

    chk_lp = distr.distribution.log_prob(x)

    tst_lp = forward(a0, log_sparse_to_dense(a), e, x, xlen)

    assert np.allclose(chk_lp.numpy(), tst_lp.numpy())


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


def test_mean():

    np.random.seed(10)
    dtype = tf.float64
    M = 20
    Lsparse = 4

    transfer_mats = mue.make_transfer(M, Lsparse, dtype=dtype)

    u1 = np.random.rand(M+1)
    u = tf.convert_to_tensor(
            np.concatenate([(1-u1)[:, None], u1[:, None]], axis=1),
            dtype=tf.float64)
    r1 = np.random.rand(M+1)
    r = tf.convert_to_tensor(
            np.concatenate([(1-r1)[:, None], r1[:, None]], axis=1),
            dtype=tf.float64)
    x = tf.random.uniform((M+1, 3), dtype=dtype)
    x = x/tf.reduce_sum(x, axis=1, keepdims=True)
    c = tf.random.uniform((M+1, 3), dtype=dtype)
    c = c/tf.reduce_sum(c, axis=1, keepdims=True)

    a0, a, e = mue.make_sparse_hmm_params(tf.math.log(x), tf.math.log(c),
                                          tf.math.log(u), tf.math.log(r),
                                          None, transfer_mats, dtype=dtype)

    x = tf.convert_to_tensor(np.array([[0., 1., 0.],
                                       [1., 0., 0.],
                                       [0., 1., 0.],
                                       [0., 1., 0.],
                                       [1., 0., 0.],
                                       [0.33, 0.33, 0.33]]))
    xlen = tf.convert_to_tensor(5)

    distr = mue.SparseHiddenMarkovModel(a0, a, e, xlen)

    chk_mn = distr.distribution.mean()

    tst_mn = forward_mean(a0, log_sparse_to_dense(a), e, xlen)

    assert np.allclose(chk_mn.numpy(), tst_mn.numpy())


def test_posterior_mode():

    np.random.seed(12)
    dtype = tf.float64
    M = 20
    Lsparse = 5

    transfer_mats = mue.make_transfer(M, Lsparse, dtype=dtype)

    u1 = np.random.rand(M+1)
    u = tf.convert_to_tensor(
            np.concatenate([(1-u1)[:, None], u1[:, None]], axis=1),
            dtype=tf.float64)
    r1 = np.random.rand(M+1)
    r = tf.convert_to_tensor(
            np.concatenate([(1-r1)[:, None], r1[:, None]], axis=1),
            dtype=tf.float64)
    x = tf.random.uniform((M+1, 3), dtype=dtype)
    x = x/tf.reduce_sum(x, axis=1, keepdims=True)
    c = tf.random.uniform((M+1, 3), dtype=dtype)
    c = c/tf.reduce_sum(c, axis=1, keepdims=True)

    a0, a, e = mue.make_sparse_hmm_params(tf.math.log(x), tf.math.log(c),
                                          tf.math.log(u), tf.math.log(r),
                                          None, transfer_mats, dtype=dtype)

    x = tf.convert_to_tensor(np.array([[0., 1., 0.],
                                       [1., 0., 0.],
                                       [0., 1., 0.],
                                       [0., 1., 0.],
                                       [1., 0., 0.],
                                       [0.33, 0.33, 0.33]]))
    xlen = tf.convert_to_tensor(5)

    distr = mue.SparseHiddenMarkovModel(a0, a, e, xlen)
    chk_pm = distr.distribution.posterior_mode(x)

    tfp_hmm = tfpd.HiddenMarkovModel(
                tfpd.Categorical(logits=a0),
                tfpd.Categorical(logits=log_sparse_to_dense(a)),
                tfpd.OneHotCategorical(logits=e), xlen)
    tst_pm = tfp_hmm.posterior_mode(x[:-1, :])

    assert np.allclose(chk_pm.numpy(), tst_pm.numpy())


def test_encode():
    dtype = tf.float64
    x = tf.convert_to_tensor(np.array([[0, 1, 0],
                                       [1, 0, 0],
                                       [0, 0, 1],
                                       [1, 0, 0],
                                       [1/3, 1/3, 1/3]]), dtype=dtype)
    qln0 = tf.convert_to_tensor([10, 0], dtype=dtype)
    rln0 = tf.convert_to_tensor([10, 0], dtype=dtype)
    lln0 = tf.convert_to_tensor([[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 0, 10]],
                                dtype=dtype)
    latent_length = 4
    latent_alphabet_size = 4
    alphabet_size = 3
    Lsparse = 6
    transfer_mats = mue.make_transfer(latent_length, Lsparse, dtype=dtype)
    padded_data_length = 5

    chk_enc = mue.encode(x, qln0, rln0, lln0, latent_length,
                         latent_alphabet_size, alphabet_size,
                         padded_data_length, transfer_mats, dtype)
    tst_enc = np.array([[0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0]])
    assert np.allclose(chk_enc.numpy(), tst_enc, atol=1e-3, rtol=1e-3)
