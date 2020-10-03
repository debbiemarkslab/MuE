from mue import core as mue
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability import distributions as tfpd


def test_get_hmm_parameters():
    np.random.seed(10)
    dtype = tf.float64
    M = 4

    transfer_mats = mue.make_transfer(M, dtype=dtype)

    q1 = np.random.rand(M+1)
    q = tf.convert_to_tensor(
            np.concatenate([(1-q1)[:, None], q1[:, None]], axis=1),
            dtype=tf.float64)
    r1 = np.random.rand(M+1)
    r = tf.convert_to_tensor(
            np.concatenate([(1-r1)[:, None], r1[:, None]], axis=1),
            dtype=tf.float64)
    s = tf.random.uniform((M+1, 4), dtype=dtype)
    s = s/tf.reduce_sum(s, axis=1, keepdims=True)
    c = tf.random.uniform((M+1, 4), dtype=dtype)
    c = c/tf.reduce_sum(c, axis=1, keepdims=True)

    a0ln, aln, eln = mue.make_hmm_params(tf.math.log(s), tf.math.log(c),
                                         tf.math.log(q), tf.math.log(r),
                                         None, transfer_mats, eps=1e-32,
                                         dtype=dtype)

    # - Remake transition matrices. -
    q1[-1] = 1e-32
    K = 2*(M+1)
    chk_a = np.zeros((K, K))
    chk_a0 = np.zeros((K,))
    m, g = -1, 0
    for mp in range(M+1):
        for gp in range(2):
            kp = mue.mg2k(mp, gp)
            if m + 1 - g == mp and gp == 0:
                chk_a0[kp] = (1 - r1[m+1-g])*(1 - q1[m+1-g])
            elif m + 1 - g < mp and gp == 0:
                chk_a0[kp] = (
                        (1 - r1[m+1-g]) * q1[m+1-g] *
                        np.prod([(1 - r1[mpp])*q1[mpp] for mpp in
                                 range(m+2-g, mp)]) *
                        (1 - r1[mp]) * (1 - q1[mp]))
            elif m + 1 - g == mp and gp == 1:
                chk_a0[kp] = r1[m+1-g]
            elif m + 1 - g < mp and gp == 1:
                chk_a0[kp] = (
                        (1 - r1[m+1-g]) * q1[m+1-g] *
                        np.prod([(1 - r1[mpp])*q1[mpp] for mpp in
                                 range(m+2-g, mp)]) * r1[mp])
    for m in range(M+1):
        for g in range(2):
            k = mue.mg2k(m, g)
            for mp in range(M+1):
                for gp in range(2):
                    kp = mue.mg2k(mp, gp)
                    if m + 1 - g == mp and gp == 0:
                        chk_a[k, kp] = (1 - r1[m+1-g])*(1 - q1[m+1-g])
                    elif m + 1 - g < mp and gp == 0:
                        chk_a[k, kp] = (
                                (1 - r1[m+1-g]) * q1[m+1-g] *
                                np.prod([(1 - r1[mpp])*q1[mpp] for mpp in
                                         range(m+2-g, mp)]) *
                                (1 - r1[mp]) * (1 - q1[mp]))
                    elif m + 1 - g == mp and gp == 1:
                        chk_a[k, kp] = r1[m+1-g]
                    elif m + 1 - g < mp and gp == 1:
                        chk_a[k, kp] = (
                                (1 - r1[m+1-g]) * q1[m+1-g] *
                                np.prod([(1 - r1[mpp])*q1[mpp] for mpp in
                                         range(m+2-g, mp)]) * r1[mp])
                    elif m == M and mp == M and g == 0 and gp == 0:
                        chk_a[k, kp] = 1.

    chk_e = np.zeros((2*(M+1), 4), dtype=np.float64)
    for m in range(M+1):
        for g in range(2):
            k = mue.mg2k(m, g)
            if g == 0:
                chk_e[k, :] = s[m, :].numpy()
            else:
                chk_e[k, :] = c[m, :].numpy()
    # - -

    assert np.allclose(chk_a0, tf.math.exp(a0ln).numpy())
    assert np.allclose(chk_a, tf.math.exp(aln).numpy())
    assert np.allclose(chk_e, tf.math.exp(eln).numpy())

    # Check normalization.
    assert np.allclose(tf.reduce_sum(tf.math.exp(a0ln)).numpy(), 1., atol=1e-3,
                       rtol=1e-3)
    assert np.allclose(tf.reduce_sum(tf.math.exp(aln), axis=1).numpy()[:-1],
                       tf.ones(2*(M+1)-1, dtype=dtype).numpy(), atol=1e-3,
                       rtol=1e-3)


def test_hmm_log_prob():

    a0 = np.array([0.9, 0.08, 0.02])
    a = np.array([[0.1, 0.8, 0.1], [0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
    e = np.array([[0.99, 0.01], [0.01, 0.99], [0.5, 0.5]])

    model = tfpd.HiddenMarkovModel(
                tfpd.Categorical(logits=tf.math.log(tf.convert_to_tensor(
                                            np.matmul(a0, a)))),
                tfpd.Categorical(logits=tf.math.log(tf.convert_to_tensor(a))),
                tfpd.Categorical(logits=tf.math.log(tf.convert_to_tensor(e))),
                5)

    x = tf.convert_to_tensor(np.array([[0., 1.],
                                       [1., 0.],
                                       [0., 1.],
                                       [0., 1.],
                                       [1., 0.],
                                       [0.5, 0.5]]))
    xlen = tf.convert_to_tensor(5)

    chk_lp = mue.hmm_log_prob(model, x, xlen)

    f = np.matmul(a0, a) * e[:, 1]
    f = np.matmul(f, a) * e[:, 0]
    f = np.matmul(f, a) * e[:, 1]
    f = np.matmul(f, a) * e[:, 1]
    f = np.matmul(f, a) * e[:, 0]
    tst_lp = np.log(np.sum(f))

    assert np.allclose(chk_lp.numpy(), tst_lp)

    # Check against (predictably incorrect) tensorflow probability
    # implementation.
    model = tfpd.HiddenMarkovModel(
                tfpd.Categorical(logits=tf.math.log(tf.convert_to_tensor(a0))),
                tfpd.Categorical(logits=tf.math.log(tf.convert_to_tensor(a))),
                tfpd.Categorical(logits=tf.math.log(tf.convert_to_tensor(e))),
                5)
    xcat = tf.convert_to_tensor([1, 0, 1, 1, 0])
    tst_lp2 = model.log_prob(xcat)

    assert np.allclose(chk_lp.numpy(), tst_lp2.numpy())


def test_forward_mean():

    a0 = np.array([0.9, 0.08, 0.02])
    a = np.array([[0.1, 0.8, 0.1], [0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
    e = np.array([[0.99, 0.01], [0.01, 0.99], [0.5, 0.5]])

    model = tfpd.HiddenMarkovModel(
                tfpd.Categorical(logits=tf.math.log(tf.convert_to_tensor(a0))),
                tfpd.Categorical(logits=tf.math.log(tf.convert_to_tensor(a))),
                tfpd.OneHotCategorical(
                        logits=tf.math.log(tf.convert_to_tensor(e))),
                5)

    tst_mean = model.mean()

    chk_mean = mue.hmm_mean(model, 5)

    assert np.allclose(tst_mean.numpy(), chk_mean.numpy())


def test_get_most_common_tuple():

    lst = [(1, 2), (1, 2, 3), (1, 2, 3, 4), (1, 2, 3), (1,)]

    assert mue.get_most_common_tuple(lst) == (1, 2, 3)


def test_MuE_encode():
    dtype = tf.float32
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
    transfer_mats = mue.make_transfer(latent_length, dtype=dtype)
    padded_data_length = 5
    eps = 1e-32

    chk_enc = mue.encode(x, qln0, rln0, lln0, latent_length,
                         latent_alphabet_size, alphabet_size,
                         padded_data_length, transfer_mats, dtype, eps)
    tst_enc = np.array([[0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0]])
    assert np.allclose(chk_enc.numpy(), tst_enc, atol=1e-3, rtol=1e-3)
