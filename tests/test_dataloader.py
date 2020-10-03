import tensorflow.compat.v2 as tf
import numpy as np

from mue import dataloader

import pdb


def test_onehot_pad():

    seq = 'ATC'
    alphabet = 'nt'
    max_len = 5
    dtype = tf.float64

    chk_sohp = np.array([[1, 0, 0, 0],
                         [0, 0, 0, 1],
                         [0, 1, 0, 0],
                         [0.25, 0.25, 0.25, 0.25],
                         [0.25, 0.25, 0.25, 0.25]], dtype=np.float64)
    tst_sohp = dataloader.onehot_pad(seq, alphabet, max_len, dtype=dtype)

    assert np.allclose(chk_sohp, tst_sohp.numpy())


def test_load():

    file = 'test_data.fasta'
    filetype = 'fasta'
    alphabet = 'nt'
    dtype = tf.float64

    data = dataloader.load(file, filetype=filetype, alphabet=alphabet,
                           dtype=dtype)
    chk_sohps = [np.array([[1, 0, 0, 0],
                           [0, 0, 0, 1],
                           [1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0.25, 0.25, 0.25, 0.25],
                           [0.25, 0.25, 0.25, 0.25]], dtype=np.float64),
                 np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1],
                           [0, 0, 1, 0],
                           [0.25, 0.25, 0.25, 0.25]], dtype=np.float64),
                 np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0],
                           [0, 1, 0, 0],
                           [0.25, 0.25, 0.25, 0.25],
                           [0.25, 0.25, 0.25, 0.25],
                           [0.25, 0.25, 0.25, 0.25],
                           [0.25, 0.25, 0.25, 0.25]], dtype=np.float64)]
    chk_lens = [5, 6, 3]

    it = 0
    for sohp, slen in data:
        assert np.allclose(sohp.numpy(), chk_sohps[it])
        assert np.allclose(slen.numpy(), chk_lens[it])
        it += 1
