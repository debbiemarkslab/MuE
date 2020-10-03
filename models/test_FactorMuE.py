import tensorflow.compat.v2 as tf
import numpy as np

from mue import core as mue
import FactorMuE


def test_single_alignment_mode():

    dtype = tf.float64
    latent_dims = 2
    latent_length = 4
    latent_alphabet_size, alphabet_size = 3, 3
    q_conc = tf.convert_to_tensor([10, 1], dtype=dtype)
    r_conc = tf.convert_to_tensor([10, 1], dtype=dtype)
    l_conc = tf.convert_to_tensor(2., dtype=dtype)
    padded_data_length = 5

    FactorMuE_variational, parameters = FactorMuE.build_FactorMuE_variational(
                                latent_dims, latent_length,
                                latent_alphabet_size, alphabet_size,
                                q_conc, r_conc, l_conc, padded_data_length,
                                z_distr='Normal', dtype=dtype)

    x = tf.convert_to_tensor(np.array([[0, 1, 0],
                                       [1, 0, 0],
                                       [0, 0, 1],
                                       [1, 0, 0],
                                       [1/3, 1/3, 1/3]]), dtype=dtype)
    xlen = 4

    transfer_mats = mue.make_transfer(latent_length, dtype=dtype)

    w_scale = tf.convert_to_tensor(1., dtype=dtype)
    b_scale = tf.convert_to_tensor(1., dtype=dtype)

    pmode_oh = FactorMuE.single_alignment_mode(
                    FactorMuE_variational, x, xlen, latent_dims, latent_length,
                    latent_alphabet_size, alphabet_size, transfer_mats,
                    w_scale, b_scale, q_conc, r_conc, l_conc, mc_samples=5,
                    dtype=dtype)
    assert pmode_oh.shape[0] == xlen
    assert pmode_oh.shape[1] == 2*(latent_length + 1)


def test_project_latent_to_sequence():

    dtype = tf.float64
    latent_dims = 2
    latent_length = 4
    latent_alphabet_size, alphabet_size = 3, 3
    q_conc = tf.convert_to_tensor([10, 1], dtype=dtype)
    r_conc = tf.convert_to_tensor([10, 1], dtype=dtype)
    l_conc = tf.convert_to_tensor(2., dtype=dtype)
    padded_data_length = 5

    FactorMuE_variational, parameters = FactorMuE.build_FactorMuE_variational(
                                latent_dims, latent_length,
                                latent_alphabet_size, alphabet_size,
                                q_conc, r_conc, l_conc, padded_data_length,
                                z_distr='Normal', dtype=dtype)

    x = tf.convert_to_tensor(np.array([[0, 1, 0],
                                       [1, 0, 0],
                                       [0, 0, 1],
                                       [1, 0, 0],
                                       [1/3, 1/3, 1/3]]), dtype=dtype)
    xlen = 4

    w_scale = tf.convert_to_tensor(1., dtype=dtype)
    b_scale = tf.convert_to_tensor(1., dtype=dtype)

    z = tf.convert_to_tensor([[1., -2.]], dtype=dtype)
    nus = FactorMuE.project_latent_to_sequence(
            z, FactorMuE_variational, latent_dims,
            latent_length, latent_alphabet_size,
            alphabet_size, w_scale, b_scale, q_conc, r_conc,
            l_conc, x=x, xlen=xlen, mc_samples=5,
            z_distr='Normal', dtype=dtype)

    assert nus[0].shape[0] == xlen
    assert nus[0].shape[1] == alphabet_size
    assert np.allclose(tf.reduce_sum(nus[0], axis=1).numpy(),
                       tf.ones((1, xlen), dtype=dtype).numpy())

    z = tf.convert_to_tensor([[1., -2.], [3., 1.]], dtype=dtype)
    nus = FactorMuE.project_latent_to_sequence(
            z, FactorMuE_variational, latent_dims,
            latent_length, latent_alphabet_size,
            alphabet_size, w_scale, b_scale, q_conc, r_conc,
            l_conc, x=None, xlen=None, mc_samples=5,
            z_distr='Normal', dtype=dtype)
    assert nus[1].shape[0] == latent_length
    assert nus[1].shape[1] == alphabet_size
    assert np.allclose(tf.reduce_sum(nus[1], axis=1).numpy(),
                       tf.ones((1, latent_length), dtype=dtype).numpy())
