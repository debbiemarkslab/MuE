import argparse
import configparser
from edward2.tracers import condition, tape
from edward2.tensorflow.generated_random_variables import (
        Dirichlet, HiddenMarkovModel, Normal)
import tensorflow.compat.v2 as tf
from tensorflow_probability import distributions as tfpd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import datetime
import dill

import pdb

from mue import sparse_core as mue
from mue import sparse_dataloader as dataloader


def get_prior_conc(latent_length, latent_alphabet_size, alphabet_size,
                   u_conc, r_conc, l_conc, dtype=tf.float64):
    """Get prior Dirichlet concentration parameters."""
    uc = (tf.convert_to_tensor(u_conc[None, :], dtype=dtype) *
          tf.ones((latent_length+1, 2), dtype=dtype))
    rc = (tf.convert_to_tensor(r_conc[None, :], dtype=dtype) *
          tf.ones((latent_length+1, 2), dtype=dtype))
    lc = l_conc * tf.ones((latent_alphabet_size, alphabet_size), dtype=dtype)

    return uc, rc, lc


def RegressMuE(z, latent_dims, latent_length, latent_alphabet_size,
               alphabet_size, seq_len, transfer_mats,
               bt_scale, b0_scale, u_conc, r_conc, l_conc, dtype=tf.float64):

    """Regress MuE model."""

    # Factors.
    bt = Normal(0., bt_scale, sample_shape=[2, latent_dims, latent_length+1,
                                            latent_alphabet_size], name="bt")
    # Offset.
    b0 = Normal(0., b0_scale, sample_shape=[2, latent_length+1,
                                            latent_alphabet_size], name="b0")

    # Ancestral sequence.
    vxln = tf.einsum('j,jkl->kl', z, bt[0, :, :, :]) + b0[0, :, :]
    # Insert biases.
    vcln = tf.einsum('j,jkl->kl', z, bt[1, :, :, :]) + b0[1, :, :]

    # Assemble priors.
    uc, rc, lc = get_prior_conc(
                    latent_length, latent_alphabet_size, alphabet_size,
                    u_conc, r_conc, l_conc, dtype=dtype)
    unit = tf.convert_to_tensor(1., dtype=dtype)
    # Deletion probability.
    uln = Normal(uc, unit, name="u")
    # Insertion probability.
    rln = Normal(rc, unit, name="r")
    # Substitution probability.
    lln = Normal(lc, unit, name="l")

    # Generate data from the MuE.
    a0, a, e = mue.make_sparse_hmm_params(
            vxln - tf.reduce_logsumexp(vxln, axis=1, keepdims=True),
            vcln - tf.reduce_logsumexp(vcln, axis=1, keepdims=True),
            uln - tf.reduce_logsumexp(uln, axis=1, keepdims=True),
            rln - tf.reduce_logsumexp(rln, axis=1, keepdims=True),
            lln - tf.reduce_logsumexp(lln, axis=1, keepdims=True),
            transfer_mats, dtype=dtype)
    x = mue.SparseHiddenMarkovModel(a0, a, e, seq_len, name="x")

    return x


def build_trainable_normal(shape, init_mean=None, name=None, dtype=tf.float64):
    """A normal distribution for VI."""
    # Parameters.
    if init_mean is None:
        mean = tf.Variable(tf.random.normal(shape, dtype=dtype))
    else:
        mean = tf.Variable(tf.random.normal(shape, dtype=dtype) + init_mean)
    scale = tf.Variable(tf.random.normal(shape, dtype=dtype))

    def normal():
        # Construct the VI distribution. Sofplus ensures scale is positive.
        return Normal(mean, tf.nn.softplus(scale), name=name)

    # Return distribution generator and trainable parameters.
    return normal, [mean, scale]


def build_RegressMuE_variational(latent_dims, latent_length,
                                 latent_alphabet_size, alphabet_size,
                                 u_conc, r_conc, l_conc, dtype=tf.float64):
    """Build complete variational approximation."""
    # Get individual generators.
    uc, rc, lc = get_prior_conc(
                    latent_length, latent_alphabet_size, alphabet_size,
                    u_conc, r_conc, l_conc, dtype=dtype)

    QW, qbt_params = build_trainable_normal(
            [2, latent_dims, latent_length+1, latent_alphabet_size],
            name="qbt", dtype=dtype)
    QB, qb0_params = build_trainable_normal(
            [2, latent_length+1, latent_alphabet_size],
            name="qb0", dtype=dtype)
    QU, qu_params = build_trainable_normal(uc.shape, init_mean=uc,
                                           name="qu", dtype=dtype)
    QR, qr_params = build_trainable_normal(rc.shape, init_mean=rc,
                                           name="qr", dtype=dtype)
    QL, ql_params = build_trainable_normal(lc.shape, init_mean=lc,
                                           name="ql", dtype=dtype)

    # Consolidate trainable parameters.
    parameters = qbt_params + qb0_params + qu_params + qr_params + ql_params

    # Construct generator for complete variational approximation.
    def RegressMuE_variational():
        return QW(), QB(), QU(), QR(), QL()

    # Return variational approximation generator and parameters.
    return RegressMuE_variational, parameters


def train_loop(dataset, RegressMuE_variational, trainable_variables,
               latent_dims, latent_length, latent_alphabet_size, alphabet_size,
               transfer_mats,  bt_scale, b0_scale, u_conc, r_conc, l_conc,
               max_epochs, shuffle_buffer, batch_size,
               optimizer_name, learning_rate, dtype, writer, out_folder):

    """Training loop."""
    # Set up the optimizer.
    optimizer = getattr(tf.keras.optimizers, optimizer_name)(
                                    learning_rate=learning_rate)
    data_size = sum(1 for i in dataset.batch(1))

    # Training loop.
    step = 0
    t0 = datetime.datetime.now()
    for epoch in range(max_epochs):
        for z_batch, x_batch, xlen_batch in dataset.shuffle(
                                    shuffle_buffer).batch(batch_size):

            step += 1

            accum_gradients = [tf.zeros(elem.shape, dtype=dtype)
                               for elem in trainable_variables]
            accum_elbo = 0.
            ix = -1
            for z, x, xlen in zip(z_batch, x_batch, xlen_batch):
                ix += 1

                # Track gradients.
                with tf.GradientTape() as gtape:
                    # Sample from variational approximation.
                    qbt, qb0, qu, qr, ql = RegressMuE_variational()
                    # Forward pass.
                    with tape() as model_tape:
                        # Condition on variational sample.
                        with condition(bt=qbt, b0=qb0, u=qu, r=qr, l=ql):
                            posterior_predictive = RegressMuE(
                                    z, latent_dims, latent_length,
                                    latent_alphabet_size, alphabet_size,
                                    xlen, transfer_mats, bt_scale, b0_scale,
                                    u_conc, r_conc, l_conc, dtype=dtype)

                    # Compute likelihood term.
                    log_likelihood = (
                            posterior_predictive.distribution.log_prob(x))

                    # Compute KL(vi posterior||prior) for global parameters.
                    kl_global = 0.
                    if ix == x_batch.shape[0] - 1:
                        for rv_name, variational_rv in [
                                    ("bt", qbt), ("b0", qb0),
                                    ("u", qu), ("r", qr), ("l", ql)]:
                            kl_global += tf.reduce_sum(
                                    variational_rv.distribution.kl_divergence(
                                        model_tape[rv_name].distribution))

                    # Compute the ELBO term, correcting for subsampling.
                    elbo = ((data_size/x_batch.shape[0])
                            * log_likelihood - kl_global)

                    # Compute gradient.
                    loss = -elbo
                    gradients = gtape.gradient(loss, trainable_variables)

                # Accumulate elbo and gradients.
                accum_elbo += elbo
                for gi, grad in enumerate(gradients):
                    if grad is not None:
                        accum_gradients[gi] += grad

            # Record.
            with writer.as_default():
                tf.summary.scalar('elbo', accum_elbo, step=step)

            # Optimization step.
            optimizer.apply_gradients(
                    zip(accum_gradients, trainable_variables))

        print('epoch {} ({})'.format(epoch, datetime.datetime.now() - t0))

    return RegressMuE_variational, trainable_variables


def evaluate_loop(dataset, RegressMuE_variational,
                  latent_dims, latent_length,
                  latent_alphabet_size, alphabet_size,
                  transfer_mats,  bt_scale, b0_scale, u_conc, r_conc, l_conc,
                  mc_samples, dtype, writer):

    """Evaluate heldout log likelihood and perplexity."""

    data_size = sum(1 for i in dataset.batch(1))
    heldout_likelihood = 0.
    heldout_log_perplex = 0.
    for z_batch, x_batch, xlen_batch in dataset.batch(1):
        z = z_batch[0]
        x = x_batch[0]
        xlen = xlen_batch[0]

        log_likelihood = 0.
        for rep in range(mc_samples):
            # Sample from variational approximation.
            qbt, qb0, qu, qr, ql = RegressMuE_variational()

            # Condition on variational sample.
            with condition(bt=qbt, b0=qb0, u=qu, r=qr, l=ql):
                posterior_predictive = RegressMuE(
                        z, latent_dims, latent_length,
                        latent_alphabet_size, alphabet_size,
                        xlen, transfer_mats, bt_scale, b0_scale,
                        u_conc, r_conc, l_conc, dtype=dtype)

            # Compute likelihood term.
            log_likelihood += (
                    posterior_predictive.distribution.log_prob(x)) / mc_samples

        # Summary
        local_elbo = log_likelihood
        heldout_likelihood += local_elbo
        heldout_log_perplex -= local_elbo/tf.cast(xlen * data_size, dtype)

    # Compute perplexity.
    heldout_perplex = tf.exp(heldout_log_perplex)

    # Record.
    with writer.as_default():
        tf.summary.scalar('perplexity', heldout_perplex, step=1)

    return heldout_perplex, heldout_likelihood


def train(dataset, dataset_train, dataset_test,
          latent_dims, latent_length, latent_alphabet_size, alphabet_size,
          bt_scale, b0_scale, u_conc, r_conc, l_conc, max_delete, max_epochs,
          shuffle_buffer, batch_size, optimizer_name, learning_rate,
          mc_samples, dtype, writer, out_folder):
    """Main training loop."""

    # Set up the variational approximation
    RegressMuE_variational, trainable_variables = build_RegressMuE_variational(
            latent_dims, latent_length, latent_alphabet_size, alphabet_size,
            u_conc, r_conc, l_conc, dtype=dtype)

    # Make transfer matrices.
    transfer_mats = mue.make_transfer(latent_length, max_delete, dtype=dtype)

    # Run training loop.
    RegressMuE_variational, trainable_variables = train_loop(
            dataset_train, RegressMuE_variational, trainable_variables,
            latent_dims, latent_length, latent_alphabet_size, alphabet_size,
            transfer_mats,  bt_scale, b0_scale, u_conc, r_conc, l_conc,
            max_epochs, shuffle_buffer, batch_size,
            optimizer_name, learning_rate, dtype, writer, out_folder)

    # Run evaluation loop.
    heldout_perplex, heldout_likelihood = evaluate_loop(
                      dataset_test, RegressMuE_variational,
                      latent_dims, latent_length,
                      latent_alphabet_size, alphabet_size,
                      transfer_mats,  bt_scale, b0_scale, u_conc, r_conc, l_conc,
                      mc_samples, dtype, writer)

    return (RegressMuE_variational, trainable_variables,
            heldout_perplex, heldout_likelihood)


def load_data(config, dtype):

    data = dataloader.load_joint(
                config['data']['covariate_file'],
                config['data']['sequence_file'],
                cov_filetype=config['data']['covariate_filetype'],
                cov_header=config['data']['covariate_header'],
                seq_filetype=config['data']['sequence_filetype'],
                alphabet=config['data']['alphabet'], dtype=dtype)

    return data


def single_alignment_mode(RegressMuE_variational, z, x, xlen, latent_dims,
                          latent_length, latent_alphabet_size, alphabet_size,
                          transfer_mats, bt_scale, b0_scale, u_conc, r_conc,
                          l_conc, mc_samples=1, z_distr='Normal',
                          dtype=tf.float64):
    """Align example sequence (eg. for plotting on structure)."""
    # Sample from variational approximation.
    qbt, qb0, qu, qr, ql = RegressMuE_variational()

    pmode_tuples = []
    for rep in range(mc_samples):
        # Condition on variational sample.
        with condition(bt=qbt.distribution.sample(),
                       b0=qb0.distribution.sample(),
                       u=qu.distribution.sample(),
                       r=qr.distribution.sample(), l=ql.distribution.sample()):
            posterior_predictive = RegressMuE(
                    z, latent_dims, latent_length,
                    latent_alphabet_size, alphabet_size,
                    xlen, transfer_mats, bt_scale, b0_scale,
                    u_conc, r_conc, l_conc, dtype=dtype)

        # Compute posterior mode.
        pmode = posterior_predictive.distribution.posterior_mode(x[:xlen, :])

        # Convert to tuple.
        pmode_tuples.append(tuple(pmode.numpy()))

    # Get mode (MAP).
    pmode_MAP = mue.get_most_common_tuple(pmode_tuples)
    print('{} unique alignment(s) out of {} Monte Carlo samples'.format(
          len(set(pmode_tuples)), mc_samples))

    # One hot encode (seq position x hidden state)
    pmode_oh = (tf.convert_to_tensor(pmode_MAP, dtype=tf.int32)[:, None]
                == tf.range(
      posterior_predictive.distribution._observation_distribution.shape[
        0], dtype=tf.int32)[None, :])

    return tf.cast(pmode_oh, dtype=dtype)


def project_latent_to_sequence(zs, RegressMuE_variational, latent_dims,
                               latent_length, max_delete, latent_alphabet_size,
                               alphabet_size, bt_scale, b0_scale, u_conc,
                               r_conc, l_conc, z_covar=None, x=None, xlen=None,
                               mc_samples=1, dtype=tf.float64):
    """Project from latent space to sequence space."""

    # Make transfer matrices.
    transfer_mats = mue.make_transfer(latent_length, max_delete, dtype=dtype)

    if x is not None:
        # Get alignment projection matrix.
        pmode_oh = single_alignment_mode(
                      RegressMuE_variational, z_covar, x, xlen, latent_dims,
                      latent_length, latent_alphabet_size, alphabet_size,
                      transfer_mats, bt_scale, b0_scale, u_conc, r_conc,
                      l_conc, mc_samples=mc_samples, dtype=dtype)

    else:
        # Project onto conserved positions.
        pmode_oh = tf.convert_to_tensor(
                mue.mg2k(np.arange(latent_length), 0)[:, None]
                == np.arange(2*latent_length + 2)[None, :], dtype=dtype)
        xlen = latent_length

    # Store results.
    nus = [tf.zeros([xlen, alphabet_size], dtype=dtype)
           for iz, z in enumerate(zs)]

    for rep in range(mc_samples):
        # Sample from variational approximation.
        qbt, qb0, qu, qr, ql = RegressMuE_variational()
        for iz, z in enumerate(zs):
            # Condition on latent space position.
            with condition(bt=qbt, b0=qb0, u=qu, r=qr, l=ql):
                posterior_predictive = RegressMuE(
                         z, latent_dims, latent_length,
                         latent_alphabet_size, alphabet_size,
                         xlen, transfer_mats, bt_scale, b0_scale,
                         u_conc, r_conc, l_conc, dtype=dtype)

            # Compute latent-sequence space observation.
            latseq = tf.exp(
             posterior_predictive.distribution._observation_distribution)
            # Project to sequence space.
            nus[iz] += tf.matmul(pmode_oh, latseq) / mc_samples
    return nus


def visualize(RegressMuE_variational, out_folder):

    """Visualize results."""
    qbt, qb0, qu, qr, ql = RegressMuE_variational()

    plt.figure(figsize=(8, 6))
    plt.plot(tf.math.softmax(qr.distribution.mean(), axis=1).numpy()[:, 1],
             'o', label='insert')
    plt.plot(tf.math.softmax(qu.distribution.mean(), axis=1).numpy()[:, 1],
             'o', label='delete')
    plt.legend(fontsize=18)
    plt.xlabel('position', fontsize=18)
    plt.ylabel('mean posterior probability', fontsize=16)
    plt.savefig(os.path.join(out_folder, 'ur.pdf'))

    plt.figure(figsize=(8, 8))
    plt.imshow(tf.math.softmax(ql.distribution.mean(), axis=1).numpy(),
               cmap='Blues')
    plt.title(r'$\ell$ mean posterior probability', fontsize=16)
    plt.colorbar()
    plt.savefig(os.path.join(out_folder, 'l.pdf'))


def main(config):

    # Setup.
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_folder = os.path.join(config['general']['out_folder'],
                              'logs', time_stamp)
    tf.random.set_seed(int(config['general']['seed']))
    dtype = getattr(tf, config['general']['precision'])

    # Set up the recording mechanism for TensorBoard.
    writer = tf.summary.create_file_writer(out_folder)

    # Load data.
    dataset = load_data(config, dtype)
    data_size = sum(1 for i in dataset.batch(1))
    for z, x, _ in dataset:
        latent_dims = z.shape[0]
        padded_data_length = x.shape[0]
        break

    # Train/test split.
    shuffle_buffer = data_size
    hold_out = int(float(config['general']['hold_out'])*data_size)
    dataset_shuf = dataset.shuffle(shuffle_buffer)
    dataset_test = dataset_shuf.take(hold_out)
    dataset_train = dataset_shuf.skip(hold_out)

    # Load hyperparameters.
    bt_scale = tf.convert_to_tensor(float(config['hyperp']['bt_scale']),
                                    dtype=dtype)
    b0_scale = tf.convert_to_tensor(float(config['hyperp']['b0_scale']),
                                    dtype=dtype)
    l_conc = tf.convert_to_tensor(float(config['hyperp']['l_conc']),
                                  dtype=dtype)
    u_conc = tf.convert_to_tensor(json.loads(config['hyperp']['u_conc']),
                                  dtype=dtype)
    r_conc = tf.convert_to_tensor(json.loads(config['hyperp']['r_conc']),
                                  dtype=dtype)

    if config['hyperp']['latent_length'] == 'auto':
        latent_length = int(1.1 * padded_data_length)
    else:
        latent_length = int(config['hyperp']['latent_length'])

    latent_alphabet_size = int(config['hyperp']['latent_alphabet_size'])
    alphabet_size = len(dataloader.alphabets[config['data']['alphabet']])

    max_delete = int(config['hyperp']['max_delete'])

    # Settings for training
    max_epochs = int(config['train']['max_epochs'])
    batch_size = int(config['train']['batch_size'])
    optimizer_name = config['train']['optimizer_name']
    learning_rate = float(config['train']['learning_rate'])
    mc_samples = int(config['train']['mc_samples'])

    # Train.
    (RegressMuE_variational, trainable_variables,
     heldout_perplex, heldout_likelihood) = train(
               dataset, dataset_train, dataset_test,
               latent_dims, latent_length, latent_alphabet_size, alphabet_size,
               bt_scale, b0_scale, u_conc, r_conc, l_conc, max_delete,
               max_epochs, shuffle_buffer, batch_size, optimizer_name,
               learning_rate, mc_samples, dtype, writer, out_folder)

    # Visualize.
    visualize(RegressMuE_variational, out_folder)

    # Save results.
    result_file = os.path.join(out_folder, 'results.dill')
    with open(result_file, 'wb') as rw:
        dill.dump({'RegressMuE_variational': RegressMuE_variational,
                   'trainable_variables': trainable_variables}, rw)

    # Save config.
    config['results']['heldout_perplex'] = str(heldout_perplex.numpy())
    config['results']['heldout_likelihood'] = str(heldout_likelihood.numpy())
    config['results']['latent_length'] = str(latent_length)
    config['results']['out_folder'] = out_folder
    config['results']['file'] = result_file
    with open(os.path.join(out_folder, 'config.cfg'), 'w') as cw:
        config.write(cw)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('configPath')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.configPath)

    main(config)
