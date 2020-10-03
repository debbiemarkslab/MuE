"""FactorMuE model."""
import argparse
import configparser
from edward2.tracers import condition, tape
from edward2.tensorflow.generated_random_variables import (
        Dirichlet, HiddenMarkovModel, Normal, Laplace, Exponential, Gamma)
import tensorflow.compat.v2 as tf
from tensorflow_probability import distributions as tfpd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import datetime
import dill

from mue import core as mue
from mue import dataloader


eps = 1e-32


def get_prior_conc(latent_length, latent_alphabet_size, alphabet_size,
                   u_conc, r_conc, l_conc, dtype=tf.float32):
    """Get prior Dirichlet concentration parameters."""
    uc = (tf.convert_to_tensor(u_conc[None, :], dtype=dtype) *
          tf.ones((latent_length+1, 2), dtype=dtype))
    rc = (tf.convert_to_tensor(r_conc[None, :], dtype=dtype) *
          tf.ones((latent_length+1, 2), dtype=dtype))
    lc = l_conc * tf.ones((latent_alphabet_size, alphabet_size), dtype=dtype)

    return uc, rc, lc


def FactorMuE(latent_dims, latent_length, latent_alphabet_size, alphabet_size,
              seq_len, transfer_mats, bt_scale, b0_scale, u_conc, r_conc,
              l_conc, z_distr='Normal', dtype=tf.float32):

    """Factor MuE model."""

    # Latent representation.
    z_scale = tf.convert_to_tensor(1., dtype=dtype)
    if z_distr == 'Normal':
        z = Normal(0., z_scale, sample_shape=latent_dims, name="z")
    elif z_distr == 'Laplace':
        z = Laplace(0., z_scale, sample_shape=latent_dims, name="z")
    elif z_distr == 'Exponential':
        z = Exponential(z_scale, sample_shape=latent_dims, name="z")

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

    # Assemble priors -- in this version, we use a Dirichlet.
    uc, rc, lc = get_prior_conc(
                    latent_length, latent_alphabet_size, alphabet_size,
                    u_conc, r_conc, l_conc, dtype=dtype)
    # Deletion probability.
    u = Dirichlet(uc, name="u")
    # Insertion probability.
    r = Dirichlet(rc, name="r")
    # Substitution probability.
    l = Dirichlet(lc, name="l")

    # Generate data from the MuE.
    a0, a, e = mue.make_hmm_params(
            vxln - tf.reduce_logsumexp(vxln, axis=1, keepdims=True),
            vcln - tf.reduce_logsumexp(vcln, axis=1, keepdims=True),
            tf.math.log(u), tf.math.log(r), tf.math.log(l),
            transfer_mats, eps=eps, dtype=dtype)
    x = HiddenMarkovModel(
            tfpd.Categorical(logits=a0), tfpd.Categorical(logits=a),
            tfpd.OneHotCategorical(logits=e), seq_len, name="x")

    return x


def build_trainable_dirichlet(prior_conc, name=None, dtype=tf.float32):

    """A Dirichlet distribution for VI"""

    # Concentration parameter, initialized with prior.
    conc = tf.Variable(Dirichlet(prior_conc))

    def dirichlet():
        # Construct the VI distribution.
        return Dirichlet(tf.nn.softplus(conc), name=name)

    return dirichlet, [conc]


def build_trainable_normal(shape, name=None, dtype=tf.float32):
    """A normal distribution for VI."""
    # Parameters.
    mean = tf.Variable(tf.random.normal(shape, dtype=dtype))
    scale = tf.Variable(tf.random.normal(shape, dtype=dtype))

    def normal():
        # Construct the VI distribution. Sofplus ensures scale is positive.
        return Normal(mean, tf.nn.softplus(scale), name=name)

    # Return distribution generator and trainable parameters.
    return normal, [mean, scale]


def build_trainable_infnet(latent_dims, latent_length,
                           latent_alphabet_size, alphabet_size,
                           padded_data_length, transfer_mats, z_distr='Normal',
                           name=None, dtype=tf.float32):
    """A basic VI inference network for the latent variable z."""
    # Insertion and deletion parameters.
    uln0 = tf.Variable(tf.random.normal([2], dtype=dtype))
    rln0 = tf.Variable(tf.random.normal([2], dtype=dtype))
    # Substitution parameter.
    lln = tf.Variable(tf.random.normal((alphabet_size, latent_alphabet_size),
                                       dtype=dtype))
    # Inference network parameters for mean parameter of q(z).
    mean_fac = tf.Variable(tf.random.normal(
            [latent_length, latent_alphabet_size, latent_dims], dtype=dtype))
    # Inference network parameters for std parameter of q(z).
    scale_fac = tf.Variable(tf.random.normal(
            [latent_length, latent_alphabet_size, latent_dims], dtype=dtype))

    def infnet(x):
        # MuE encoder.
        v = mue.encode(x, uln0, rln0, lln, latent_length,
                       latent_alphabet_size, alphabet_size,
                       padded_data_length, transfer_mats, dtype, eps)
        # Construct the approximate posterior using the inference network
        # parameters. Softplus ensures scale parameter is positive.
        loc = tf.einsum('jk,jkl->l', v, mean_fac)
        scale = tf.nn.softplus(tf.einsum('jk,jkl->l', v, scale_fac))
        if z_distr == 'Normal':
            return Normal(loc, scale, name=name)
        elif z_distr == 'Laplace':
            return Laplace(loc, scale, name=name)
        elif z_distr == 'Exponential':
            return Gamma(tf.nn.softplus(loc), tf.nn.softplus(scale), name=name)

    # Return distribution generator and trainable parameters.
    return infnet, [uln0, rln0, lln, mean_fac, scale_fac]


def build_FactorMuE_variational(latent_dims, latent_length,
                                latent_alphabet_size, alphabet_size,
                                u_conc, r_conc, l_conc, padded_data_length,
                                z_distr='Normal', dtype=tf.float32):
    """Build complete variational approximation."""
    # Get individual generators.
    uc, rc, lc = get_prior_conc(
                    latent_length, latent_alphabet_size, alphabet_size,
                    u_conc, r_conc, l_conc, dtype=dtype)

    enc_transfer_mats = mue.make_transfer(padded_data_length-1, dtype=dtype)
    QZ, qz_params = build_trainable_infnet(
            latent_dims, latent_length, latent_alphabet_size, alphabet_size,
            padded_data_length, enc_transfer_mats,
            z_distr=z_distr, name="qz", dtype=dtype)

    QW, qbt_params = build_trainable_normal(
            [2, latent_dims, latent_length+1, latent_alphabet_size],
            name="qbt", dtype=dtype)
    QB, qb0_params = build_trainable_normal(
            [2, latent_length+1, latent_alphabet_size],
            name="qb0", dtype=dtype)
    QU, qu_params = build_trainable_dirichlet(uc, name="qu", dtype=dtype)
    QR, qr_params = build_trainable_dirichlet(rc, name="qr", dtype=dtype)
    QL, ql_params = build_trainable_dirichlet(lc, name="ql", dtype=dtype)

    # Consolidate trainable parameters.
    parameters = (qz_params + qbt_params + qb0_params + qu_params + qr_params
                  + ql_params)

    # Construct generator for complete variational approximation.
    def FactorMuE_variational(x=None):
        if x is None:
            return QW(), QB(), QU(), QR(), QL()
        else:
            return QZ(x), QW(), QB(), QU(), QR(), QL()

    # Return variational approximation generator and parameters.
    return FactorMuE_variational, parameters


def anneal_beta(data_count, data_size, anneal_epochs):

    """Annealing schedule for beta (beta-VAE-style)."""
    return np.minimum(data_count/(data_size*anneal_epochs), 1.)


def train_loop(dataset, FactorMuE_variational, trainable_variables,
               latent_dims, latent_length, latent_alphabet_size, alphabet_size,
               transfer_mats,  bt_scale, b0_scale, u_conc, r_conc, l_conc,
               z_distr, anneal_epochs, max_epochs, shuffle_buffer, batch_size,
               optimizer_name, learning_rate, dtype, writer, out_folder):

    """Training loop."""
    # Set up the optimizer.
    optimizer = getattr(tf.keras.optimizers, optimizer_name)(
                                    learning_rate=learning_rate)
    data_size = sum(1 for i in dataset.batch(1))

    # Training loop.
    step = 0
    data_count = 0
    t0 = datetime.datetime.now()
    for epoch in range(max_epochs):
        for x_batch, xlen_batch in dataset.shuffle(
                                    shuffle_buffer).batch(batch_size):

            step += 1

            accum_gradients = [tf.zeros(elem.shape, dtype=dtype)
                               for elem in trainable_variables]
            accum_elbo = 0.
            ix = -1
            for x, xlen in zip(x_batch, xlen_batch):
                ix += 1

                # Get beta adjustment.
                data_count += 1
                beta = anneal_beta(data_count, data_size, anneal_epochs)

                # Track gradients.
                with tf.GradientTape() as gtape:
                    # Sample from variational approximation.
                    qz, qbt, qb0, qu, qr, ql = FactorMuE_variational(x)
                    # Forward pass.
                    with tape() as model_tape:
                        # Condition on variational sample.
                        with condition(z=qz, bt=qbt, b0=qb0, u=qu, r=qr, l=ql):
                            posterior_predictive = FactorMuE(
                                    latent_dims, latent_length,
                                    latent_alphabet_size, alphabet_size,
                                    xlen, transfer_mats, bt_scale, b0_scale,
                                    u_conc, r_conc, l_conc,
                                    z_distr=z_distr, dtype=dtype)

                    # Compute likelihood term.
                    log_likelihood = mue.hmm_log_prob(
                            posterior_predictive.distribution, x, xlen)

                    # Compute KL(vi posterior||prior) for local parameters.
                    kl_local = 0.
                    for rv_name, variational_rv in [("z", qz)]:
                        kl_local += tf.reduce_sum(
                                variational_rv.distribution.kl_divergence(
                                    model_tape[rv_name].distribution))

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
                            * (log_likelihood - beta*kl_local) - kl_global)

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
                tf.summary.scalar('beta', beta, step=step)

            # Optimization step.
            optimizer.apply_gradients(
                    zip(accum_gradients, trainable_variables))

        print('epoch {} ({})'.format(epoch, datetime.datetime.now() - t0))

    return FactorMuE_variational, trainable_variables


def evaluate_loop(dataset, FactorMuE_variational,
                  latent_dims, latent_length,
                  latent_alphabet_size, alphabet_size,
                  transfer_mats,  bt_scale, b0_scale, u_conc, r_conc, l_conc,
                  z_distr,  mc_samples, dtype, writer):

    """Evaluate heldout log likelihood and perplexity."""

    data_size = sum(1 for i in dataset.batch(1))
    heldout_likelihood = 0.
    heldout_log_perplex = 0.
    for x_batch, xlen_batch in dataset.batch(1):
        x = x_batch[0]
        xlen = xlen_batch[0]

        log_likelihood = 0.
        for rep in range(mc_samples):
            # Sample from variational approximation.
            qz, qbt, qb0, qu, qr, ql = FactorMuE_variational(x)

            # Forward pass.
            with tape() as model_tape:
                # Condition on variational sample.
                with condition(z=qz, bt=qbt, b0=qb0, u=qu, r=qr, l=ql):
                    posterior_predictive = FactorMuE(
                            latent_dims, latent_length,
                            latent_alphabet_size, alphabet_size,
                            xlen, transfer_mats, bt_scale, b0_scale,
                            u_conc, r_conc, l_conc,
                            z_distr=z_distr, dtype=dtype)

            # Compute likelihood term.
            log_likelihood += mue.hmm_log_prob(
                    posterior_predictive.distribution, x, xlen) / mc_samples

        # Compute KL(vi posterior||prior) for local parameters.
        kl_local = 0.
        for rv_name, variational_rv in [("z", qz)]:
            kl_local += tf.reduce_sum(
                    variational_rv.distribution.kl_divergence(
                        model_tape[rv_name].distribution))

        # Summary
        local_elbo = log_likelihood - kl_local
        heldout_likelihood += local_elbo
        heldout_log_perplex -= local_elbo/tf.cast(xlen * data_size, dtype)

    # Compute perplexity.
    heldout_perplex = tf.exp(heldout_log_perplex)

    # Record.
    with writer.as_default():
        tf.summary.scalar('perplexity', heldout_perplex, step=1)

    return heldout_perplex, heldout_likelihood


def embed(dataset, FactorMuE_variational, latent_dims, dtype):

    """Embed full dataset."""

    embed_mean = []
    embed_std = []
    ix = -1
    for x_batch, xlen_batch in dataset.batch(1):
        ix += 1
        x = x_batch[0]

        # Sample from variational approximation.
        qz, qbt, qb0, qu, qr, ql = FactorMuE_variational(x)

        # Save embedding.
        embed_mean.append(qz.distribution.mean()[None, :])
        embed_std.append(qz.distribution.stddev()[None, :])

    return tf.concat(embed_mean, axis=0), tf.concat(embed_std, axis=0)


def train(dataset, dataset_train, dataset_test, padded_data_length,
          latent_dims, latent_length, latent_alphabet_size, alphabet_size,
          bt_scale, b0_scale, u_conc, r_conc, l_conc, z_distr,
          anneal_epochs, max_epochs, shuffle_buffer,
          batch_size, optimizer_name, learning_rate, mc_samples, dtype,
          writer, out_folder):
    """Main training loop."""

    # Set up the variational approximation
    FactorMuE_variational, trainable_variables = build_FactorMuE_variational(
            latent_dims, latent_length, latent_alphabet_size, alphabet_size,
            u_conc, r_conc, l_conc, padded_data_length,
            z_distr=z_distr, dtype=dtype)

    # Make transfer matrices.
    transfer_mats = mue.make_transfer(latent_length, dtype=dtype)

    # Run training loop.
    FactorMuE_variational, trainable_variables = train_loop(
            dataset_train, FactorMuE_variational, trainable_variables,
            latent_dims, latent_length, latent_alphabet_size, alphabet_size,
            transfer_mats,  bt_scale, b0_scale, u_conc, r_conc, l_conc,
            z_distr, anneal_epochs, max_epochs, shuffle_buffer, batch_size,
            optimizer_name, learning_rate, dtype, writer, out_folder)

    # Run evaluation loop.
    heldout_perplex, heldout_likelihood = evaluate_loop(
                      dataset_test, FactorMuE_variational,
                      latent_dims, latent_length,
                      latent_alphabet_size, alphabet_size,
                      transfer_mats,  bt_scale, b0_scale, u_conc, r_conc, l_conc,
                      z_distr, mc_samples, dtype, writer)

    # Create embedding.
    embed_mean, embed_std = embed(dataset, FactorMuE_variational, latent_dims,
                                  dtype)

    return (FactorMuE_variational, trainable_variables,
            heldout_perplex, heldout_likelihood,
            embed_mean, embed_std)


def load_data(config, dtype):

    data = dataloader.load(config['data']['file'],
                           filetype=config['data']['filetype'],
                           alphabet=config['data']['alphabet'],
                           dtype=dtype)

    return data


def single_alignment_mode(FactorMuE_variational, x, xlen, latent_dims,
                          latent_length, latent_alphabet_size, alphabet_size,
                          transfer_mats, bt_scale, b0_scale, u_conc, r_conc,
                          l_conc, mc_samples=1, z_distr='Normal',
                          dtype=tf.float32):
    """Align example sequence (eg. for plotting on structure)."""
    # Sample from variational approximation.
    qz, qbt, qb0, qu, qr, ql = FactorMuE_variational(x)

    pmode_tuples = []
    for rep in range(mc_samples):
        # Condition on variational sample.
        with condition(z=qz.distribution.sample(),
                       bt=qbt.distribution.sample(),
                       b0=qb0.distribution.sample(),
                       u=qu.distribution.sample(),
                       r=qr.distribution.sample(), l=ql.distribution.sample()):
            posterior_predictive = FactorMuE(
                    latent_dims, latent_length,
                    latent_alphabet_size, alphabet_size,
                    xlen, transfer_mats, bt_scale, b0_scale,
                    u_conc, r_conc, l_conc, z_distr=z_distr, dtype=dtype)

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
      posterior_predictive.distribution.observation_distribution.logits.shape[
        0], dtype=tf.int32)[None, :])

    return tf.cast(pmode_oh, dtype=dtype)


def project_latent_to_sequence(zs, FactorMuE_variational, latent_dims,
                               latent_length, latent_alphabet_size,
                               alphabet_size, bt_scale, b0_scale, u_conc,
                               r_conc, l_conc, x=None, xlen=None,
                               mc_samples=1, z_distr='Normal',
                               dtype=tf.float32):
    """Project from latent space to sequence space."""

    # Make transfer matrices.
    transfer_mats = mue.make_transfer(latent_length, dtype=dtype)

    if x is not None:
        # Get alignment projection matrix.
        pmode_oh = single_alignment_mode(
                          FactorMuE_variational, x, xlen, latent_dims,
                          latent_length, latent_alphabet_size, alphabet_size,
                          transfer_mats, bt_scale, b0_scale, u_conc, r_conc,
                          l_conc, mc_samples=mc_samples, z_distr=z_distr,
                          dtype=dtype)

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
        qbt, qb0, qu, qr, ql = FactorMuE_variational()
        for iz, z in enumerate(zs):
            # Condition on latent space position.
            with condition(z=z, bt=qbt, b0=qb0, u=qu, r=qr, l=ql):
                posterior_predictive = FactorMuE(
                         latent_dims, latent_length,
                         latent_alphabet_size, alphabet_size,
                         xlen, transfer_mats, bt_scale, b0_scale,
                         u_conc, r_conc, l_conc,
                         z_distr=z_distr, dtype=dtype)

            # Compute latent-sequence space observation.
            latseq = tf.exp(
             posterior_predictive.distribution.observation_distribution.logits)
            # Project to sequence space.
            nus[iz] += tf.matmul(pmode_oh, latseq) / mc_samples
    return nus


def visualize(FactorMuE_variational, embed_mean, embed_std, out_folder):

    """Visualize results."""
    qbt, qb0, qu, qr, ql = FactorMuE_variational()

    plt.figure(figsize=(8, 6))
    plt.plot(qr.distribution.mean().numpy()[:, 1], 'o', label='insert')
    plt.plot(qu.distribution.mean().numpy()[:, 1], 'o', label='delete')
    plt.legend(fontsize=18)
    plt.xlabel('position', fontsize=18)
    plt.ylabel('mean posterior probability', fontsize=16)
    plt.savefig(os.path.join(out_folder, 'ur.pdf'))

    plt.figure(figsize=(8, 8))
    plt.imshow(ql.distribution.mean().numpy(), cmap='Blues')
    plt.title(r'$\ell$ mean posterior probability', fontsize=16)
    plt.colorbar()
    plt.savefig(os.path.join(out_folder, 'l.pdf'))

    plt.figure(figsize=(8, 8))
    emn = embed_mean.numpy()
    esn = embed_std.numpy()
    plt.errorbar(emn[:, 0], emn[:, -1], xerr=esn[:, 0], yerr=esn[:, -1],
                 fmt='o')
    plt.xlabel(r'$z_0$', fontsize=18)
    plt.ylabel(r'$z_{}$'.format(emn.shape[1]-1), fontsize=18)
    plt.savefig(os.path.join(out_folder, 'z.pdf'))


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
    for x, _ in dataset:
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
    z_distr = config['hyperp']['z_distr']

    if config['hyperp']['latent_length'] == 'auto':
        latent_length = int(1.1 * padded_data_length)
    else:
        latent_length = int(config['hyperp']['latent_length'])

    latent_dims = int(config['hyperp']['latent_dims'])
    latent_alphabet_size = int(config['hyperp']['latent_alphabet_size'])
    alphabet_size = len(dataloader.alphabets[config['data']['alphabet']])

    # Settings for training
    anneal_epochs = int(config['train']['anneal_epochs'])
    max_epochs = int(config['train']['max_epochs'])
    batch_size = int(config['train']['batch_size'])
    optimizer_name = config['train']['optimizer_name']
    learning_rate = float(config['train']['learning_rate'])
    mc_samples = int(config['train']['mc_samples'])

    # Train.
    (FactorMuE_variational, trainable_variables,
     heldout_perplex, heldout_likelihood, embed_mean, embed_std) = train(
               dataset, dataset_train, dataset_test, padded_data_length,
               latent_dims, latent_length, latent_alphabet_size, alphabet_size,
               bt_scale, b0_scale, u_conc, r_conc, l_conc, z_distr,
               anneal_epochs, max_epochs, shuffle_buffer,
               batch_size, optimizer_name, learning_rate, mc_samples, dtype,
               writer, out_folder)

    # Visualize.
    visualize(FactorMuE_variational, embed_mean, embed_std, out_folder)

    # Save results.
    result_file = os.path.join(out_folder, 'results.pickle')
    with open(result_file, 'wb') as rw:
        dill.dump({'FactorMuE_variational': FactorMuE_variational,
                   'trainable_variables': trainable_variables,
                   'embed_mean': embed_mean.numpy(),
                   'embed_std': embed_std.numpy()}, rw)

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
