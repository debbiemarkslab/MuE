import argparse
import configparser
import os
import matplotlib.pyplot as plt
import dill
from sklearn.manifold import TSNE
import numpy as np
import datetime
import json
import tensorflow.compat.v2 as tf
import pandas as pd
import logomaker
from edward2.tensorflow.generated_random_variables import (
        Dirichlet, Normal, Laplace, Exponential, Gamma)

from mue import sparse_core as mue
from mue import sparse_dataloader as dataloader
import FactorMuE


def plot_z(results, out_folder):
    """Plot first two dimensions of latent variable z."""

    z = results['embed_mean']

    plt.figure(figsize=(8, 8))
    plt.scatter(z[:, 0], z[:, -1], s=5)
    plt.xlabel(r'$z_0$', fontsize=18)
    plt.ylabel(r'$z_{}$'.format(z.shape[1]-1), fontsize=18)
    plt.savefig(os.path.join(out_folder, 'z1m1.pdf'))


def plot_z_tsne(results, out_folder):
    """Plot TSNE embedding of latent variable z."""
    # Embed.
    z = results['embed_mean']
    z_embed = TSNE(n_components=2).fit_transform(z)

    # Plot.
    plt.figure(figsize=(8, 8))
    plt.scatter(z_embed[:, 0], z_embed[:, -1], s=5)
    plt.xlabel(r'TSNE 1', fontsize=18)
    plt.ylabel(r'TSNE 2', fontsize=18)
    plt.savefig(os.path.join(out_folder, 'z_tsne.pdf'))

    # Save.
    np.save(os.path.join(out_folder, 'z_tsne.npy'), z_embed)


def _load_hyperparameters(config):
    """Load model hyperparameters."""
    dtype = getattr(tf, config['general']['precision'])
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
    latent_dims = int(config['hyperp']['latent_dims'])
    latent_alphabet_size = int(config['hyperp']['latent_alphabet_size'])
    alphabet = dataloader.alphabets[config['data']['alphabet']]
    alphabet_size = len(alphabet)
    latent_length = int(config['results']['latent_length'])

    max_delete = int(config['hyperp']['max_delete'])

    return (dtype, bt_scale, b0_scale, l_conc, u_conc, r_conc, z_distr,
            latent_dims, latent_alphabet_size, alphabet, alphabet_size,
            latent_length, max_delete)


def plot_shift(results, config, out_folder, z_tail, z_head, mc_samples=None,
               seq_ref=None):
    """Plot shift in sequence space across latent vector z."""
    # Setup.
    FactorMuE_variational = results['FactorMuE_variational']
    if mc_samples is None:
        mc_samples = int(config['train']['mc_samples'])
    else:
        mc_samples = int(mc_samples)

    # Load hyperparameters.
    (dtype, bt_scale, b0_scale, l_conc, u_conc, r_conc, z_distr,
     latent_dims, latent_alphabet_size, alphabet, alphabet_size,
     latent_length, max_delete) = _load_hyperparameters(config)

    # Load reference file.
    if seq_ref is not None:
        data_ref = dataloader.load(
                        seq_ref, filetype=config['data']['filetype'],
                        alphabet=config['data']['alphabet'], dtype=dtype)
        for x_batch, xlen_batch in data_ref.batch(1):
            x, xlen = x_batch[0], xlen_batch[0]
        prefix = 'aligned_'
    else:
        x, xlen = None, None
        prefix = ''

    # Load latent vector.
    z_head = tf.convert_to_tensor(json.loads(z_head), dtype=dtype)
    z_tail = tf.convert_to_tensor(json.loads(z_tail), dtype=dtype)
    zs = tf.concat((z_tail[None, :], z_head[None, :]), axis=0)

    # Plot embedding.
    z = results['embed_mean']
    plt.figure(figsize=(8, 8))
    plt.scatter(z[:, 0], z[:, -1], s=5)
    plt.arrow(z_tail[0], z_tail[-1],
              z_head[0] - z_tail[0], z_head[-1] - z_tail[-1],
              length_includes_head=True, head_width=0.03, color='black')
    plt.xlabel(r'$z_1$', fontsize=18)
    plt.ylabel(r'$z_2$', fontsize=18)
    plt.savefig(os.path.join(out_folder, 'z1m1_shift.pdf'))

    # Get projection.
    nus = FactorMuE.project_latent_to_sequence(
                zs, FactorMuE_variational, latent_dims,
                latent_length, max_delete, latent_alphabet_size,
                alphabet_size, bt_scale, b0_scale, u_conc, r_conc,
                l_conc, x=x, xlen=xlen, mc_samples=mc_samples, z_distr=z_distr,
                dtype=dtype)

    # Plot shift magnitude.
    nu_shift = np.sqrt(np.sum((nus[1] - nus[0]).numpy()**2, axis=1))
    plt.figure(figsize=(8, 6))
    plt.plot(nu_shift, linewidth=2)
    plt.xlabel('conserved position', fontsize=18)
    plt.ylabel('preference shift magnitude', fontsize=18)
    plt.savefig(os.path.join(out_folder, prefix + 'shift_magnitude.pdf'))
    plt.close()

    # Plot tail logo.
    df = pd.DataFrame(nus[0].numpy(), columns=alphabet)
    logomaker.Logo(df)
    plt.savefig(os.path.join(out_folder, prefix + 'tail_logo.pdf'))
    plt.close()
    df.to_csv(os.path.join(out_folder, prefix + 'tail_logo.csv'))

    # Plot head logo.
    df = pd.DataFrame(nus[1].numpy(), columns=alphabet)
    logomaker.Logo(df)
    plt.savefig(os.path.join(out_folder, prefix + 'head_logo.pdf'))
    plt.close()
    df.to_csv(os.path.join(out_folder, prefix + 'head_logo.csv'))

    # Plot shift logo.
    df = pd.DataFrame(nus[1].numpy() - nus[0].numpy(), columns=alphabet)
    logomaker.Logo(df)
    plt.savefig(os.path.join(out_folder, prefix + 'shift_logo.pdf'))
    plt.close()
    df.to_csv(os.path.join(out_folder, prefix + 'shift_logo.csv'))


def main(config, z_plot=False, z_tsne_plot=False, shift=False,
         proj_shift=False, z_tail=None, z_head=None, seq_ref=None,
         mc_samples=None):

    # Setup saving.
    run_out_folder = config['results']['out_folder']
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_folder = os.path.join(run_out_folder, time_stamp)
    os.mkdir(out_folder)

    # Load.
    with open(config['results']['file'], 'rb') as fr:
        results = dill.load(fr)

    # Plot z.
    if z_plot:
        plot_z(results, out_folder)

    # Plot z TSNE.
    if z_tsne_plot:
        plot_z_tsne(results, out_folder)

    # Plot shift.
    if shift:
        plot_shift(results, config, out_folder, z_tail, z_head,
                   mc_samples=mc_samples)

    # Plot aligned shift.
    if proj_shift:
        plot_shift(results, config, out_folder, z_tail, z_head,
                   seq_ref=seq_ref, mc_samples=mc_samples)

    # Save config.
    config.add_section('visualize')
    config['visualize']['z_plot'] = str(z_plot)
    config['visualize']['z_tsne_plot'] = str(z_tsne_plot)
    config['visualize']['shift'] = str(shift)
    config['visualize']['proj_shift'] = str(proj_shift)
    config['visualize']['z_tail'] = str(z_tail)
    config['visualize']['z_head'] = str(z_head)
    config['visualize']['seq_ref'] = str(seq_ref)
    config['visualize']['mc_samples'] = str(mc_samples)
    with open(os.path.join(out_folder, 'config.cfg'), 'w') as cw:
        config.write(cw)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('configPath')
    parser.add_argument('--z_plot', type=bool, default=False)
    parser.add_argument('--z_tsne', type=bool, default=False)
    parser.add_argument('--shift', type=bool, default=False)
    parser.add_argument('--proj_shift', type=bool, default=False)
    parser.add_argument('--z_tail', default=None)
    parser.add_argument('--z_head', default=None)
    parser.add_argument('--seq_ref', default=None)
    parser.add_argument('--mc_samples', default=None)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.configPath)

    main(config, z_plot=args.z_plot, z_tsne_plot=args.z_tsne, shift=args.shift,
         proj_shift=args.proj_shift, z_tail=args.z_tail, z_head=args.z_head,
         seq_ref=args.seq_ref, mc_samples=args.mc_samples)
