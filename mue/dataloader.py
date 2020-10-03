import tensorflow.compat.v2 as tf
import numpy as np
from Bio import SeqIO
import pandas as pd


alphabets = {'prot': np.array(
                ['R', 'H', 'K', 'D', 'E',
                 'S', 'T', 'N', 'Q', 'C',
                 'G', 'P', 'A', 'V', 'I',
                 'L', 'M', 'F', 'Y', 'W']),
             'nt': np.array(['A', 'C', 'G', 'T'])}


def onehot_pad(seq, alphabet, max_len, dtype=tf.float32):
    """One hot encode sequence, pad with zeros."""
    # Get alphabet.
    alph = alphabets[alphabet]

    # One hot encode.
    soh = tf.convert_to_tensor(np.array(list(seq))[:, None] == alph[None, :],
                               dtype=dtype)
    # Pad with 1/alphabet_size.
    sohp = tf.concat((soh, tf.ones((max_len-len(seq), len(alph)),
                                   dtype=dtype) * (1/len(alph))), axis=0)

    return sohp


def load_seqs(file, filetype='fasta', alphabet='prot', dtype=tf.float32):
    """Load sequence file."""
    seqs = [str(elem.seq) for elem in list(SeqIO.parse(file, filetype))]
    seq_lens = np.array([len(elem) for elem in seqs], dtype=np.int32)
    max_len = np.max(seq_lens)
    data = [onehot_pad(elem, alphabet, max_len+1, dtype=dtype)
            for elem in seqs]

    seq_dataset = tf.data.Dataset.from_tensor_slices(data)
    len_dataset = tf.data.Dataset.from_tensor_slices(
                    tf.convert_to_tensor(seq_lens))

    return seq_dataset, len_dataset


def load_covariates(covariate_file, cov_filetype='csv', cov_header='None',
                    dtype=tf.float32):
    """Load covariate file."""

    # Load covariate dataset with pandas.
    if cov_filetype == 'csv':
        if cov_header == 'None':
            mat = pd.read_csv(covariate_file, header=None)
        elif cov_header == 'infer':
            mat = pd.read_csv(covariate_file)
    elif cov_filetype == 'pickle':
        mat = pd.read_pickle(covariate_file)

    # Convert to tf dataset.
    cov_dataset = tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor(mat.to_numpy(), dtype=dtype))

    return cov_dataset


def load(file, filetype='fasta', alphabet='prot', dtype=tf.float32):
    """Load sequence dataset."""

    seq_dataset, len_dataset = load_seqs(
            file, filetype=filetype, alphabet=alphabet, dtype=dtype)

    return tf.data.Dataset.zip((seq_dataset, len_dataset))


def load_joint(covariate_file, sequence_file,
               cov_filetype='csv', cov_header='None', seq_filetype='fasta',
               alphabet='prot', dtype=tf.float32):
    """Load covariate and sequence dataset."""
    # Sequence dataset.
    seq_dataset, len_dataset = load_seqs(
            sequence_file, filetype=seq_filetype, alphabet=alphabet,
            dtype=dtype)

    cov_dataset = load_covariates(covariate_file, cov_filetype=cov_filetype,
                                  cov_header=cov_header, dtype=dtype)

    return tf.data.Dataset.zip((cov_dataset, seq_dataset, len_dataset))
