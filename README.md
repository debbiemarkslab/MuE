
# MuE
Tools for developing H-MuE models in Edward2. See  [Weinstein and Marks (2020)](https://www.biorxiv.org/content/10.1101/2020.07.31.231381v1) for details.

 - The core `mue` package has tools for working with the MuE distribution.
 - The `models` folder of the repo has two example H-MuE models, the FactorMuE and the RegressMuE, which illustrate how to use the `mue` package.

## Installation

### Default:

To install, run:

    pip install "git+https://github.com/debbiemarkslab/MuE.git#egg=MuE[extras]" --use-feature=2020-resolver

    pip install "git+https://github.com/debbiemarkslab/edward2.git#egg=edward2"

### Minimal:

For a minimal installation (the MuE package alone, and not the example models and analysis scripts) run:

    pip install "git+https://github.com/debbiemarkslab/MuE.git#egg=MuE" --use-feature=2020-resolver

    pip install "git+https://github.com/debbiemarkslab/edward2.git#egg=edward2"

## Example H-MuE models

To run the example models, first clone the MuE repo and navigate to the `models` directory. Then run the scripts below. The output of each run, including plots of point estimates of key parameters, is saved in the subdirectory `examples/logs`. You can look at the optimization curve using `tensorboard --logdir=./examples/logs`.

### FactorMuE

Train the model `FactorMuE.py` with the (annotated) config file `examples/factor_config.cfg`.

    python FactorMuE.py examples/factor_config.cfg

Visualize the a latent space vector in ancestral sequence space (replace "####" with the timestamp of your output folder):

    python visualize_FactorMuE_results.py examples/logs/####/config.cfg --z_plot=True --proj_shift=True --z_tail=[-0.25] --z_head=[0.25] --seq_ref=examples/data1.fasta

### RegressMuE
Train the model `RegressMuE.py` with the (annotated) config file `examples/regress_config.cfg`.

    python RegressMuE.py examples/regress_config.cfg

Visualize the shift in the ancestral sequence with changing covariates (replace "####" with the timestamp of your output folder):

    python visualize_RegressMuE_results.py examples/logs/####/config.cfg --proj_shift=True --z_tail=[1,0] --z_head=[0,1] --covar_ref=examples/data1_covariate.csv --seq_ref=examples/data1.fasta

### Running on your own data

 - Make a new version of the config file, and edit the `[data]` section.
 - Change the `[hyperp]` section to the suggested defaults for large sequence datasets, given below in the section on Hyperparameters.
 - Set `max_epochs` to a small value (eg. 5 or 10) to start.
 - Make sure you have access to a GPU and that tensorflow detects the GPU when the code starts running (this will show up in the initial output).
 - Monitor the training curve using tensorboard.
 - Be patient. It takes 1-2 hours to train each model on a dataset of 1000 sequences. Future versions of the `mue` package will be faster.
 - Typical GPUs run out of memory when working with sequences longer than about 275 amino acids. Future versions of the `mue` package will require less memory and so be able to handle longer sequences.


## Writing new H-MuE models

Writing new H-MuE models requires a familiarity with building and training probabilistic models in Edward2.
There are three key functions in the `mue` package:

 - `make_transfer(M, dtype=tf.float64)` takes in the length of the ancestral sequence M and provides a set of constant transfer matrices (`transfer_mats`) that are used to structure the MuE distribution. It should be run once, before the training loop.
 - `make_hmm_params(vxln, vcln, uln, rln, lln, transfer_mats, dtype=tf.float64)` takes in the log of the MuE parameters, along with the transfer matrix returned by `make_transfer`, and returns the parameters of a hidden Markov model (HMM). These parameters can be fed to the HMM distribution in tensorflow probability.
 - `hmm_log_prob(distr, x, xlen)` takes in a tensorflow probability HMM distribution, and returns the probability of generating a sequence `x` of length `xlen` (the sequence is represented as a one-hot and padded encoding) . This function acts as a replacement for the built-in HMM `log_prob` function, which is incorrect ([https://github.com/tensorflow/probability/issues/958](https://github.com/tensorflow/probability/issues/958)).

 Together these functions are enough to build new H-MuE models in Edward2.
 Also note, the function `encode` offers a utility for building amortized inference networks for H-MuE models with local latent variables.

## Hyperparameters

 As defaults for the FactorMuE and RegressMuE, for large protein datasets, we recommend

    [hyperp]
    bt_scale = 1.0
    b0_scale = 1.0
    u_conc = [1000,1]
    r_conc = [1000,1]
    l_conc = 2.0
    latent_dims = 2
    latent_length = auto
    latent_alphabet_size = 25
    z_distr = Normal
    [train]
    batch_size = 5
    learning_rate = 0.01


If you start getting NaNs in the estimated ELBO, lower the learning rate. Increasing `l_conc` can also help, since the substitution matrix is the typical source of numerical errors.

Usually the number of epochs of KL annealing, `anneal_epochs`, is set to about half the total number of training epochs, `max_epochs`. If you don't find much structure in the FactorMuE latent space, you can increase `anneal_epochs` to be much larger than `max_epochs` to force the model into the auto-encoding limit.

## Harvard Medical School-specific pointers
Training is dramatically faster on a GPU. In your slurm script, include the flag

    #SBATCH -p gpu

and load the gcc and cuda modules

    module load gcc/6.2.0
    module load cuda/10.1
