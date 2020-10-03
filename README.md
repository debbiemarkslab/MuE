

# MuE
Tools for developing H-MuE models in Edward2. See  [Weinstein and Marks (2020)](https://www.biorxiv.org/content/10.1101/2020.07.31.231381v1) for details.

 - The core `mue` package has tools for working with the MuE distribution.
 - The `models` and `sparse-models` folders each have two example H-MuE models, the FactorMuE and the RegressMuE, which illustrate how to use the `mue` package.

## Installation

To install the package, create a new python 3 virtual environment (eg. using conda) and run:

    pip install "git+https://github.com/debbiemarkslab/MuE.git#egg=MuE[extras]" --use-feature=2020-resolver

    pip install "git+https://github.com/debbiemarkslab/edward2.git#egg=edward2"

This shouldn't take more than a minute or two. You can find out more about the package requirements in the `setup.py` file.

## Demonstration H-MuE models

To run the example models, first clone this MuE repo and navigate to the `sparse-models` directory.
Each model (`FactorMuE.py` and `RegressMuE.py`) can be configured with a config file, such as `examples/factor_config.cfg` (for `FactorMuE.py`) and  `examples/regress_config.cfg` (for `RegressMuE.py`). The config file sets the dataset, hyperparameters, training time, etc. Descriptions of all the options can be found in the config files themselves.
The output of each run, including plots of point estimates of key parameters, is currently configured to be saved in the subdirectory `examples/logs`.
You can inspect the ELBO optimization curve by running `tensorboard --logdir=./examples/logs` from the `sparse-models` directory.
After training,

### FactorMuE

To run the example FactorMuE model, on the example dataset `examples/data1.fasta`, enter

    python FactorMuE.py examples/factor_config.cfg

The model should take roughly ~1 min. to finish the training run (100 epochs). In `examples/logs` you should find a new folder, named with a timestamp corresponding the when the run began (eg. 20201003-144700 for Oct. 3, 2020, 2:47 pm). This contains:

 - A summary of the run in `config.cfg`, which copies the input config file but also adds scalar results such as the heldout perplexity. Under the `[results]` heading the entry `heldout_perplex` should be roughly 2.0 in this example.
 - A tensorflow log file (the filename starts with `events.out`), which can be read by tensorboard.
 - Three plots, showing the mean and variance of the posterior latent representation (`z.pdf` there should be four distinct points near zero) as well as the posterior mean mutation matrix (`l.pdf`) and the posterior mean deletion and insertion parameters (`ur.pdf` all the points should be very close to zero in this example).
 - The detailed results in `results.dill`. This file is saved using the `dill` package https://dill.readthedocs.io/en/latest/dill.html. An illustration of how to load and analyze these results can be found in `visualize_FactorMuE_results.py`.

To visualize the results of the model, we often want to project vectors from the latent space onto a reference sequence (this is described in detail in the article). The script `visualize_FactorMuE_results.py` provides a tool for accomplishing this. Run the following command, replacing "####" with the timestamp of your output folder:

    python visualize_FactorMuE_results.py examples/logs/####/config.cfg --z_plot=True --proj_shift=True --z_tail=[0.1] --z_head=[0.2] --seq_ref=examples/data1.fasta

`z_tail` sets the position of the start of the latent space vector and `z_head` sets the position of the end of the latent space vector. The first entry in the fasta file input to `seq_ref` controls the reference sequence (in this case, ATAT).  The output will be in a new time-stamped folder within the model output folder `examples/logs/####`.  You'll find plots of
 - The projected tail and head of the vector, represented as a logo (`aligned_tail_logo.pdf` and `aligned_head_logo.pdf`), and the difference between them (`aligned_shift_logo.pdf`).
 - The magnitude of the shift in amino acid preference across the reference sequence (`aligned_shift_magnitude.pdf`), which is the nu vector described in the supplementary material of the paper. In this small example dataset, there isn't enough evidence for the model to find much change across the latent space, so the magnitude should be small (less than 0.1).

### RegressMuE
The `RegressMuE.py` works the same way. You can run the example dataset (with the covariate file `data1_covariate.csv`):

    python RegressMuE.py examples/regress_config.cfg

Then visualize the shift in the ancestral sequence with changing covariates, rather than changing latent representation (replace "####" with the timestamp of your output folder):

    python visualize_RegressMuE_results.py examples/logs/####/config.cfg --proj_shift=True --z_tail=[1,0] --z_head=[0,1] --covar_ref=examples/data1_covariate.csv --seq_ref=examples/data1.fasta

The resulting plot `aligned_shift_logo` should show the nucleotide T gaining in probability at positions 1 and 3 of the logo, while the nucleotide C drops in probability about the same amount, revealing that sequences with covariate [0,1] are more likely to have a T and less likely to have a C at these positions when compared to sequences with covariate [1,0].

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

## Tests
To run the unit tests, navigate to the `tests` directory of the repo and run

	pytest

All tests should pass.
