import sys
import argparse
import torch
import jax.numpy as jnp
import numpy as np


# Parse arguments for simulation
def get_sim_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--manual-control',
                        action='store_true', help='Start manual control')
    parser.add_argument('-i', '--inference',
                        action='store_true', help='Start inference')

    args = parser.parse_args()

    return args


# Parse arguments for plots
def get_plot_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video',
                        action='store_true', help='Record video')

    args = parser.parse_args()
    return args


# Print info
def print_score(log):
    pass


# Print simulation info
def print_info(trial, n_trials, step, n_steps):
    sys.stdout.write('\rTrial: {:4d}/{:4d}\tStep: {:4d}/{:4d}'.format(
        trial + 1, n_trials, step + 1, n_steps))
    sys.stdout.flush()


# Add Gaussian noise to array
def add_gaussian_noise(array, noise):
    sigma = noise ** 0.5
    return array + np.random.normal(0, sigma, np.shape(array))


# Normalize data
def normalize(x, limits, rng=True):
    x_norm = (x - limits[0]) / (limits[1] - limits[0])
    if rng:
        x_norm = x_norm * 2 - 1
    return x_norm


# Denormalize data
def denormalize(x, limits, rng=True):
    x_denorm = (x + 1) / 2 if rng else x
    x_denorm = x_denorm * (limits[1] - limits[0]) + limits[0]
    return x_denorm


# Get onehot array
def one_hot(lengths, indices):
    arrays = []
    for length, index in zip(lengths, indices):
        onehot = jnp.zeros(length)
        onehot = onehot.at[index].set(1.0)
        arrays.append(jnp.expand_dims(onehot, [0, 1]))

    return arrays


# Compute stable logarithm
def log_stable(arr):
    return torch.log(arr + 1e-16)


# Compute softmax function
def softmax(dist, precision=1.0):
    output = dist - dist.max(dim=0).values
    output = torch.exp(precision * output)
    output = output / output.sum(dim=0)
    return output


def acc_log_evidence(eta, Eta_m, mu, pi, pi_m, p):
    """
    Accumulate log evidence for discrete model
    :param eta: prior of full model
    :param Eta_m: priors of reduced models
    :param mu: posterior of full model
    :param pi: prior precision of full model
    :param pi_m: prior precision of reduced models
    :param p: posterior precision of full model
    :return: log evidence
    """
    eta_rng = normalize(eta.detach().clone(), (-1, 1), rng=False)
    Eta_m_rng = normalize(Eta_m.detach().clone(), (-1, 1), rng=False)
    mu_rng = normalize(mu.detach().clone(), (-1, 1), rng=False)

    p_m = p - pi + pi_m

    Mu_m_rng = (p * mu_rng - pi * eta_rng + pi_m * Eta_m_rng) / p_m
    L_t = (p_m * Mu_m_rng ** 2 - p * mu_rng ** 2 +
           pi * eta_rng ** 2 - pi_m * Eta_m_rng ** 2)

    return L_t.sum(dim=tuple(range(1, L_t.dim()))) / 2


def bmc(probs, log_evidence, weight, gain_prior, gain_evidence):
    """
    Perform Bayesian model comparison
    :param probs: prior probabilities
    :param log_evidence: log evidence
    :param weight: precision of softmax
    :param gain_prior: gain of prior
    :param gain_evidence: gain of evidence
    :return:
    """
    E = - log_stable(probs) * gain_prior - log_evidence * gain_evidence

    return softmax(-E, weight)


# Shift (D) operator
def shift(array):
    if len(array) > 1:
        return torch.stack((*array[1:], torch.zeros_like(array[0])))
    else:
        return torch.zeros((1, len(array)))