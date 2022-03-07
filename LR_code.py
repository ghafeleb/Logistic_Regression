#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ali Ghafelebashi

"""
import numpy as np
import os
import itertools
from collections import defaultdict
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import argparse

np.set_printoptions(precision=3, linewidth=240, suppress=True)
np.random.seed(2021)

###Function to download and pre-process (normalize, PCA) mnist and store in "data" folder:
##Returns 4 arrays: train/test_features_by_machine = , train/test_labels_by_machine
def load_MNIST2(p, dim, path, q):
    if 'data' not in os.listdir('./'):
        os.mkdir('./data')
    if path not in os.listdir('./data'):
        os.mkdir('./data/' + path)
    # if data folder is not there, make one and download/preprocess mnist:
    if 'processed_mnist_features_{:d}_q{:d}.npy'.format(dim, q) not in os.listdir('./data/' + path):
        # convert image to tensor and normalize (mean 0, std dev 1):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.,), (1.,)), ])
        # download and store transformed dataset as variable mnist:
        mnist = datasets.MNIST('data', download=True, train=True, transform=transform)
        # separate features from labels and reshape features to be 1D array:
        features = np.array([np.array(mnist[i][0]).reshape(-1) for i in range(len(mnist))])
        labels = np.array([mnist[i][1] for i in range(len(mnist))])
        # apply PCA to features to reduce dimensionality to dim
        features = PCA(n_components=dim).fit_transform(features)
        # save processed features in "data" folder:
        np.save('data/' + path + 'processed_mnist_features_{:d}_q{:d}.npy'.format(dim, q), features)
        np.save('data/' + path + 'processed_mnist_features_{:d}_q{:d}.npy'.format(dim, q), labels)
    # else (data is already there), load data:
    else:
        features = np.load('data/' + path + 'processed_mnist_features_{:d}_q{:d}.npy'.format(dim, q))
        labels = np.load('data/' + path + 'processed_mnist_features_{:d}_q{:d}.npy'.format(dim, q))

    ## Group the data by digit
    n_m = int(min([np.sum(labels == i) for i in range(10)]) * q)  # smaller scale version
    # use defaultdict to avoid key error https://stackoverflow.com/questions/5900578/how-does-collections-defaultdict-work:
    by_number = defaultdict(list)
    # append feature vectors to by_number until there are n_m of each digit
    for i, feat in enumerate(features):
        if len(by_number[labels[i]]) < n_m:
            by_number[labels[i]].append(feat)
    # convert each list of n_m feature vectors (for each digit) in by_number to np array
    for i in range(10):
        by_number[i] = np.array(by_number[i])

    ## Enumerate the even vs. odd tasks
    even_numbers = [0, 2, 4, 6, 8]
    odd_numbers = [1, 3, 5, 7, 9]
    # make list of all 25 pairs of (even, odd):
    even_odd_pairs = list(itertools.product(even_numbers, odd_numbers))

    ## Group data into 25 single even vs single odd tasks
    all_tasks = []
    for (e, o) in even_odd_pairs:
        # eo_feautres: concatenate even feats, odd feats for each e,o pair:
        eo_features = np.concatenate([by_number[e], by_number[o]], axis=0)
        # (0,...,0, 1, ... ,1) labels of length 2*n_m corresponding to eo_features:
        eo_labels = np.concatenate([np.ones(n_m), np.zeros(n_m)])
        # concatenate eo_feautures and eo_labels into array of length 4*n_m:
        eo_both = np.concatenate([eo_labels.reshape(-1, 1), eo_features], axis=1)
        # add eo_both to all_tasks:
        all_tasks.append(eo_both)
    # all_tasks is a list of 25 ndarrays, each array corresponds to an (e,o) pair of digits (aka task) and is 10,842 (examples) x 101 (100=dim (features) plus 1 =dim(label))
    # all_evens: concatenated array of 5*n_m ones and 5*n_m = 27,105 feauture vectors (n_m for each even digit):
    all_evens = np.concatenate([np.ones((5 * n_m, 1)), np.concatenate([by_number[i] for i in even_numbers], axis=0)],
                               axis=1)
    # all_odds: same thing but for odds and with zeros instead of ones:
    all_odds = np.concatenate([np.zeros((5 * n_m, 1)), np.concatenate([by_number[i] for i in odd_numbers], axis=0)],
                              axis=1)
    # combine all_evens and _odds into all_nums (contains all 10*n_m = 54210 training examples):
    all_nums = np.concatenate([all_evens, all_odds], axis=0)

    ## Mix individual tasks with overall task
    # each worker m gets (1-p)* 2*n = (1-p)*10,842 examples from specific tasks and p*10,842 from mixture of all tasks.
    # So p=1 -> homogeneous (zeta = 0); p=0 -> heterogeneous
    features_by_machine = []
    labels_by_machine = []
    n_individual = int(np.round(2 * n_m * (1. - p)))  # int (1-p)*2n_m = (1-p)*10,842
    n_all = 2 * n_m - n_individual  # =int p*2n_m  = p*10,842
    for m, task_m in enumerate(all_tasks):  # m is btwn 0 and 24 inclusive
        task_m_idxs = np.random.choice(task_m.shape[0],
                                       size=n_individual)  # specific: randomly choose (1-p)*2n_m examples from 2*n_m = 10,842 examples for task m (one (e,o) pair)
        all_nums_idxs = np.random.choice(all_nums.shape[0],
                                         size=n_all)  # mixture of tasks: randomly choose p*2n_m examples from all 54,210 examples (all digits)
        data_for_m = np.concatenate([task_m[task_m_idxs, :], all_nums[all_nums_idxs, :]],
                                    axis=0)  # pair m gets 10,842 total examples: fraction p are mixed, 1-p are specific to task m (one eo pair)
        features_by_machine.append(data_for_m[:, 1:])
        labels_by_machine.append(data_for_m[:, 0])
    features_by_machine = np.array(
        features_by_machine)  # array of all 25 feauture sets (each set has 10,842 feauture vectors)
    labels_by_machine = np.array(labels_by_machine)  # array of corresponding label sets
    ###Train/Test split for each pair###
    train_features_by_machine = []
    test_features_by_machine = []
    train_labels_by_machine = []
    test_labels_by_machine = []
    for m, task_m in enumerate(all_tasks):
        train_feat, test_feat, train_label, test_label = train_test_split(features_by_machine[m], labels_by_machine[m],
                                                                          test_size=0.20, random_state=1)
        train_features_by_machine.append(train_feat)
        test_features_by_machine.append(test_feat)
        train_labels_by_machine.append(train_label)
        test_labels_by_machine.append(test_label)
    train_features_by_machine = np.array(train_features_by_machine)
    test_features_by_machine = np.array(test_features_by_machine)
    train_labels_by_machine = np.array(train_labels_by_machine)
    test_labels_by_machine = np.array(test_labels_by_machine)
    print(train_features_by_machine.shape)
    return train_features_by_machine, train_labels_by_machine, test_features_by_machine, test_labels_by_machine, n_m


############################################## Logistic Regression ###############################################

def sigmoid(z):
    return 1. / (1. + np.exp(-np.clip(z, -15, 15)))  # input is clipped i.e. projected onto [-15,15].


def logistic_loss(w, features, labels):  # returns average val of log loss over data = features, labels
    probs = sigmoid(np.dot(features, w))
    return (-1. / features.shape[0]) * (np.dot(labels, np.log(1e-12 + probs)) + np.dot(1 - labels, np.log(
        1e-12 + 1 - probs)))  # vectorized empirical loss with 1e-12 to avoid log(0)


def logistic_loss_gradient(w, features, labels):
    return np.dot(np.transpose(features), sigmoid(np.dot(features, w)) - labels) / features.shape[
        0]  # dot here is used for matrix mult. result is d-vector

def test_err(w, features, labels, pair_idx):  # computes prediction error given a parameter w and data = features, labels
    errors = 0  # count number of errors
    test_size = labels.shape[0] * labels.shape[1]  # total number of test examples across all clients

    for j in range(features.shape[1]):
        prob = sigmoid(np.dot(features[pair_idx, j], w))
        if prob > 0.5:
            prediction = 1
        else:
            prediction = 0
        if prediction != labels[pair_idx, j]:
            errors += 1
    return errors / test_size

############################################## Mini-batch SGD ###############################################
def minibatch_sgd(x_len, batch_size, n_epoch, stepsize, loss_freq, f_eval, grad_eval, pair_idx=0, avg_window=1):
    losses = []
    iterates = np.zeros(x_len)
    for r in range(n_epoch):
        g = np.zeros(x_len)  # start with g = 0 vector of dim x_len = 100
        g += grad_eval(iterates, batch_size, pair_idx)  # evaluate stoch grad of log loss at last iterate
        iterates = iterates - stepsize * g  # take SGD step and add new iterate to list iterates
        if (r + 1) % loss_freq == 0:
            losses.append(f_eval(iterates))  # evalute f (at average of last 7 iterates) every loss_freq rounds and append to list "losses"
            print('Iteration: {:d}/{:d}   Loss: {:f}                 \r'.format(r + 1, n_epoch, losses[-1]), end='')
            if losses[-1] > 100:
                print('\nLoss is diverging: Loss = {:f}'.format(losses[-1]))
                return iterates, losses, 'diverged'
    print('')
    return iterates, losses, 'converged'
##################################################################################################################


def LogisticRegression(args, train_features, train_labels, my_test=False):
    pair_idx = args.pair_idx
    batch_size = args.batch_size
    n_epoch = args.n_epoch
    n_stepsize = args.n_stepsize
    q = args.q

    loss_freq = 5 # Frequency of checking loss for printing loss

    # keep track of train excess risk too
    if my_test:
        dim = 50 # Dimension of input data
        p = 0.0  # for full heterogeneity
        path = 'shipt_mnist_p={:.2f}_K={:d}_R={:d}'.format(p, batch_size, n_epoch)
        train_features, train_labels, test_features, test_labels, n_m = load_MNIST2(p, dim, path, q)  # number of examples (train and test) per digit per pair
        n = int(n_m * 2 * 0.8)  # total number of TRAINING examples (two digits) per pair
        print(f"Total number of TRAINING examples (two digits) per pair: {n}")
    else:
        train_features, train_labels = np.array([np.array(train_features)]), np.array([np.array(train_labels)])
        dim = train_features[0].shape

    if batch_size==0:
        batch_size = train_features[pair_idx].shape[0]

    x_len = train_features.shape[2]  # dim of data

    def f_eval(w):
        return logistic_loss(w, train_features.reshape(-1, x_len), train_labels.reshape(-1))

    def grad_eval(w, minibatch_size, m):  # stochastic (minibatch) grad eval
        idxs = np.random.randint(0, train_features[m].shape[0],
                                 minibatch_size)  # draw minibatch (unif w/replacement) index set of size minibatch_size from pair m's dataset
        return logistic_loss_gradient(w, train_features[m, idxs, :], train_labels[
            m, idxs])  # returns average gradient across minibatch of size minibatch_size (=batch_size)

    lg_stepsizes = [np.exp(exponent) for exponent in np.linspace(-6, 0, n_stepsize)]  # MB SGD
    print('Running Minibatch SGD')  # for each stepsize option, compute average excess risk of MBSGD over n_reps trials
    MB_results = np.zeros((n_epoch // loss_freq, len(lg_stepsizes)))
    MB_w = [np.zeros(dim)] * len(lg_stepsizes)
    for i, stepsize in enumerate(lg_stepsizes):
        print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize, i + 1, len(lg_stepsizes)))
        iterates, l, success = minibatch_sgd(x_len, batch_size, n_epoch, stepsize, loss_freq, f_eval, grad_eval)
        if success == 'converged':
            MB_w[i] += iterates
        else:
            MB_results[:, i] += 100
    MB_step_index = np.argmin(MB_results, axis=1)
    final_MB_step_index = MB_step_index[-1]

    if my_test:
        MB_test_error = test_err(MB_w[final_MB_step_index], test_features, test_labels, pair_idx)
    MB_train_error = test_err(MB_w[final_MB_step_index], train_features, train_labels, pair_idx)

    print("MB train error", MB_train_error)
    print("MB test error", MB_test_error)
    return MB_w[final_MB_step_index]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair_idx', type=int, default=0, help="Index of pair of digits for MNIST data, defaul=0 means the pair (0, 1)")
    parser.add_argument('--batch_size', type=int, default=0, help="Batch size in Mini-batch SGD, default=0 means full batch (i.e. GD)")
    parser.add_argument('--n_epoch', type=int, default=25, help="Number of epochs")
    parser.add_argument('--n_stepsize', type=int, default=10, help="Number of stepsizes used in hyperparameter tuning")
    parser.add_argument('--q', type=int, default=1, help="Fraction of mnist data we wish to use; q = 1 -> 8673 train examples per pair; q = 1/10 -> 867 train examples per pair")
    args = parser.parse_args()
    beta_hat = LogisticRegression(args, [], [], True)
