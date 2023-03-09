import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def plot_history(train_history, val_metrics, title='loss'):
    plt.figure(figsize=(20, 6))
    plt.subplot(1, 2, 1)
    plt.title('{}'.format(title))
    plt.plot(train_history, label='train', zorder=1)
    plt.xlabel('train steps')
    plt.semilogy()
    plt.ylabel('train loss')
    plt.legend(loc='best')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.xticks(np.arange(len(val_metrics)))
    plt.title("Square error distribution per position in prompt")
    plt.ylabel("Err distribution")
    plt.xlabel("Prompt position")
    x = np.arange(len(val_metrics))
    if len(val_metrics.shape) > 1:
        plt.plot(x, val_metrics[:, 0], 'o', color="tab:orange")
        plt.fill_between(x, val_metrics[:, 1],
                         val_metrics[:, 2], alpha=0.3,
                         edgecolor='#CC4F1B', facecolor='#FF9848')
    else:
        plt.plot(x, val_metrics, 'o', color="tab:orange")
    plt.show()


def mse(pred, target):
    return torch.pow((target - pred), 2).mean()


def metrics_bootstrap_smoothing(res):
    batch_size = res.shape[0]
    resample_idx = torch.randint(batch_size, size=(1000, batch_size))
    bootstrap_means = res[resample_idx].mean(dim=1).sort(dim=0)[0]
    metrics = torch.zeros(res.shape[1], 3)
    metrics[:, 0] = res.mean(dim=0)
    metrics[:, 1] = bootstrap_means[50, :]
    metrics[:, 2] = bootstrap_means[950, :]
    return metrics


def load_mnist(flatten=False):
    """taken from https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py"""
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test
