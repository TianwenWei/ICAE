import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
import matplotlib
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.decomposition import FastICA, PCA
import time

matplotlib.rcParams.update({'font.size': 18})


class SyntheticData:
    def __init__(self, n_features, sample_size, source_dist, mixture):
        validation_size = 1024

        self.n_sources = 5
        self.n_features = n_features
        self.source_dist = source_dist
        self.mixture = mixture
        self.mixing_matrix_1 = np.random.randn(self.n_sources, self.n_features) / np.sqrt(self.n_features)
        if mixture == "mlp":
            self.mixing_matrix_1 = np.random.randn(self.n_sources, self.n_features // 2) / np.sqrt(self.n_features // 2)
            self.mixing_matrix_2 = np.random.randn(self.n_features // 2, self.n_features) / np.sqrt(self.n_features)
        else:
            self.mixing_matrix_1 = np.random.randn(self.n_sources, self.n_features) / np.sqrt(self.n_features)
            self.mixing_matrix_2 = np.random.randn(self.n_features, self.n_sources, self.n_sources) / np.sqrt(n_features)

        self.sources = generate_sources(sample_size=validation_size + sample_size, dist=source_dist)
        self.observation = apply_mixture(self.sources, self.mixture, self.mixing_matrix_1, self.mixing_matrix_2)

        self.source_val = self.sources[:validation_size, :]
        self.source_train = self.sources[validation_size:, :]

        self.observation_val = self.observation[:validation_size, :]
        self.observation_train = self.observation[validation_size:, :]


def save_params(f, *args):
    f.write('=' * 20 + ' Parameters ' + '=' * 20 + '\n')
    for arg in args:
        if isinstance(arg, tuple) and len(arg) == 2:
            string = '{0} = {1}'.format(arg[0], arg[1])
            print(string)
            f.write(string + '\n')
        else:
            print(arg)
            f.write('\n' + arg + '\n')
    f.write('=' * 52 + '\n')


def symsqrt(mat, eps=1e-6):
    """Symmetric square root."""
    s, u, v = tf.svd(mat)
    # sqrt is unstable around 0, just use 0 in such case
    si = tf.where(tf.less(s, eps), 1e3 * tf.ones_like(s), tf.div(1.0, tf.sqrt(s)))
    return u @ tf.diag(si) @ tf.transpose(v)


def symsqrt_t(mat, eps=1e-6):
    """Symmetric square root."""
    s, u, v = tf.svd(mat)
    # sqrt is unstable around 0, just use 0 in such case
    si = tf.where(tf.less(s, eps), np.sqrt(1 / eps) * tf.ones_like(s), tf.div(1.0, tf.sqrt(s)))
    return u @ tf.diag(si) @ tf.transpose(v)


def whiten_layer(input_mat, batch_size):
    m = tf.reduce_mean(input_mat, axis=0, keep_dims=True)
    centered = input_mat - m
    cov = tf.matmul(a=centered, b=centered, transpose_a=True, transpose_b=False) / batch_size
    return tf.matmul(a=centered, b=tf.stop_gradient(symsqrt(cov)))


def whiten_layer_t(input_mat, batch_size, eps=1e-6):
    m = tf.reduce_mean(input_mat, axis=0, keep_dims=True)
    centered = input_mat - m
    cov = tf.matmul(a=centered, b=centered, transpose_a=True, transpose_b=False) / batch_size
    return tf.matmul(a=centered, b=tf.stop_gradient(symsqrt_t(cov, eps)))


def check_source(idx, input_source):
    if idx < 2:
        source_idx = 0
    else:
        source_idx = 1

    if idx % 2 == 0:
        source_sign = 1
    else:
        source_sign = -1

    return input_source[:, source_idx] * source_sign


def random_orthogonal_matrix(d):
    M = np.random.randn(d, d)
    U, D, V = np.linalg.svd(M @ np.transpose(M))
    N = U @ np.diag(1/np.sqrt(D)) @ V @ M
    assert(np.linalg.norm(N @ np.transpose(N) - np.eye(d, d)) < 1e-8)
    return N


def generate_single_source(sample_size, dist):
    if (dist == 'uniform') or (dist == 'u'):  # kurtosis = 1.8
        return 2 * np.sqrt(3) * np.random.rand(sample_size) - np.sqrt(3)
    elif (dist == 'normal') or (dist == 'n') or (dist == 'g'):  # kurtosis = 3
        return np.random.rand(sample_size)
    elif (dist == 'laplace') or (dist == 'l'):  # kurtosis = 6
        return np.random.laplace(loc=0, scale=1, size=sample_size) * np.sqrt(1 / 2)
    elif (dist == 'triangular') or (dist == 't'):  # kurtosis = 2.4
        return np.random.triangular(-np.sqrt(6), 0, np.sqrt(6), size=sample_size)
    elif (dist == 'bernoulli') or (dist == 'b'):  # kurtosis = 1
        return np.random.choice([1.0, -1.0], sample_size)
    else:
        raise ValueError('Unknown source distribution.')


def generate_uniform_batch(sample_size, number_of_sources):
    return 2 * np.sqrt(3) * np.random.rand(sample_size, number_of_sources) - np.sqrt(3)


def generate_sources(sample_size=10000, dist="uniform", seed=None):
    if seed is not None:
        np.random.seed(seed)

    if dist == "uniform":
        print("Generating five uniform sources...")
        return 2 * np.sqrt(3) * np.random.rand(sample_size, 5) - np.sqrt(3)
    else:
        print("Generating five high frequency sources...")
        t = np.linspace(0, sample_size * 1e-4, sample_size)
        two_pi = 2 * np.pi
        s0 = np.sign(np.cos(two_pi * 155 * t))
        s1 = np.sin(two_pi * 800 * t)
        s2 = np.sin(two_pi * 300 * t + 6 * np.cos(two_pi * 60 * t))
        s3 = np.sin(two_pi * 90 * t)
        s4 = np.random.uniform(-1, 1, (sample_size,))
        # s5 = np.random.laplace(0, 1, (sample_size,))
        # x = np.stack([s0, s1, s2, s3, s4, s5])
        x = np.stack([s0, s1, s2, s3, s4])
        x_mean = np.mean(x, axis=1, keepdims=True)
        # print("Mean of the original sources: {}".format(list(map(lambda a: "{0:.3f}".format(a), x_mean[:, 0]))))
        x_centered = x - x_mean
        x_std = np.std(x_centered, axis=1, keepdims=True)
        # print("Std of the original sources: {}".format(list(map(lambda a: "{0:.3f}".format(a), x_std[:, 0]))))
        x_normalized = x_centered / x_std
        x_p2 = x_normalized * x_normalized
        x_p4 = x_p2 * x_p2
        kurtosis = np.mean(x_p4, axis=1)
        print("kurtosis of the normalized sources: {}".format(list(map(lambda a: "{0:.3f}".format(a), kurtosis))))
    return x_normalized.T


def generate_mix_batch(sample_size, number_of_sources):
    output = np.zeros((sample_size, number_of_sources))
    output[:, 0] = generate_single_source(sample_size, dist='u')
    output[:, 1] = generate_single_source(sample_size, dist='n')
    output[:, 2] = generate_single_source(sample_size, dist='l')
    output[:, 3] = generate_single_source(sample_size, dist='t')
    output[:, 4] = generate_single_source(sample_size, dist='b')
    return output


def _identity(x):
    return x


def _power(x):
    return np.power(x, 3)


def _sigmoid(x):
    temp = np.exp(x)
    return temp / (1 + temp)


def _tanh(x):
    temp = np.exp(2 * x)
    return (temp - 1) / (temp + 1)


def _exp(x):
    return np.exp(x / 3)


def _softplus(x):
    return np.log(1 + np.exp(x))


def apply_nonlinearity(x):
    _output = np.zeros_like(x)
    for i in range(x.shape[1]):
        if i % 5 == 0:
            _output[:, i] = _power(x[:, i])
        elif i % 5 == 1:
            _output[:, i] = _sigmoid(x[:, i])
        elif i % 5 == 2:
            _output[:, i] = _exp(x[:, i])
        elif i % 5 == 3:
            _output[:, i] = _tanh(x[:, i])
        else:
            # _output[:, i] = x[:, i]
            _output[:, i] = _softplus(x[:, i])
    return _output


def get_random_batch(array, batch_size):
    indices = np.random.randint(array.shape[0], size=(batch_size,))
    return array[indices, :]


def apply_mixture(s, mixture_type, A, B=None):
    mixture_type = mixture_type.lower()
    if mixture_type == 'linear':
        x = s @ A
    elif mixture_type == 'pnl':
        t = s @ A
        x = apply_nonlinearity(t)
    elif mixture_type == 'lq':
        x = LQ_mixture(s, A, B)
    elif mixture_type == 'lq-pnl':
        t = LQ_mixture(s, A, B)
        x = apply_nonlinearity(t)
    elif mixture_type == 'mlp':
        t1 = s @ A
        t2 = apply_nonlinearity(t1)
        t3 = t2 @ B
        x = apply_nonlinearity(t3)
    else:
        raise ValueError('Cannot recognize mixture type!')
    return x


def nonlinear_transform(x, *mats):
    for j in range(len(mats)):
        x = x @ mats[j]
        x = apply_nonlinearity(x)
    return x


def LQ_mixture(sources, mat, tensor):
    feature_dim = mat.shape[1]
    batch_size = sources.shape[0]
    L = sources @ mat
    T = np.tensordot(sources, tensor, axes=(1, 2))
    Q = np.zeros((batch_size, feature_dim))
    for i in range(feature_dim):
        Q[:, i] = np.mean(T[:, i, :] * sources, axis=1)
    mixture = L + Q
    return mixture


def convert_index(idx, ncol):
    row_index = idx // ncol
    col_index = idx % ncol
    return row_index, col_index


def validate(s_val, x_val, batch_size, sess, input_tensor, rep_tensor):
    n_components = rep_tensor.get_shape().as_list()[1]
    sample_size = s_val.shape[0]
    num_batches = sample_size // batch_size
    estimates = np.zeros((sample_size, n_components))
    for i in range(num_batches):
        estimates_batch = sess.run(rep_tensor, feed_dict={input_tensor: x_val[i * batch_size: (i + 1) * batch_size, :]})
        estimates[i * batch_size: (i + 1) * batch_size, :] = estimates_batch
    return estimates


def fast_ica(data: SyntheticData):
    mixture = data.observation_train
    n_components = data.n_sources

    # set whiten=True for PCA to obtain PC with unit variance
    pca = PCA(n_components=n_components, whiten=True)
    principal_components = pca.fit_transform(mixture)
    # set whiten=False for ICA because there seems to be a bug
    ica = FastICA(max_iter=100, whiten=False)
    estimated_sources = ica.fit_transform(principal_components)  # Reconstruct signals
    return estimated_sources


def find_match_new(original, estimate):
    sample_size = original.shape[0]
    n_sources = original.shape[1]
    n_estimates = estimate.shape[1]
    corr = np.transpose(original) @ estimate / sample_size  # shape = (n_sources, n_estimates)
    abs_corr = np.abs(corr)
    if n_estimates < n_sources:
        N = n_estimates
    else:
        N = n_sources
    match = np.zeros((N, 3), dtype=np.int32)
    mse = np.zeros(N)
    score = np.zeros(N)
    for i in range(N):
        position = np.argmax(abs_corr)
        source_idx = position // n_estimates
        estimate_idx = position % n_estimates
        score[i] = np.abs(corr[source_idx, estimate_idx])

        sign = np.sign(corr[source_idx, estimate_idx])
        match[i, :] = [source_idx, estimate_idx, sign]
        fnorm = np.linalg.norm(original[:, source_idx] - estimate[:, estimate_idx] * sign)
        mse[i] = np.power(fnorm, 2) / sample_size
        abs_corr[source_idx, :] = 0
        abs_corr[:, estimate_idx] = 0
    return np.mean(mse), np.mean(score), match


def save_zigzag2(sources, estimates, step=0, plot_size=250, dir='imgs', save_path=None,
                 figsize=(12, 12), linewidth=2, title='Estimated Components', color='grayscale'):
    if color == "grayscale":
        plot_style1 = "k-"
        plot_style2 = "k--"
    else:
        plot_style1 = "b-"
        plot_style2 = "r-"

    m_s = np.mean(sources, axis=0, keepdims=True)
    m_r = np.mean(estimates, axis=0, keepdims=True)

    std_s = np.std(sources, axis=0, keepdims=True)
    std_r = np.std(estimates, axis=0, keepdims=True)

    normalized_sources = (sources - m_s) / std_s
    normalized_estimates = (estimates - m_r) / std_r

    n_sources = sources.shape[1]
    n_components = estimates.shape[1]

    mse, score, match = find_match_new(normalized_sources, normalized_estimates)
    # print("MSE: {0:.4f}".format(mse))
    seq = np.arange(0, plot_size)
    plt.close("all")
    plt.figure(figsize=figsize)
    estimate_indices = set()
    for j in range(match.shape[0]):
        source_idx = match[j, 0]
        estimate_idx = match[j, 1]
        estimate_indices.add(estimate_idx)
        sign = match[j, 2]
        plt.subplot(n_components, 1, j + 1)

        if j == 0:
            plt.title(title)

        plt.plot(seq, normalized_sources[:plot_size, source_idx], '{}'.format(plot_style1), linewidth=linewidth)
        plt.plot(seq, normalized_estimates[:plot_size, estimate_idx] * sign, '{}'.format(plot_style2), linewidth=linewidth)

        plt.plot(seq, np.zeros(plot_size), 'k:', linewidth=1)

        plt.ylim([-2.5, 2.5])
        plt.xlim([1, plot_size - 1])
        plt.xticks([], [])
        plt.ylabel('IC{0:d}'.format(j + 1))
        # if source_idx == 0:
        #     plt.title(title)
        # if source_idx == n_sources - 1:  # previously it was 4
        #     plt.xlabel('sample index')

    if n_components > n_sources:
        garbage_components = set(range(n_components)) - estimate_indices
        # if len(garbage_components) != n_components - n_sources:
        #     print("Identified {} sources from the estimates".format(len(estimate_indices)))
        i = 0
        for idx in garbage_components:
            plt.subplot(n_components, 1, n_sources + i + 1)
            plt.plot(seq, estimates[:plot_size, idx], '{}--'.format(plot_style2), linewidth=linewidth)
            plt.plot(seq, np.zeros(plot_size), 'k:', linewidth=1)
            plt.ylim([-2.5, 2.5])
            plt.xticks([], [])
            plt.xlim([1, plot_size - 1])
            plt.ylabel('IC{0:d}'.format(n_sources + i + 1))
            i += 1
    plt.xticks(list(range(0, plot_size + 1, 50)), list(map(str, range(0, plot_size + 1, 50))))

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.savefig(dir + '\\zigzag{0:05d}.png'.format(step + 1))
    return mse, score


def model_selection(result):
    # mse, std_loss, std_kurt, recon_loss, score
    array = np.array(result)
    if np.min(array[:, 1]) < 0.005:
        subarr = array[array[:, 1] < 0.005, :]
    elif np.min(array[:, 1]) < 0.01:
        subarr = array[array[:, 1] < 0.01, :]
    else:
        subarr = array
    lst = subarr.tolist()
    sorted_list = sorted(lst, key=lambda x: x[2])
    print(sorted_list[0])
    return sorted_list[0]