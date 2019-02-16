"""
Created by Tianwen on 2018/10/1
"""


import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils import *


def train(data: SyntheticData, n_components=5, batch_size=512, n_layers=2, renormalize=False,
          n_iters=50000, learning_rate=0.002, c1=0.005, folder="result", figsize=(12, 12)):

    n_sources = data.n_sources
    n_features = data.n_features
    mixture = data.mixture

    source_train = data.source_train
    source_val = data.source_val

    observation_train = data.observation_train
    observation_val = data.observation_val

    multiplier = 1
    activation_function = tf.nn.tanh
    do_whitenning = True

    c2 = 0.1

    # visualization parameters
    eval_freq = 1000
    print_loss = True

    savedir = os.path.join(folder, datetime.strftime(datetime.now(), '%Y-%b-%d_%Hh%Mm%Ss'))
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    filename = os.path.join(savedir, 'log.txt')
    if os.path.isfile(filename):
        print('File {0} already exists. Deleting file...'.format(filename))
        os.remove(filename)
    f = open(filename, mode='x')
    save_params(f,
                # '# Model parameters: ',
                ('mixture', mixture),
                ('n_sources', n_sources),
                ('n_features', n_features),
                ('n_components', n_components),
                ('n_layers', n_layers),
                ('multiplier', multiplier),
                ('activation_function', activation_function.__name__),
                ('do_whitenning', do_whitenning),
                # '# Training parameters: ',
                ('batch_size', batch_size),
                ('num_iterations', n_iters),
                ('learning_rate', learning_rate),
                ('c1', c1),
                ('c2', c2),
                )

    x = tf.placeholder(tf.float32, shape=(batch_size, n_features))
    lr = tf.placeholder(tf.float32)

    # data normalization
    m, v = tf.nn.moments(x, axes=0, keep_dims=True)
    normalized = tf.div(x - m, tf.sqrt(v))
    output = normalized

    with slim.arg_scope([slim.fully_connected], activation_fn=activation_function):
        # encoder network
        if n_layers == 1:
            output = slim.fully_connected(output, num_outputs=64 * multiplier)
        elif n_layers == 2:
            output = slim.fully_connected(output, num_outputs=64 * multiplier)
            output = slim.fully_connected(output, num_outputs=16 * multiplier)
        elif n_layers >= 3:
            output = slim.fully_connected(output, num_outputs=64 * multiplier)
            output = slim.fully_connected(output, num_outputs=32 * multiplier)
            output = slim.fully_connected(output, num_outputs=16 * multiplier)
        ic = slim.fully_connected(output, num_outputs=n_components, activation_fn=None)

        # representation whitening
        if do_whitenning:
            representation = whiten_layer(ic, batch_size)
        else:
            m, v = tf.nn.moments(ic, axes=0, keep_dims=1)
            representation = tf.div(ic - m, tf.sqrt(v))

        # decoder network
        output = representation
        if n_layers == 1:
            output = slim.fully_connected(output, num_outputs=64 * multiplier)
        elif n_layers == 2:
            output = slim.fully_connected(output, num_outputs=64 * multiplier)
            output = slim.fully_connected(output, num_outputs=16 * multiplier)
        elif n_layers >= 3:
            output = slim.fully_connected(output, num_outputs=64 * multiplier)
            output = slim.fully_connected(output, num_outputs=32 * multiplier)
            output = slim.fully_connected(output, num_outputs=16 * multiplier)

        output = slim.fully_connected(output, num_outputs=n_features, activation_fn=None)

    # losses
    cov = tf.matmul(a=representation, b=representation, transpose_a=True, transpose_b=False) / batch_size
    eye = tf.convert_to_tensor(np.eye(n_components, n_components, dtype=np.float32))
    loss_cov = tf.reduce_mean(tf.abs(cov - eye))

    sq = tf.multiply(representation, representation)
    kurtosis = tf.reduce_mean(tf.multiply(sq, sq), axis=0)
    loss_ica = tf.reduce_mean(kurtosis)

    loss_reconstruction = tf.nn.l2_loss(output - normalized) / batch_size

    total_loss = loss_reconstruction + c1 * loss_ica + c2 * loss_cov

    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(total_loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        total_loss_history = list()
        reconstruction_loss_history = list()
        kurtosis_history = list()

        for i in range(n_iters):
            observation_batch = get_random_batch(observation_train, batch_size)

            _, estimate_batch, _total_loss, _loss_ica, _loss_reconstruction, _loss_cov, _cov, _kurt = sess.run(
                [train_op, representation, total_loss, loss_ica, loss_reconstruction, loss_cov, cov, kurtosis],
                feed_dict={x: observation_batch,
                           lr: learning_rate})

            if (i + 1) % eval_freq == 0:
                if print_loss:
                    string = 'Step: {0}\t loss: {1:.3f} = {2:.3f} + {4:.3f}*{3:.2f} + {5:.3f}*{6:.2f}' \
                        .format(i + 1, _total_loss, _loss_reconstruction, _loss_ica, c1, c2, _loss_cov)
                    print(string)
                    f.write(string + '\n')

                kurt = _kurt.tolist()

                total_loss_history.append(_total_loss)
                reconstruction_loss_history.append(_loss_reconstruction)
                kurtosis_history.append(kurt)

                kurt_info="kurtosis: {}".format(list(map(lambda x: "{0:.3f}".format(x), kurt)))
                f.write(kurt_info + "\n")
                estimate_val = validate(source_val, observation_val, batch_size, sess, x, representation)
                mse_val, score_val = save_zigzag2(source_val, estimate_val, renormalize=renormalize,
                                                  step=i, plot_size=250, dir=savedir, figsize=figsize)
                string = 'MSE: {0:.3f}\t Score: {1:.3f}'.format(mse_val, score_val)
                print(string)
                f.write(string + '\n')
                print("=" * 40)
                f.flush()

            if (i + 1) == n_iters // 2:
                learning_rate = 0.001
                print('Learning rate reduced to: {0}'.format(learning_rate))

    std_total_loss = np.std(np.array(total_loss_history[-5:]))
    std_reconstruction_loss = np.std(np.array(reconstruction_loss_history[-5:]))
    std_kurt = np.std(np.array(kurtosis_history[-5:]), axis=0)
    std_kurt_avg = np.mean(std_kurt)
    string_summary = "\nstd total loss: {0:.6f}\n".format(std_total_loss) + \
                     "std reconstruction loss: {0:.6f}\n".format(std_reconstruction_loss) + \
                     "std kurtosis: {0} (avg={1})\n".format(list(map(lambda x: "{0:.6f}".format(x), std_kurt.tolist())),
                                                            std_kurt_avg)
    # print(string_summary)
    f.write(string_summary)
    f.close()
    return mse_val, std_total_loss, std_kurt_avg, reconstruction_loss_history[-1], score_val


if __name__ == '__main__':
    log = "training_log_GPU0.txt"
    w = open(log, 'w', encoding="UTF-8")
    n_layers = 2
    for j in range(3):
        string = "Experiment {0}, generating sources".format(j + 1)
        w.write(string + "\n")
        synthetic_data = SyntheticData(n_features=10, sample_size=512, source_dist="other", mixture="pnl")

        fastica_estimates = fast_ica(synthetic_data)
        for i in range(5):
            fastica_mse, fastica_score = save_zigzag2(synthetic_data.source_train, fastica_estimates, save_path="fastica_estimate.png")
            string = "FastICA\tmse {0:.3f}\tscore {1:.3f}".format(fastica_mse, fastica_score)
            print(string)
            w.write(string + "\n")

        string = "Experiments with num layers={} mixture".format(n_layers)
        print(string)
        w.write(string + "\n")

        res_list = list()
        for i in range(5):
            print("Run {} begins!".format(i + 1))
            result = train(synthetic_data, renormalize=True, n_components=5,
                           n_iters=60000, folder="experiment_lq", n_layers=n_layers)
            res_list.append(result)
            string = "\t".join(map(str, result))
            print(string)

            w.write(string + "\n")
            w.flush()

            #  sort according to mse_val
            sorted_list1 = sorted(res_list, key=lambda x: x[0])
            print("Best result a posteriori:")
            print(sorted_list1[0])

            print("Best result by model selection:")
            model_selection(res_list)

    w.close()
