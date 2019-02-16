from utils import *


if __name__ == '__main__':
    for j in range(5):
        print("Repetition {}".format(j + 1))
        data = SyntheticData(n_features=10, sample_size=10000, source_dist="other", mixture="lq")
        res = list()
        for i in range(5):
            estimate = fast_ica(data)
            fastica_mse, fastica_score = save_zigzag2(data.source_train, estimate,
                                                      save_path="fastica_estimate.png")
            string = "FastICA\tmse {0:.3f}\tscore {1:.3f}".format(fastica_mse, fastica_score)
            print(string)
            res.append((fastica_mse, fastica_score))

        mse, score = zip(*res)
        print("mse: {0:.4f}".format(np.mean(np.array(mse))))
        print("score: {0:.4f}".format(np.mean(np.array(score))))
