import numpy as np


def fifty_fifty_scoring(y_pred, y_true, sum=True):
    """Compares two by two:
    Prediction A: PA, Prediction B: PB
    True value A: TA, True value B: TB

    Compares hamming distance of (PA, PB) to
    (TA, TB) and to (TB, TA).
    If hamming distance to (TA, TB) smaller,
    then hit, otherwise miss


    Parameters:
    ----------
    y_pred:    nd_array, shape n_samples, n_features
    y_true:    nd_array, shape n_samples, n_features

    """

    all_hamming_distances = (
        (y_pred[:, np.newaxis, :] != y_true[np.newaxis, :, :])
        ).sum(-1)

    true_distances = np.diag(all_hamming_distances)

    distance_difference = ((all_hamming_distances +
                            all_hamming_distances.T) -
                           (true_distances[:, np.newaxis] +
                            true_distances[np.newaxis, :]))

    res = ((distance_difference > 0).sum(axis=1).astype(np.float64) /
            (len(y_pred) - 1))

    if sum:
        return res.mean()
    else:
        return res


def likelihoods(y_pred, y_true):
    """
    Calculates for each pair in y_pred, y_true the likelihood that
    the prediction represents the true value

    Parameters:
    -----------

    y_pred: ndarray, shape=(n_predictions, n_features)
    y_true: ndarray, shape=(n_true, n_features)

    output:
    likelihoods: ndarray, shape=(n_predictions, n_features)
    """

    adjusted_probabilities = np.abs((1. - y_true)[np.newaxis, :, :] -
                                    y_pred[:, np.newaxis, :])

    cumulated = np.prod(adjusted_probabilities, axis=-1)

    return cumulated


def rankings(y_pred, y_true):

    l = likelihoods(y_pred, y_true)

    all_ranks = l.argsort(axis=1)[::-1]
    placement_bin = all_ranks == np.arange(len(y_pred))[:, np.newaxis]

    placement_number = (np.arange(len(y_true), dtype=np.int64)[np.newaxis, :] *
                        np.ones([len(y_pred), 1], dtype=np.int64))[placement_bin]

    return placement_number


def roc(y_pred, y_true):

    r = rankings(y_pred, y_true)

    counts_bin = r[:, np.newaxis] <= np.arange(len(y_true))[np.newaxis, :]

    counts = counts_bin.sum(0)

    percentages = counts / np.float64(len(y_pred))

    return percentages


def auc(y_pred, y_true):

    rc = roc(y_pred, y_true)

    return rc.sum() / np.float64(len(rc))


