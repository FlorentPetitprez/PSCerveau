import numpy as np


def fifty_fifty_scoring(y_pred, y_true):
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

    return ((distance_difference > 0).sum(axis=1).astype(np.float64) /
            (len(y_pred) - 1))
