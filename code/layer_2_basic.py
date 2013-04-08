import numpy as np


if __name__ == "__main__":

    # re-generate layer 1 results
    from layer_1_predictor import CvPredictTransform
    from prepare_data import subjects, get_nii_data, load_stimuli
    _, _, stimuli = load_stimuli()
    data = get_nii_data(subjects[2])

    from multi_select_k_best import MultiSelectKBest
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif

    from sklearn.linear_model import LogisticRegression

    from sklearn.pipeline import Pipeline


    ## A first pipeline using globally selected features
    ## and l2 logistic regression
    selector = MultiSelectKBest(f_classif, k=500)
    estimators = [LogisticRegression(C=C, penalty='l2')
                  for C in 2. ** np.arange(-24, 0, 2)]

    first_layer_predictor = CvPredictTransform(model_estimators=estimators)
    pipeline = Pipeline([('feature_reduction', selector),
                         ('first_layer_prediction', first_layer_predictor)])

    res1 = pipeline.fit_transform(data, stimuli)



    # Use the layer 1 results to learn a second level classifier on words

    from sklearn.ensemble import ExtraTreesClassifier

    forest = ExtraTreesClassifier(n_estimators=200)

    train_data = res1[:200]
    train_target = stimuli[:200]

    test_data = res1[200:]
    test_target = stimuli[200:]

    forest.fit(train_data, train_target)

    p = forest.predict(test_data)

    # Use the layer 1 results to learn a second level classifier on letters

    forests = [ExtraTreesClassifier(n_estimators=200) for i in range(4)]
    letter_length = stimuli.shape[1] / 4
    predictions = []
    for i, forest in zip(
            range(0, stimuli.shape[1], letter_length),
            forests):
        forest.fit(train_data, train_target[:, i:i + letter_length])
        predictions.append(forest.predict(test_data))
    predictions = np.hstack(predictions)

    # visualise the random forests result

    from viz import get_bars, draw_words, pad, make_collage
    bars = get_bars(img_size=(50, 50))
    words_layer1 = draw_words(res1[200:], bars)
    words_forest = draw_words(p, bars)
    words_forest_by_letters = draw_words(predictions, bars)
    words = draw_words(stimuli[200:], bars)

    stacked = np.concatenate([words_layer1, words_forest,
                              words_forest_by_letters, words], axis=1)
    # pad this slightly in order to be able to distinguish groups

    stacked = pad(stacked, [0, 10, 10])

    num_x = 5
    num_y = 8

    start_at = 0

    collage = make_collage(stacked[start_at:start_at + (num_x * num_y)].\
        reshape(num_x, num_y, stacked.shape[1], stacked.shape[2]))

    import pylab as pl
    pl.figure()
    pl.imshow(collage)
    pl.gray()
    pl.show()

