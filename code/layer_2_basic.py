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
    from sklearn.svm import SVC

    from sklearn.pipeline import Pipeline


    ## A first pipeline using globally selected features
    ## and l2 logistic regression
    selector = MultiSelectKBest(f_classif, k=100)
    estimators = [LogisticRegression(C=C, penalty='l2')
                  for C in [10]]#]2. ** np.arange(-24, 0, 2)]

    # estimators = [SVC(C=C, probability=True, kernel="linear")
    #               for C in 2. ** np.arange(-24, 2, 2)]

    first_layer_predictor = CvPredictTransform(model_estimators=estimators)
    pipeline = Pipeline([('feature_reduction', selector),
                         ('first_layer_prediction', first_layer_predictor)])



    def call_pipeline(p, data, stimuli):
        return p.fit_transform(data, stimuli)

    from sklearn.externals.joblib import Memory
    mem = Memory(cachedir="/tmp/joblib")

    call_pipeline = mem.cache(call_pipeline)

    res1 = call_pipeline(pipeline, data, stimuli)

    # Use the layer 1 results to learn a second level classifier on words

    from sklearn.ensemble import ExtraTreesClassifier

    forest = ExtraTreesClassifier(n_estimators=200)

    train_data = res1[:200]
    train_target = stimuli[:200]

    test_data = res1[200:]
    test_target = stimuli[200:]

    forest.fit(train_data, train_target)

    p = np.array(forest.predict_proba(test_data)).T[1]

    # Use the layer 1 results to learn a second level classifier on letters

    forests = [ExtraTreesClassifier(n_estimators=200) for i in range(4)]
    letter_length = stimuli.shape[1] / 4
    predictions = []
    for i, forest in zip(
            range(0, stimuli.shape[1], letter_length),
            forests):
        forest.fit(train_data, train_target[:, i:i + letter_length])
        predictions.append(np.array(forest.predict_proba(test_data)).T[1])
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

    def score_func(y_true, y_pred):
	return 0.5 * ((y_true == y_pred) * y_true).sum() / y_true.sum() +\
            0.5 * ((y_true == y_pred) * (1 - y_true)).sum() / (1 - y_true).sum()
        
    from sklearn.cross_validation import cross_val_score
    scores1 = cross_val_score(forest, res1 > .5, stimuli, cv=6,
                              score_func=score_func)
    #scores2 = cross_val_score(, res1 > .5, stimuli, cv=6,
    #                          score_func=score_func)


    import pylab as pl
    pl.figure()
    pl.imshow(collage)
    pl.gray()

    import scoring
    pl.figure()
    roc1 = scoring.roc(p, stimuli)
    roc2 = scoring.roc(predictions, stimuli)

    pl.plot(roc1, c='b', label = 'Mot entier')
    pl.plot(roc2, c='g', label = 'Lettre par lettre')
    pl.grid()
    pl.title('Résultats de deuxième couche, avec une première couche logistique L2 et k=100 voxels')
    pl.plot([0, len(stimuli)], [0, 1])

pl.show()
