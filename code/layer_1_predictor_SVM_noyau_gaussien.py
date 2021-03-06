import numpy as np
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.base import BaseEstimator
import scoring

def cv_predict_transform(model_estimators, data, targets,
                         prediction_function="predict_proba",
                         verbose=100):

    # We have 6 sessions and will try to predict each one of
    # them using the best of the model_estimators, which are
    # learnt on the 5 others.
    # model_estimators should be a list of estimators of the
    # same type, but varying parameters

    outer_cv = KFold(len(data), 6, indices=True)

    all_scores = []
    all_predictions = []

    for outer_train, outer_val in outer_cv:

        if verbose >= 1:
            print "Working on fold %d - %d" % (outer_val[0], outer_val[-1])

        # We have 5 sessions left on which to cross_validate
        # and will thus cut the data into 5 folds

        inner_cv = KFold(len(outer_train), 5)

        # now try out all the model estimators:
        model_scores = []

        for estimator in model_estimators:

            if verbose >= 10:
                print "Working on estimator %s" % (repr(estimator))

            # we suppose the estimator is single target, so we
            # loop over the targets, but we can easily add a
            # a multitarget option in the signature of the function
            # if necessary

            scores = []

            for t, target in enumerate(targets[outer_train].T):
                if verbose >= 100:
                    print "treating target %d" % t

                score = cross_val_score(estimator,
                                        data[outer_train],
                                        target,
                                        cv=inner_cv)
                scores.append(score)

            model_scores.append(scores)

        model_scores = np.array(model_scores)
        all_scores.append(model_scores)

        # now perform the prediction on the left out outer fold using
        # the best estimator per target

        best_model_per_target = model_scores.mean(-1).argmax(0)

        predictions = []
        if verbose >= 5:
            print "Performing predictions"
        for t, (target, best_model) in enumerate(
                zip(targets[outer_train].T,
                best_model_per_target)):

            if verbose >= 100:
                print "Predicting target %d" % t
            est = model_estimators[best_model]
            '''
            best_gamma=10.0**(-5+best_model%11)
            best_C=10.0**(-5+(best_model-best_model%11)/11)
            print "best_gamma=", best_gamma, ", best_C=", best_C
            '''
            est.fit(data[outer_train], target)

            if prediction_function == "predict_proba":
                # weird predict_proba gives back (1-p) and p
                # choose only p
                predictions.append(
                est.predict_proba(data[outer_val])[:, 1:2])
            else:
                predictions.append(est.predict(data[outer_val])[:, np.newaxis])

        predictions = np.hstack(predictions)
        all_predictions.append(predictions)

    all_predictions = np.array(all_predictions)
    all_predictions = all_predictions.reshape(-1, all_predictions.shape[-1])
    all_scores = np.array(all_scores)

    return all_predictions, all_scores


# We will create a scikit-learn type estimator wrapping cv_predict_transform
# in order to be able to use it in pipelines

class CvPredictTransform(BaseEstimator):

    def __init__(self, model_estimators,
                 prediction_function="predict",
                 verbose=10):
        self.model_estimators = model_estimators
        self.prediction_function = prediction_function
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def transform(self, X):
        raise Exception("Must call fit_transform")

    def fit_transform(self, X, y):
        predictions, scores = cv_predict_transform(self.model_estimators, X, y,
                                    self.prediction_function,
                                    self.verbose)

        self.scores = scores
        self.predictions = predictions
        return predictions, scores


if __name__ == "__main__":

    from prepare_data import subjects, get_nii_data, load_stimuli
    _, _, stimuli = load_stimuli()
    data = get_nii_data(subjects[0])

    from multi_select_k_best import MultiSelectKBest
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif

    from sklearn.svm import SVC

    from sklearn.pipeline import Pipeline


    ## A first pipeline using globally selected features
    ## and Radial Basis Function (RBF) Kernel
    selector = MultiSelectKBest(f_classif, k=500)
    estimators = []
    for C in np.arange(0.01, 1.1, 0.1): 
        for gamma in np.arange(0.01, 1.1, 0.1):
            estimators.append(SVC(C=C, gamma=gamma))
                  

    first_layer_predictor = CvPredictTransform(model_estimators=estimators)
    pipeline = Pipeline([('feature_reduction', selector),
                         ('first_layer_prediction', first_layer_predictor)])

    res1,scores1 = pipeline.fit_transform(data, stimuli)
    print "score1=", np.mean(scores1)
    
    ## A second pipeline using features selected per bar
    estimators2 = []
    for C in np.arange(0.01, 1.1, 0.1):
        for gamma in np.arange(0.01, 1.1, 0.1):
            estimators2.append(Pipeline([
            ('feature_selection', SelectKBest(f_classif, k=100)),
            ('SVM_noyeau_gaussien', SVC(C=C, gamma=gamma))]))


    first_layer_predictor2 = CvPredictTransform(model_estimators=estimators2)

    # this method is slow, because it keeps calling a feature reduction
    # method for each bar and each estimator. We will globally reduce the
    # features before starting

    global_f_select = MultiSelectKBest(f_classif,
                                       pooling_function=np.min,
                                       k=3000)

    res2,scores2 = first_layer_predictor2.fit_transform(
        global_f_select.fit_transform(data, stimuli), stimuli)
    print "score2=", np.mean(scores2)

    
    # Now visualise the predictions.
    from viz import get_bars, draw_words, pad, make_collage
    bars = get_bars(img_size=(50, 50))
    words1 = draw_words(res1, bars)
    words2 = draw_words(res2, bars)
    words = draw_words(stimuli, bars)

    stacked = np.concatenate([words1, words2, words], axis=1)
    # pad this slightly in order to be able to distinguish groups

    stacked = pad(stacked, [0, 10, 10])

    num_x = 8
    num_y = 12

    start_at = 0

    collage = make_collage(stacked[start_at:start_at + (num_x * num_y)].\
        reshape(num_x, num_y, stacked.shape[1], stacked.shape[2]))

    import pylab as pl
    pl.figure()
    pl.imshow(collage)
    pl.gray()
    pl.show()

    import scoring
    pl.figure()
    roc1 = scoring.roc(res1, stimuli)
    roc2 = scoring.roc(res2, stimuli)

    pl.plot(roc1)
    pl.plot(roc2)
    pl.plot([0, len(stimuli)], [0, 1])

