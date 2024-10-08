import os
import sys
import numpy as np
from math import log2
from sklearn.svm import SVC
from scipy.stats import entropy
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import mne
import logging

mne.set_log_level(logging.ERROR)
from mne.decoding import CSP
from sklearn.model_selection import ParameterSampler

current_directory = os.path.abspath("")

project_root = current_directory

sys.path.append(project_root)
print("ROOT:", project_root)

VALID_CONFIDENCE_TYPES = {
    "highest_prob",
    "difference_two_highest",
    "neg_norm_shannon",
}  # No longer a tuned parameter: neg_norm_shannon only.
# Static hyperparameter tuning
from Early_predict_UQ.data.make_dataset import make_data


# Set confidence value given probabilities and method
def find_confidence(confidence_type, probabilities):
    probabilities = probabilities.flatten()

    sorted_probs = sorted(probabilities, reverse=True)
    if confidence_type == "highest_prob":
        confidence = sorted_probs[0]
    elif confidence_type == "difference_two_highest":
        confidence = sorted_probs[0] - sorted_probs[1]
    else:
        # complement of the normalized shannon entropy
        confidence = 1 - entropy(
            pk=probabilities, base=len(probabilities)
        )  # Always true
    return confidence, probabilities


def early_pred(
    probabilities,
    predict,
    numTimesThresholdCrossed,
    patience,
    confidence_type,
    threshold,
    previous_class_index,
):
    if confidence_type not in VALID_CONFIDENCE_TYPES:
        raise ValueError(" Confidence must be one of %r." % VALID_CONFIDENCE_TYPES)
    # Set confidence
    confidence, probabilities_flattened = find_confidence(
        confidence_type, probabilities
    )

    # find current highest index
    index_highest = np.argmax(probabilities_flattened)

    # Determine wether the dominant class has changed to reset the patience counter
    if previous_class_index == None:
        previous_class_index = index_highest
    else:
        # The dominant class has changed, reset the counter
        if index_highest != previous_class_index:
            numTimesThresholdCrossed = 0
            previous_class_index = index_highest

    # Stopping rule: If confidence > threshold, and threshold reached n = patience times -> Predict early
    if np.round(confidence, 2) > threshold and not predict:
        numTimesThresholdCrossed += 1
        # Predict early
        if numTimesThresholdCrossed == patience:
            predict = True
    return predict, numTimesThresholdCrossed, previous_class_index


# Given sliding window and stopping values, we average the accuracy and prediction time for the model
def run_expanding_classification_dynamic(
    subjects,
    threshold,
    patience,
    confidence_type,
    initial_window_length,
    expansion_rate,
    sfreq,
    csp_components,
    c,
    kernel,
    gamma,
    degree,
):
    scores_across_subjects = []
    kappa_across_subjects = []
    prediction_time_across_subjects = []

    subjects_accuracies = []
    subjects_kappa = []
    subjects_prediction_times = []
    # confusion matrix
    number_cm = 0
    cm = np.zeros((4, 4))
    current_person = 0
    for person in subjects:
        current_person += 1
        print("Person %d" % (person))
        subject = [person]
        epochs, labels = make_data(subject)
        epochs_data = epochs.get_data(copy=False)

        # Get the training set - first session of the data
        train_indexes = [
            i
            for i, epoch in enumerate(epochs)
            if epochs[i].metadata["session"].iloc[0] == "0train"
        ]
        train_data = epochs_data[train_indexes]
        train_labels = labels[train_indexes]

        cv = KFold(n_splits=10, shuffle=True, random_state=42)

        scores_cv_splits = []
        kappa_cv_splits = []
        prediction_time_cv_splits = []

        if kernel != "linear":
            if kernel == "poly":
                svm = SVC(
                    C=c, kernel=kernel, gamma=gamma, degree=degree, probability=True
                )
            svm = SVC(C=c, kernel=kernel, gamma=gamma, probability=True)
        else:
            svm = SVC(C=c, kernel=kernel, probability=True)

        csp = CSP(n_components=csp_components, reg=None, log=True, norm_trace=False)

        for train_idx, test_idx in cv.split(train_data):
            y_train, y_test = train_labels[train_idx], train_labels[test_idx]
            X_train = csp.fit_transform(train_data[train_idx], y_train)
            svm.fit(X_train, y_train)

            scores_across_epochs = []
            kappa_across_epochs = []
            predict_time_across_epochs = []
            predictions = []
            w_start = np.arange(
                0, train_data.shape[2] - initial_window_length, expansion_rate
            )

            for epoch_idx in range(len(test_idx)):
                predict = False
                numTimesThresholdCrossed = 0
                previous_class_index = None

                for n, window_start in enumerate(w_start):
                    window_length = initial_window_length + n * expansion_rate
                    X_test_window = csp.transform(
                        train_data[test_idx][:, :, w_start[0] : window_length]
                    )
                    X_test_epoch_window = X_test_window[epoch_idx]

                    probabilities = svm.predict_proba([X_test_epoch_window])
                    probabilities = np.array(probabilities).flatten()

                    predict, numTimesThresholdCrossed, previous_class_index = (
                        early_pred(
                            probabilities,
                            predict,
                            numTimesThresholdCrossed,
                            patience,
                            confidence_type,
                            threshold,
                            previous_class_index,
                        )
                    )

                    if predict:
                        # Score
                        score = svm.score(
                            X_test_epoch_window.reshape(1, -1), [y_test[epoch_idx]]
                        )
                        scores_across_epochs.append(score)

                        # Prediction time
                        predict_time = window_length
                        predict_time_across_epochs.append(predict_time)

                        # Kappa and confusion matrix
                        prediction = svm.predict(X_test_epoch_window.reshape(1, -1))
                        predictions.append(prediction)

                        break
                else:
                    # Score
                    score = svm.score(
                        X_test_epoch_window.reshape(1, -1), [y_test[epoch_idx]]
                    )
                    scores_across_epochs.append(score)

                    # Prediction time
                    predict_time_last = window_length
                    predict_time_across_epochs.append(predict_time_last)

                    # Kappa and confusion matrix
                    prediction = svm.predict(X_test_epoch_window.reshape(1, -1))
                    predictions.append(prediction)
            # Kappa
            kappa_score = cohen_kappa_score(predictions, y_test)
            kappa_across_epochs.append(kappa_score)

            # Append scores for this CV split
            scores_cv_splits.append(np.mean(scores_across_epochs))
            kappa_cv_splits.append(kappa_score)
            prediction_time_cv_splits.append(np.mean(predict_time_across_epochs))

        mean_scores_across_cv = np.mean(scores_cv_splits)
        mean_kappa_across_cv = np.mean(kappa_cv_splits)
        mean_prediction_time_across_cv = np.mean(prediction_time_cv_splits)

        scores_across_subjects.append(mean_scores_across_cv)
        kappa_across_subjects.append(mean_kappa_across_cv)
        prediction_time_across_subjects.append(mean_prediction_time_across_cv)

        subjects_accuracies.append(mean_scores_across_cv)
        subjects_kappa.append(mean_kappa_across_cv)
        subjects_prediction_times.append(mean_prediction_time_across_cv)

        # Confusion matrix (average over CV splits)
        cm += confusion_matrix(
            train_labels[test_idx],
            predictions,
            labels=["left_hand", "right_hand", "tongue", "feet"],
        )
        number_cm += 1

    # Final averages
    accuracy = np.mean(scores_across_subjects)
    kappa = np.mean(kappa_across_subjects)
    prediction_time = np.mean(prediction_time_across_subjects)

    cm = np.divide(cm, number_cm)

    return (
        accuracy,
        kappa,
        prediction_time,
        cm,
        subjects_accuracies,
        subjects_prediction_times,
        subjects_kappa,
    )


# Expanding window - classification - tuning using kfold cross validation
##calculate kappa and accuracy at each window step
def run_expanding_classification_tuning_static(
    subjects,
    initial_window_length,
    expansion_rate,
    csp_components,
    c,
    kernel,
    gamma,
    degree,
):
    scores_across_subjects = []
    kappa_across_subjects = []

    subjects_accuracies = []
    subjects_kappa = []
    current_person = 0
    for person in subjects:
        current_person += 1
        print("Person %d" % (person))
        subject = [person]
        epochs, labels = make_data(subject)

        # get the training set - first session of the data for each subject
        train_indexes = [
            i
            for i, epoch in enumerate(epochs)
            if epochs[i].metadata["session"].iloc[0] == "0train"
        ]
        epochs_data = epochs.get_data(copy=False)
        train_data = epochs_data[train_indexes]
        train_labels = labels[train_indexes]

        cv = KFold(n_splits=10, shuffle=True, random_state=42)

        scores_cv_splits = []
        kappa_cv_splits = []

        if kernel != "linear":
            if kernel == "poly":
                svm = SVC(
                    C=c, kernel=kernel, gamma=gamma, degree=degree, probability=True
                )
            svm = SVC(C=c, kernel=kernel, gamma=gamma, probability=True)
        else:
            svm = SVC(C=c, kernel=kernel, probability=True)
        csp = CSP(n_components=csp_components, reg=None, log=True, norm_trace=False)
        current_cv = 0
        for train_idx, test_idx in cv.split(train_data):
            current_cv += 1
            y_train, y_test = train_labels[train_idx], train_labels[test_idx]
            X_train = csp.fit_transform(train_data[train_idx], y_train)
            svm.fit(X_train, y_train)
            w_start = np.arange(
                0, train_data.shape[2] - initial_window_length, expansion_rate
            )
            scores_across_epochs = []
            kappa_across_epochs = []
            for n, window_start in enumerate(w_start):
                window_length = initial_window_length + n * expansion_rate
                X_test_window = csp.transform(
                    train_data[test_idx][:, :, w_start[0] : window_length]
                )
                # accuracy
                score = svm.score(X_test_window, y_test)
                scores_across_epochs.append(score)

                # kappa
                kappa = cohen_kappa_score(svm.predict(X_test_window), y_test)
                kappa_across_epochs.append(kappa)
            if current_cv == 1:
                scores_cv_splits = np.array(scores_across_epochs)
                kappa_cv_splits = np.array(kappa_across_epochs)
            else:
                scores_cv_splits = np.vstack(
                    (scores_cv_splits, np.array(scores_across_epochs))
                )
                kappa_cv_splits = np.vstack(
                    (kappa_cv_splits, np.array(kappa_across_epochs))
                )
        mean_scores_across_cv = np.mean(scores_cv_splits, axis=0)
        mean_kappa_across_cv = np.mean(kappa_cv_splits, axis=0)
        if current_person == 1:
            scores_across_subjects = np.array(mean_scores_across_cv)
            kappa_across_subjects = np.array(mean_kappa_across_cv)
        else:
            scores_across_subjects = np.vstack(
                (scores_across_subjects, np.array(mean_scores_across_cv))
            )
            kappa_across_subjects = np.vstack(
                (kappa_across_subjects, np.array(mean_kappa_across_cv))
            )

        subjects_accuracies.append(np.mean(mean_scores_across_cv))
        subjects_kappa.append(np.mean(mean_kappa_across_cv))

    mean_scores_across_subjects = np.mean(scores_across_subjects, axis=0)
    mean_kappa_across_subjects = np.mean(kappa_across_subjects, axis=0)
    accuracy = mean_scores_across_subjects
    kappa = mean_kappa_across_subjects

    return (
        subjects_accuracies,
        scores_across_subjects,
        subjects_kappa,
        kappa_across_subjects,
        accuracy,
        kappa,
    )


"""Create parameter search space
    - Sample n_iter sets of parameters - set to 60 iterations. 
    - 60 iterations lets us reach the 95% confidence interval that we have found a set of parameters with "good values",
      if the good space covers only 5% of the search space.
    - 1-0.95^(60) = 0.953 > 0.95
    (Random Search for Hyper-Parameter Optimization - Bergstra & Bengio (2012))
"""


def create_parameterslist(sfreq):
    rng = np.random.RandomState(42)

    # max intial_length and epxansion rate to be 1/4 of the trial during the MI task, or 1 seconds
    initial_window_length = np.round(rng.uniform(0.1, 1, 10), 1)
    expansion_rate = np.round(rng.uniform(0.1, 0.5, 10), 1)

    parameters = {
        # 1, 2 filters for each class, as we have 4 classes.
        "csp_components": [
            4,
            8,
        ],  # more than 8 was reading into noise - occipital lobe pattern
        "initial_window_length": np.round(sfreq * initial_window_length).astype(int),
        "expansion_rate": np.round(sfreq * expansion_rate).astype(int),
        "C": [0.1, 1, 10, 100, 1000],
        "kernel": ["linear", "rbf", "poly", "sigmoid"],
        "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
        "degree": [2, 3, 4, 5],
    }

    # Random search parameters - n_tier sets of parameter values
    parameters_list = list(ParameterSampler(parameters, n_iter=60, random_state=rng))
    for param_set in parameters_list:
        if param_set["kernel"] == "linear":
            param_set["gamma"] = None
        if param_set["kernel"] != "poly":
            param_set["degree"] = None
    return parameters_list


# Random search and return the best parameter values and its accuracy
def hyperparameter_tuning(parameters_list, subjects):
    mean_accuracy = 0
    best_accuracy = 0
    for n, param_set in enumerate(parameters_list):
        csp_components = param_set["csp_components"]
        initial_window_length = param_set["initial_window_length"]
        expansion_rate = param_set["expansion_rate"]
        c = param_set["C"]
        kernel = param_set["kernel"]
        gamma = param_set["gamma"]
        degree = param_set["degree"]
        _, _, _, _, accuracy, _ = run_expanding_classification_tuning_static(
            subjects,
            initial_window_length,
            expansion_rate,
            csp_components,
            c,
            kernel,
            gamma,
            degree,
        )

        mean_accuracy = np.mean(accuracy)

        print(
            f"Iteration {n+1}/{len(parameters_list)}: Mean accuracy for parameters {param_set} is {mean_accuracy}",
            flush=True,
        )
        # Write to file
        h = open(
            project_root
            + "/reports/figures/cumulative/SVM/dynamicVSstatic/expanding_model_tuning1.txt",
            "w",
        )
        h.write(f"Accuracy: {mean_accuracy}, Parameters: {param_set}")
        h.close()
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_params = param_set

    return best_params, best_accuracy


# Access general epoch information
def epochs_info(labels=False, tmin=False, tmax=False, length=False):
    global epochs_data
    global labels_data
    if labels or tmax or tmin or length:
        epochs, labels_data = make_data([1])
        epochs_data = epochs.get_data(copy=False)
    if labels and tmin:
        return labels_data, epochs.tmin
    if labels and tmax:
        return labels_data, epochs.tmin
    if tmin and tmax:
        return epochs.tmin, epochs.tmax
    elif labels and length:
        return labels_data, epochs_data.shape[2]
    elif tmin and length:
        return epochs.tmin, epochs_data.shape[2]
    elif tmax and length:
        return epochs.tmax, epochs_data.shape[2]
    elif labels:
        return labels_data
    elif tmin:
        return epochs.tmin
    elif tmax:
        return epochs.tmin
    elif length:
        return epochs_data.shape[2]
    else:
        raise ValueError(
            "At least one of 'labels', 'tmin', 'tmax' or 'length' must be True."
        )


# calculate the information transfer rate
# B = log2 N + P log2 P + (1 − P)log2[(1 − P)/(N − 1)]
# Q =  (60/T)
# ITR  = B * Q (bits/min)
def calculate_best_itr_dyn_tune(
    best_itr,
    best_itr_patience,
    best_itr_threshold,
    best_confidence_type,
    accuracy,
    prediction_time,
    patience,
    threshold,
    confidence_type,
):
    number_classes = 4  # we have 4 classes in this dataset
    current_B = (
        log2(number_classes)
        + accuracy * log2(accuracy)
        + (1 - accuracy) * log2((1 - accuracy) / (number_classes - 1))
    )

    prediction_time = prediction_time / 250  # turning prediction time to seconds first
    current_Q = 60 / prediction_time

    current_itr = current_B * current_Q
    if current_itr > best_itr:
        best_itr_patience = patience
        best_itr_threshold = threshold
        best_itr = current_itr
        best_confidence_type = confidence_type
    return (
        best_itr,
        best_itr_patience,
        best_itr_threshold,
        best_confidence_type,
        current_itr,
    )


def run_random_search(
    subjects,
    confidence_types,
    patience_values,
    threshold_values,
    initial_window_length,
    expansion_rate,
    sfreq,
    csp_components,
    c,
    kernel,
    gamma,
    degree,
):
    best_itr_tune = 0
    best_itr_threshold = 0
    best_itr_patience = 0
    best_confidence_type = None

    iterations = 60
    rng_dyn = np.random.RandomState(42)

    # Perform 60 iterations
    for iteration in range(iterations):
        confidence_type = rng_dyn.choice(confidence_types)
        patience = rng_dyn.choice(patience_values)

        # Initialize averages
        avg_accuracy = 0
        avg_prediction_time = 0

        # Loop over threshold values
        for threshold in threshold_values:
            accuracy, kappa, prediction_time, _, _, _, _ = (
                run_expanding_classification_dynamic(
                    subjects,
                    threshold,
                    patience,
                    confidence_type,
                    initial_window_length,
                    expansion_rate,
                    sfreq,
                    csp_components,
                    c,
                    kernel,
                    gamma,
                    degree,
                )
            )
            avg_accuracy += accuracy
            avg_prediction_time += prediction_time

        # Calculate averages
        avg_accuracy /= len(threshold_values)
        avg_prediction_time /= len(threshold_values)

        # Calculate best itr using tuning function
        (
            best_itr_tune,
            best_itr_patience,
            best_itr_threshold,
            best_confidence_type,
            current_itr,
        ) = calculate_best_itr_dyn_tune(
            best_itr_tune,
            best_itr_patience,
            best_itr_threshold,
            best_confidence_type,
            avg_accuracy,
            avg_prediction_time,
            patience,
            threshold,
            confidence_type,
        )

        print(
            f"Iteration {iteration+1}/{iterations} completed. Current ITR: {current_itr}, current confidence type: {confidence_type}, current patience:{patience} \n Best ITR: {best_itr_tune} | Best Confidence Type: {best_confidence_type} | Best Patience: {best_itr_patience}",
            flush=True,
        )

    return best_itr_tune, best_itr_patience, best_confidence_type


def tune_svm_expanding():
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 9 subjects
    sfreq = 250  # Sampling frequency - 250Hz
    """
    Hyperparameter tuning
    """
    # Hyperparameter_tuning for csp, and expanding window parameters using the static model
    print("\n\n Hyperparameter tuning (1): csp and window parameters \n\n", flush=True)
    parameters_list = create_parameterslist(sfreq)
    best_params_expanding, _ = hyperparameter_tuning(parameters_list, subjects)
    print("\n\n Hyperparameter tuning (1): completed \n\n", flush=True)

    csp_components = best_params_expanding["csp_components"]
    initial_window_length = best_params_expanding["initial_window_length"]
    expansion_rate = best_params_expanding["expansion_rate"]
    c = best_params_expanding["C"]
    kernel = best_params_expanding["kernel"]
    gamma = best_params_expanding["gamma"]
    degree = best_params_expanding["degree"]

    print(
        f"Chosen_hyperparameter_values: \n - csp: {csp_components},\n - initial_window_length: {initial_window_length}, \n -expansion_rate: {expansion_rate}, \n c: {c}, \n kernel: {kernel}, \n gamma: {gamma}, \n degree: {degree}",
        flush=True,
    )

    w_start = np.arange(
        0, epochs_info(length=True) - initial_window_length, expansion_rate
    )
    confidence_types = ["highest_prob", "difference_two_highest", "neg_norm_shannon"]
    patience_values = np.arange(1, len(w_start))
    # Threshold values -> [0.001, 0.01, 0.1 - 0.9, 0.99, 0.999]
    threshold_values = np.array(np.arange(0.1, 1, 0.1))
    threshold_values = np.concatenate(([0.001, 0.01], threshold_values))
    threshold_values = np.append(threshold_values, [0.99, 0.999])

    # Find optimal confidence type and patience from the sliding dynamic model using itr
    print(
        "\n\n Hyperparameter tuning (2): patience and confidence type \n\n", flush=True
    )
    best_itr_tune, best_itr_patience, best_confidence_type = run_random_search(
        subjects,
        confidence_types,
        patience_values,
        threshold_values,
        initial_window_length,
        expansion_rate,
        sfreq,
        csp_components,
        c,
        kernel,
        gamma,
        degree,
    )
    print("\n\n Hyperparameter tuning (2): completed\n\n", flush=True)
    print(
        f"chosen patience: {best_itr_patience}, chosen confidence_type: {best_confidence_type}",
        flush=True,
    )

    return (
        best_params_expanding,
        best_itr_patience,
        best_confidence_type,
    )


if __name__ == "__main__":
    tune_svm_expanding()
