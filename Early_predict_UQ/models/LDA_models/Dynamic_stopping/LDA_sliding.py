from cProfile import label
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from mne.decoding import CSP

current_directory = os.path.abspath('')

project_root = current_directory

sys.path.append(project_root)

print("ROOT:", project_root)
from Early_predict_UQ.data.make_dataset import make_data


VALID_CONFIDENCE_TYPES = {'highest_prob', 'difference_two_highest', 'neg_norm_shannon'}
    
def early_pred(probabilities, predict, numTimesBelowThreshold, patience, confidence_type, threshold):
    if confidence_type not in VALID_CONFIDENCE_TYPES:
        raise ValueError("results: status must be one of %r." % VALID_CONFIDENCE_TYPES)
    probabilities = probabilities.flatten()
    sorted_probs = sorted(probabilities, reverse=True)
    if confidence_type == 'highest_prob':
        confidence = sorted_probs[0]
    elif  confidence_type == 'difference_two_highest':
        confidence = 1 - (1 / (1 + (sorted_probs[0] + (sorted_probs[0] - sorted_probs[1]))))
    else:
        confidence = 1 - entropy(pk = probabilities, base = len(probabilities))
    if np.round(confidence, 2) > threshold and not predict:
        #print("confindence:", confidence)
        sorted_probs[0]
        numTimesBelowThreshold += 1
        if numTimesBelowThreshold == patience:
            predict = True
    return predict, confidence, numTimesBelowThreshold

def run_sliding_classification(subjects, threshold, patience, confidence_type, w_length, w_step, sfreq):
    scores_across_subjects = []
    prediction_time_across_subjects = []
    current_person = 0
    for person in subjects:
        current_person += 1
        print("Person %d" % (person))
        subject= [person]
        epochs, labels = make_data(subject)
        epochs_train = epochs.copy()
        labels = epochs.events[:, -1] - 4
        epochs_data = epochs.get_data(copy=False)
        epochs_data_train = epochs_train.get_data(copy=False)

        kf = KFold(n_splits=2, shuffle=True, random_state=42)
        scores_cv_splits = []
        predict_time_cv_splits = []

        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        current_cv = 0 
        for train_idx, test_idx in kf.split(epochs_data):
            current_cv += 1
            y_train, y_test = labels[train_idx], labels[test_idx]
            X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
            lda.fit(X_train, y_train)
            w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step) 
            scores_across_epochs = []
            predict_time_across_epochs = []

            for epoch_idx in range(len(test_idx)):
                predict = False
                numTimesBelowThreshold = 0
                for n in w_start:
                    X_test_window = csp.transform(epochs_data_train[test_idx][:, :, n:(n + w_length)])
                    X_test_epoch_window = X_test_window[epoch_idx]
                    probabilities = lda.predict_proba([X_test_epoch_window])
                    probabilities = np.array(probabilities)
                    probabilities = probabilities.flatten()
                    predict, confidence, numTimesBelowThreshold = early_pred(
                        probabilities, predict, numTimesBelowThreshold, patience, confidence_type, threshold
                    )
                    if predict:
                        #IF WE DIDNT PREDICT EARLY, MAYBE PREDICT ON THE WHOLE EPOCH?
                        predict_time = n
                        score = lda.score(X_test_epoch_window.reshape(1, -1), [y_test[epoch_idx]])
                        break
                else:
                    predict_time = n
                    score = lda.score(X_test_epoch_window.reshape(1, -1), [y_test[epoch_idx]])
                predict_time = (predict_time + w_length / 2.0) / sfreq + epochs.tmin
                scores_across_epochs.append(score)
                predict_time_across_epochs.append(predict_time)

            if current_cv == 1:
                scores_cv_splits = np.array(scores_across_epochs)
                predict_time_cv_splits = np.array(predict_time_across_epochs)
            else:
                scores_cv_splits = np.vstack((scores_cv_splits,np.array(scores_across_epochs)))
                predict_time_cv_splits = np.vstack((predict_time_cv_splits,np.array(predict_time_across_epochs)))

        mean_scores_across_cv = np.mean(scores_cv_splits, axis=0)
        mean_predict_time_across_cv = np.mean(predict_time_cv_splits, axis=0)
        if current_person == 1:
            scores_across_subjects  = np.array(mean_scores_across_cv)
            prediction_time_across_subjects = np.array(mean_predict_time_across_cv)
        else:
            scores_across_subjects = np.vstack((scores_across_subjects,np.array(mean_scores_across_cv)))
            prediction_time_across_subjects = np.vstack((predict_time_cv_splits,np.array(mean_predict_time_across_cv)))

        mean_scores_across_subjects = np.mean(scores_across_subjects, axis=0)
        accuracy = np.mean(mean_scores_across_subjects)

        mean_prediction_time_across_subjects = np.mean(prediction_time_across_subjects, axis=0)
        mean_prediction_time = np.mean(mean_prediction_time_across_subjects)
    return accuracy, mean_prediction_time

def evaluate_and_plot(accuracy_array, prediction_time_array, threshold_values, patience_values):
    threshold_labels = [f'{threshold:.1f}' for threshold in threshold_values]
    labels = epochs_info(labels = True)
    # A formality as classes are balanced
    class_balance = np.zeros(4)
    for i in range(4):
        class_balance[i] = np.mean(labels == i)
    class_balance = np.max(class_balance)

    tmin, tmax = epochs_info(tmin = True, tmax = True)
    onset = tmin
    offset = tmax


    # Plotting accuracy
    #plt.figure(figsize=(10, 5))
    for i in range(len(accuracy_array)):
        plt.plot(patience_values, accuracy_array[i], label=f'Threshold {threshold_labels[i]}')

    plt.xlabel('Patience')
    plt.ylabel('Accuracy')
    plt.axhline(class_balance, linestyle="-", color="k", label="Chance")
    plt.title('Accuracy vs Patience for Different Thresholds')
    plt.legend()
    plt.grid(True)
    if confidence_type == 'highest_prob':
        plt.savefig(project_root + '/reports/figures/cumulitive/LDA/dynamic/sliding/highest_prob/accuracy_thresholds.png')
    elif confidence_type == 'difference_two_highest':
        plt.savefig(project_root + '/reports/figures/cumulitive/LDA/dynamic/sliding/difference_two_highest/accuracy_thresholds.png')
    else:
        plt.savefig(project_root + '/reports/figures/cumulitive/LDA/dynamic/sliding/neg_norm_shannon/accuracy_thresholds.png')
    plt.show()

    # Plotting prediction time
    #plt.figure(figsize=(10, 5))
    for i in range(len(prediction_time_array)):
        plt.plot(patience_values, prediction_time_array[i], label=f'Threshold {threshold_labels[i]}')

    plt.xlabel('Patience')
    plt.ylabel('Prediction Time')
    plt.axhline(onset, linestyle="--", color="r", label="Onset")
    plt.axhline(offset, linestyle="--", color="b", label="Offset")
    plt.title('Prediction Time vs Patience for Different Thresholds')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def epochs_info(labels=False, tmin=False, tmax = False, length=False):
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
        raise ValueError("At least one of 'labels', 'tmin', 'tmax' or 'length' must be True.")

if __name__ == "__main__":
    #threshold = 0.6  # values - {0,1}
    #patience = 4 # values - {1, number_of_windows}
    subjects = [1, 2]  # 9 subjects

    confidence_type = 'neg_norm_shannon' #{'highest_prob', 'difference_two_highest', 'neg_norm_shannon'}
    sfreq = 250      
    w_length = int(sfreq * 0.5)  
    w_step = int(sfreq * 0.1)   

    # just to access 
    
    w_start= np.arange(0, epochs_info(length= True) - w_length, w_step) 

    patience_values = np.arange(1, 2, 2) 
    threshold_values = np.arange(0, 0.2, 0.2)

    #csp components #hyperparameter
    #cross validation #hyperparmater
    accuracy_array = []
    prediction_time_array = []

    # over threshold values
    for threshold in threshold_values:
        accuracy_row = []
        prediction_time_row = []
        # over patience values
        for patience in patience_values:
            print("\n")
            print("threshold: %f / 1, patience: %d / 20" % (threshold,  patience))
            print("\n")
            accuracy, mean_prediction_time = run_sliding_classification(subjects, threshold, patience, confidence_type, w_length, w_step, sfreq)
            accuracy_row.append(accuracy)
            prediction_time_row.append(mean_prediction_time)
        accuracy_array.append(accuracy_row)
        prediction_time_array.append(prediction_time_row)

    accuracy_array = np.array(accuracy_array)
    prediction_time_array = np.array(prediction_time_array)

    print("accuracy_array: ", accuracy_array[0])
    print("prediction_time_array: ",  prediction_time_array[0])
    evaluate_and_plot(accuracy_array, prediction_time_array, threshold_values, patience_values)