from operator import index
import os
import sys
from textwrap import indent
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit
from mne.decoding import CSP

current_directory = os.path.abspath('')

project_root = current_directory

sys.path.append(project_root)

print("ROOT:", project_root)
from Early_predict_UQ.data.make_dataset import make_data


VALID_CONFIDENCE_TYPES = {'highest_prob', 'difference_two_highest', 'neg_norm_shannon'}

#Set confidence value given probabilities and method
def find_confidence(confidence_type, probabilities):
    probabilities = probabilities.flatten()

    sorted_probs = sorted(probabilities, reverse=True)
    if confidence_type == 'highest_prob':
        confidence = sorted_probs[0]
    elif  confidence_type == 'difference_two_highest':
        confidence = 1 - (1 / (1 + (sorted_probs[0] + (sorted_probs[0] - sorted_probs[1]))))
    else:
        confidence = 1 - entropy(pk = probabilities, base = len(probabilities))
    return confidence, probabilities

def early_pred(probabilities, predict, numTimesThresholdCrossed, patience, confidence_type, threshold, previous_class_index):
    if confidence_type not in VALID_CONFIDENCE_TYPES:
        raise ValueError(" Confidence must be one of %r." % VALID_CONFIDENCE_TYPES)
    #Set confidence
    confidence, probabilities_flattened = find_confidence(confidence_type,  probabilities)

    #find current highest index
    index_highest =  np.argmax(probabilities_flattened)

    #Determine wether the dominant class has changed to reset the patience counter
    if previous_class_index == None:
        previous_class_index = index_highest
    else:
        #The dominant class has changed, reset the counter
        if index_highest != previous_class_index:
            numTimesThresholdCrossed = 0
            previous_class_index = index_highest 

    #Stopping rule: If confidence > threshold, and threshold reached n = patience times -> Predict early
    if np.round(confidence, 2) > threshold and not predict:
        numTimesThresholdCrossed += 1
        #Predicit early
        if numTimesThresholdCrossed == patience:
            predict = True
    return predict, numTimesThresholdCrossed, previous_class_index

#Given sliding window and stopping values, we average the accuracy and prediction time for the model
def run_expanding_classification(subjects, threshold, patience, confidence_type, initial_window_length, expansion_rate, sfreq):
    scores_across_subjects = []
    prediction_time_across_subjects = []
    current_person = 0
    for person in subjects:
        current_person += 1
        print("Person %d" % (person))
        subject= [person]
        epochs, labels = make_data(subject)
        labels = epochs.events[:, -1] - 4
        epochs_data = epochs.get_data(copy=False)

        cv = ShuffleSplit(n_splits=10, test_size = 0.2, random_state=42)

        scores_cv_splits = []
        predict_time_cv_splits = []


        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        current_cv = 0 
        
        scores_cv_splits = []
        predict_time_cv_splits = []

        for train_idx, test_idx in cv.split(epochs_data):

            current_cv += 1
            y_train, y_test = labels[train_idx], labels[test_idx]
            X_train = csp.fit_transform(epochs_data[train_idx], y_train)
            lda.fit(X_train, y_train)
            w_start = np.arange(0, epochs_data.shape[2] - initial_window_length, expansion_rate) 

            scores_across_epochs = []
            predict_time_across_epochs = []

            for epoch_idx in range(len(test_idx)):
                previous_class_index = None
                predict = False
                numTimesThresholdCrossed = 0
                for n, window_start in enumerate(w_start):
                    window_length = initial_window_length + n * expansion_rate
                    X_test_window = csp.transform(epochs_data[test_idx][:, :,  window_start:(window_start + window_length)])
                    X_test_epoch_window = X_test_window[epoch_idx]
                    probabilities = lda.predict_proba([X_test_epoch_window])
                    probabilities = np.array(probabilities)
                    probabilities = probabilities.flatten()
                    predict, numTimesThresholdCrossed,  previous_class_index = early_pred(
                        probabilities, predict, numTimesThresholdCrossed, patience, confidence_type, threshold,  previous_class_index
                    )
                    if predict:
                        predict_time = n
                        score = lda.score(X_test_epoch_window.reshape(1, -1), [y_test[epoch_idx]])
                        break
                else:
                    predict_time = n
                    score = lda.score(X_test_epoch_window.reshape(1, -1), [y_test[epoch_idx]])
                predict_time = (predict_time + window_length / 2.0) / sfreq + epochs.tmin
                scores_across_epochs.append(score)
                predict_time_across_epochs.append(predict_time)
                
            scores_cv_splits.append(scores_across_epochs)
            predict_time_cv_splits.append(predict_time_across_epochs)

        
        scores_cv_splits = np.array(scores_cv_splits)
        predict_time_cv_splits = np.array(predict_time_cv_splits)

        mean_scores_across_cv = np.mean(scores_cv_splits, axis=0)
        mean_predict_time_across_cv = np.mean(predict_time_cv_splits, axis=0)
        if current_person == 1:
            scores_across_subjects  = mean_scores_across_cv
            prediction_time_across_subjects = mean_predict_time_across_cv
        else:
            scores_across_subjects = np.vstack((scores_across_subjects, mean_scores_across_cv))
            prediction_time_across_subjects = np.vstack((prediction_time_across_subjects, mean_predict_time_across_cv))

        mean_scores_across_subjects = np.mean(scores_across_subjects, axis=0)
        accuracy = np.mean(mean_scores_across_subjects)

        mean_prediction_time_across_subjects = np.mean(prediction_time_across_subjects, axis=0)
        prediction_time = np.mean(mean_prediction_time_across_subjects)
    return accuracy, prediction_time



def evaluate_and_plot(accuracy_array, prediction_time_array, threshold_values, patience_values, initial_window_length, sfreq,confidence_type):
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

    patience_values = (patience_values * initial_window_length) / sfreq # ? patience values cant be made into seconds for the expanding window
    # Plotting accuracy
    for i in range(len(accuracy_array)):
        plt.plot(patience_values, accuracy_array[i], label=f'Threshold {threshold_labels[i]}', linestyle='-', marker='o')

    plt.xlabel('Patience (sec)')
    plt.ylabel('Accuracy')
    plt.axhline(class_balance, linestyle="-", color="k", label="Chance")
    plt.title('Accuracy vs Patience for Different Thresholds: LDA - Dynamic - Expanding model')
    plt.legend()
    plt.grid(True)
    if confidence_type == 'highest_prob':
        plt.savefig(project_root + '/reports/figures/cumulitive/LDA/dynamic/expanding/highest_prob/accuracy_thresholds.png')
    elif confidence_type == 'difference_two_highest':
        plt.savefig(project_root + '/reports/figures/cumulitive/LDA/dynamic/expanding/difference_two_highest/accuracy_thresholds.png')
    else:
        plt.savefig(project_root + '/reports/figures/cumulitive/LDA/dynamic/expanding/neg_norm_shannon/accuracy_thresholds.png')
    plt.show()

    # Plotting prediction time
    #plt.figure(figsize=(10, 5))
    for i in range(len(prediction_time_array)):
        plt.plot(patience_values, prediction_time_array[i], label=f'Threshold {threshold_labels[i]}', linestyle='-', marker='o')

    plt.xlabel('Patience (sec)')
    plt.ylabel('Prediction Time')
    plt.axhline(onset, linestyle="--", color="r", label="Onset")
    plt.axhline(offset, linestyle="--", color="b", label="Offset")
    plt.title('Prediction Time vs Patience for Different Thresholds: LDA - Dynamic - Expanding model')
    plt.legend()
    plt.grid(True)
    if confidence_type == 'highest_prob':
        plt.savefig(project_root + '/reports/figures/cumulitive/LDA/dynamic/expanding/highest_prob/pred_time_thresholds.png')
    elif confidence_type == 'difference_two_highest':
        plt.savefig(project_root + '/reports/figures/cumulitive/LDA/dynamic/expanding/difference_two_highest/pred_time_thresholds.png')
    else:
        plt.savefig(project_root + '/reports/figures/cumulitive/LDA/dynamic/expanding/neg_norm_shannon/pred_time_thresholds.png')
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

    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9] 
    confidence_types = {'highest_prob', 'difference_two_highest', 'neg_norm_shannon'} 
    sfreq = 250      

    #Use tuned hyperparams from?
    initial_window_length = int(sfreq * 0.5)  
    expansion_rate = int(sfreq * 0.1)   
    w_start= np.arange(0, epochs_info(length= True) - initial_window_length, expansion_rate) 

    patience_values = np.arange(1, len(w_start), 4) 
    print("patience_values: ", patience_values)
    print("len patience: ", len(patience_values))
    threshold_values = np.arange(0.1, 1, 0.2)
    print("threshold_values: ", threshold_values)
    print("len threshold: ", len(threshold_values))


    # evaluate everything for each of the 3 methods
    for confidence_type in VALID_CONFIDENCE_TYPES:
        # array to hold the average accuracy and prediction times with size len(confidence_type) x len(thre)
        accuracy_array = []
        prediction_time_array = []
    # over threshold values
        for n, threshold in enumerate(threshold_values):
            accuracy_row = []
            prediction_time_row = []
            # over patience values
            for m, patience in enumerate(patience_values):
                print("\n")
                print(f"Threshold:{n+1}/{len(threshold_values)},  Patience: {m+1}/{len(patience_values)}")
                print("\n")
                #given the varaibles, provide the average accuracy and prediction times (early prediction)
                accuracy, prediction_time = run_expanding_classification(subjects, threshold, patience, confidence_type, initial_window_length, expansion_rate, sfreq)
                accuracy_row.append(accuracy)
                prediction_time_row.append(prediction_time)
            accuracy_array.append(accuracy_row)
            prediction_time_array.append(prediction_time_row)

        #Plotting the average accuracy and prediction times (early prediction) as well as the different threshold and patience values across subjects for each of the confidence types
        accuracy_array = np.array(accuracy_array)
        prediction_time_array = np.array(prediction_time_array)
        evaluate_and_plot(accuracy_array, prediction_time_array, threshold_values, patience_values, initial_window_length, sfreq, confidence_type)
