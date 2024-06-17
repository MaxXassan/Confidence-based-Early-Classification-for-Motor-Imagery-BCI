from math import log2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold # see if u an intergrate this
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
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
        ''' 
            - p0 -> highest probability
            - p1 -> second highest probability
            conditions that are and should be true:
                - 0 <= p0 <= 1 
                - 0 <= p1 <= 1                 
                - p0 >= p1
                - p1 <= 1 - p0
            potential options for confidence calculation comparing the divergence between the two highest probabilities:
                - 2(p0-p1/(1+p0-p1))
                - p0 - p1  # current choice, stable increase in confidence with increasing divergence between p0 and p1
                - (p0-p1)^2
        '''
        confidence = sorted_probs[0] - sorted_probs[1]
    else:
        #complement of the normalized shannon entropy
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
def run_expanding_classification(subjects, threshold, patience, confidence_type, initial_window_length, expansion_rate, sfreq, csp_components):
    scores_across_subjects = []
    kappa_across_subjects = []
    prediction_time_across_subjects = []

    subjects_accuracies = []
    subjects_kappa = []
    subjects_prediction_times = []
    #confusion matrix
    number_cm = 0 
    cm = np.zeros((4,4))
    current_person = 0
    for person in subjects:
        current_person += 1
        print("Person %d" % (person))
        subject= [person]
        epochs, labels = make_data(subject)
        #labels = epochs.events[:, -1] - 4
        epochs_data = epochs.get_data(copy=False)

        #get the training set - first session of the data
        train_indexes = [i for i, epoch in enumerate(epochs) if epochs[i].metadata['session'].iloc[0] == '0train']
        train_data = epochs_data[train_indexes]
        train_labels = labels[train_indexes]

        #get the test set - second session of the data
        test_indexes = [i for i, epoch in enumerate(epochs) if epochs[i].metadata['session'].iloc[0] == '1test']
        test_data = epochs_data[test_indexes]
        test_labels = labels[test_indexes]

        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components=csp_components, reg=None, log=True, norm_trace=False)
        current_cv = 0 
        
        #Training
        current_cv += 1
        X_train = csp.fit_transform(train_data, train_labels)
        lda.fit(X_train, train_labels)
        w_start = np.arange(0, epochs_data.shape[2] - initial_window_length, expansion_rate) 
        #Testing/inference
        scores_across_epochs = []
        kappa_across_epochs = []
        predict_time_across_epochs = []
        for epoch_idx in range(len(test_indexes)):
            previous_class_index = None
            predict = False
            numTimesThresholdCrossed = 0
            for n, window_start in enumerate(w_start):
                window_length = initial_window_length + n * expansion_rate
                X_test_window = csp.transform(test_data[:, :,  w_start[0]:window_length])
                X_test_epoch_window = X_test_window[epoch_idx]
                probabilities = lda.predict_proba([X_test_epoch_window])
                probabilities = np.array(probabilities)
                probabilities = probabilities.flatten()
                predict, numTimesThresholdCrossed,  previous_class_index = early_pred(
                    probabilities, predict, numTimesThresholdCrossed, patience, confidence_type, threshold,  previous_class_index
                )
                if predict:
                    #score
                    score = lda.score(X_test_epoch_window.reshape(1, -1), [test_labels[epoch_idx]])
                    scores_across_epochs.append(score)

                    # Calculate kappa for the window
                    kappa = cohen_kappa_score(lda.predict(X_test_window), test_labels)
                    kappa_across_epochs.append(kappa)

                    #prediction time
                    predict_time = window_length /sfreq + epochs.tmin
                    #predict_time = (predict_time + window_length) / sfreq + epochs.tmin
                    predict_time_across_epochs.append(predict_time)

                    #Confusion matrix
                    predictions = lda.predict(X_test_window)
                    cm = np.array(cm) + np.array(confusion_matrix(test_labels, predictions, labels = ['left_hand', 'right_hand', 'tongue', 'feet']))
                    number_cm +=1
                    break
            else:
                    #score
                    score = lda.score(X_test_epoch_window.reshape(1, -1), [test_labels[epoch_idx]])
                    scores_across_epochs.append(score)

                    # Calculate kappa for the window
                    kappa = cohen_kappa_score(lda.predict(X_test_window), test_labels)
                    kappa_across_epochs.append(kappa)

                    #prediction time
                    #predict_time = n
                    predict_time =  window_length/sfreq + epochs.tmin
                    predict_time_across_epochs.append(predict_time)
                    
                    #Confusion matrix
                    predictions = lda.predict(X_test_window)
                    cm = np.array(cm) + np.array(confusion_matrix(test_labels, predictions, labels = ['left_hand', 'right_hand', 'tongue', 'feet']))
                    number_cm +=1

        subjects_accuracies.append(np.mean(scores_across_epochs))
        subjects_prediction_times.append(np.mean(predict_time_across_epochs))
        subjects_kappa.append(np.mean(kappa_across_epochs))

        if current_person == 1:
            scores_across_subjects  = scores_across_epochs
            prediction_time_across_subjects = predict_time_across_epochs
            kappa_across_subjects = kappa_across_epochs
        else:
            scores_across_subjects = np.vstack((scores_across_subjects, scores_across_epochs))
            prediction_time_across_subjects = np.vstack((prediction_time_across_subjects, predict_time_across_epochs))
            kappa_across_subjects = np.vstack((kappa_across_subjects, kappa_across_epochs))
    #accuracy
    mean_scores_across_subjects = np.mean(scores_across_subjects, axis=0)
    accuracy = np.mean(mean_scores_across_subjects) #single number
    #kappa
    mean_kappa_across_subjects = np.mean(kappa_across_subjects)
    kappa = mean_kappa_across_subjects# single number
    #prediction time
    mean_prediction_time_across_subjects = np.mean(prediction_time_across_subjects, axis=0)
    prediction_time = np.mean(mean_prediction_time_across_subjects) #single number
    # calculate average confusion 
    cm = np.divide(cm, number_cm)
    return accuracy, kappa, prediction_time, cm, subjects_accuracies, subjects_prediction_times, subjects_kappa


def evaluate_and_plot(accuracy_array, prediction_time_array, kappa_array, threshold_values, patience_values, initial_window_length, sfreq,confidence_type):
    threshold_labels = [f'{threshold:.1f}' for threshold in threshold_values]
    labels = epochs_info(labels = True)

    # A formality as classes are balanced
    classes = ['left_hand', 'right_hand', 'tongue', 'feet']
    class_balance = np.zeros(4)
    for i, class_type in enumerate(classes):
        class_balance[i] = np.mean(labels == class_type)
    class_balance = np.max(class_balance)
    print("class balance",  class_balance)

    tmin, tmax = epochs_info(tmin = True, tmax = True)
    onset = tmin
    offset = tmax
    #patience_values = (patience_values * initial_window_length) / sfreq # ? patience values cant be made into seconds for the expanding window
    # Plotting accuracy
    plt.figure()
    for i in range(len(accuracy_array)):
        plt.plot(patience_values, accuracy_array[i], label=f'Threshold {threshold_labels[i]}', linestyle='-', marker='o')

    plt.xlabel('Patience')
    plt.ylabel('Accuracy')
    plt.axhline(class_balance, linestyle="-", color="k", label="Chance")
    if confidence_type == 'highest_prob':
       plt.title(f"Accuracy vs Patience (Highest Probability): LDA - Dynamic - Expanding model", fontsize=8)
    elif confidence_type == 'difference_two_highest':
        plt.title(f"Accuracy vs Patience (Divergence of two highest probability): LDA - Dynamic - Expanding model", fontsize=8)
    else:
        plt.title(f"Accuracy vs Patience (Shannon entropy): LDA - Dynamic - Expanding model", fontsize=8)
    plt.legend()
    plt.grid(True)
    if confidence_type == 'highest_prob':
        plt.savefig(project_root + '/reports/figures/cumulative/LDA/dynamic/expanding/highest_prob/accuracy_thresholds.png')
    elif confidence_type == 'difference_two_highest':
        plt.savefig(project_root + '/reports/figures/cumulative/LDA/dynamic/expanding/difference_two_highest/accuracy_thresholds.png')
    else:
        plt.savefig(project_root + '/reports/figures/cumulative/LDA/dynamic/expanding/neg_norm_shannon/accuracy_thresholds.png')
    plt.show()

    tmin, tmax = epochs_info(tmin = True, tmax = True)
    onset = tmin
    offset = tmax
    prediction_time_array = prediction_time_array 
   # Prediciton time
    plt.figure()
    for i in range(len(prediction_time_array)):
        plt.plot(patience_values, prediction_time_array[i], label=f'Threshold {threshold_labels[i]}', linestyle='-', marker='o')

    plt.xlabel('Patience')
    plt.ylabel('Prediction Time')
    plt.axhline(onset, linestyle="--", color="r", label="Onset")
    plt.axhline(offset, linestyle="--", color="b", label="Offset")
    if confidence_type == 'highest_prob':
       plt.title(f"Prediction Time vs Patience (Highest Probability): LDA - Dynamic - Expanding model", fontsize=8)
    elif confidence_type == 'difference_two_highest':
        plt.title(f"Prediction Time vs Patience (Divergence of two highest probability): LDA - Dynamic - Expanding model", fontsize=8)
    else:
        plt.title(f"Prediction Timevs Patience (Shannon entropy): LDA - Dynamic - Expanding model", fontsize=8)
    plt.legend()
    plt.grid(True)
    if confidence_type == 'highest_prob':
        plt.savefig(project_root + '/reports/figures/cumulative/LDA/dynamic/expanding/highest_prob/prediction_time_thresholds.png')
    elif confidence_type == 'difference_two_highest':
        plt.savefig(project_root + '/reports/figures/cumulative/LDA/dynamic/expanding/difference_two_highest/prediction_time_thresholds.png')
    else:
        plt.savefig(project_root + '/reports/figures/cumulative/LDA/dynamic/expanding/neg_norm_shannon/prediction_time_thresholds.png')
    plt.show()

    # Kappa
    plt.figure()
    for i in range(len(kappa_array)):
        plt.plot(patience_values, kappa_array[i], label=f'Threshold {threshold_labels[i]}', linestyle='-', marker='o')

    plt.xlabel('Patience')
    plt.ylabel('Kappa')
    if confidence_type == 'highest_prob':
       plt.title(f"Kappa vs Patience (Highest Probability): LDA - Dynamic - Expanding model", fontsize=8)
    elif confidence_type == 'difference_two_highest':
        plt.title(f"Kappa vs Patience (Divergence of two highest probability): LDA - Dynamic - Expanding model", fontsize=8)
    else:
        plt.title(f"Kappa vs Patience (Shannon entropy): LDA - Dynamic - Expanding model", fontsize=8)
    plt.legend()
    plt.grid(True)
    if confidence_type == 'highest_prob':
        plt.savefig(project_root + '/reports/figures/cumulative/LDA/dynamic/expanding/highest_prob/kappa_thresholds.png')
    elif confidence_type == 'difference_two_highest':
        plt.savefig(project_root + '/reports/figures/cumulative/LDA/dynamic/expanding/difference_two_highest/kappa_thresholds.png')
    else:
        plt.savefig(project_root + '/reports/figures/cumulative/LDA/dynamic/expanding/neg_norm_shannon/kappa_thresholds.png')
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
def plot_confusion_matrix(cm):
    plt.figure()
    #confusion matrix
    displaycm = ConfusionMatrixDisplay(cm, display_labels=['left_hand', 'right_hand', 'tongue', 'feet'])
    displaycm.plot()
    plt.title(f"Confusion Matrix : LDA - Dynamic - Expanding model", fontsize=12)
    plt.grid(False)
    confusion_matrix_save_path = os.path.join(project_root, 'reports/figures/cumulative/LDA/dynamic/expanding/confusionMatrix.png')
    plt.savefig(confusion_matrix_save_path)
    plt.show()

def write(accuracy_array, prediction_time_array, kappa_array):
    f = open(project_root + "/reports/figures/cumulative/LDA/dynamic/expanding/lda_dynamic_expanding_accuracy.txt", "w")
    f.write(f"Classification accuracy across all patience and thresholds: {np.mean(np.mean(accuracy_array, 0))}\n")
    f.close()
    g = open(project_root + "/reports/figures/cumulative/LDA/dynamic/expanding/lda_dynamic_expanding_kappa.txt", "w")
    g.write(f"Average kappa across all patience and thresholds: {np.mean(np.mean(kappa_array, 0))}\n")
    g.close()
    h = open(project_root + "/reports/figures/cumulative/LDA/dynamic/expanding/lda_dynamic_expanding_prediction_time.txt", "w")
    h.write(f"Average prediction times across all patience and thresholds: {np.mean(np.mean(prediction_time_array, 0))}\n")
    h.close()

#calculate the information transfer rate
def calculate_best_itr(best_itr, best_itr_patience, best_itr_threshold, best_confidence_type, accuracy, patience, threshold, confidence_type):
    number_classes = 4 # we have classes in this dataset
    current_itr = log2(number_classes) + accuracy*log2(accuracy)+(1-accuracy)*log2((1-accuracy)/(number_classes-1))
    if current_itr > best_itr:
        best_itr_patience = patience
        best_itr_threshold = threshold
        best_itr = current_itr
        best_confidence_type = confidence_type
    return best_itr, best_itr_patience, best_itr_threshold, best_confidence_type

def main_lda_dynamic_expanding():
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9] 
    sfreq = 250         
    confidence_types = ['highest_prob','difference_two_highest', 'neg_norm_shannon' ]
    #Use tuned hyperparams from best params! later work
    #best_params_intial_length = 175
    #best_params_expansion_rate = 175 
    #best_csp_components = 4

    csp_components = 8
    initial_window_length = 475 #int(sfreq * 0.5)  
    expansion_rate = 225 #int(sfreq * 0.1)   
    w_start= np.arange(0, epochs_info(length= True) - initial_window_length, expansion_rate) 
    print("w_start:", w_start )

    patience_values = np.arange(1, len(w_start)) 
    print("patience_values: ", patience_values)
    print("len patience: ", len(patience_values))
    threshold_values = np.arange(0.1, 1, 0.1)
    print("threshold_values: ", threshold_values)
    print("len threshold: ", len(threshold_values))

    tmin, tmax = epochs_info(tmin = True, tmax = True)
    print(f"\n\n\ntmin {tmin}, tmax {tmax}")
    # evaluate everything for each of the 3 methods
    for confidence_type in confidence_types:
        # array to hold the average accuracy and prediction times with size len(confidence_type) x len(thre)
        accuracy_array = []
        prediction_time_array = []
        kappa_array = []
    # over threshold values
        best_itr = 0
        best_itr_threshold = 0
        best_itr_patience = 0
        best_confidence_type = 'none'
        for n, threshold in enumerate(threshold_values):
            accuracy_row = []
            prediction_time_row = []
            kappa_array_row = []
            # over patience values
            for m, patience in enumerate(patience_values):
                print("\n")
                print(f"Threshold:{n+1}/{len(threshold_values)},  Patience: {m+1}/{len(patience_values)}")
                print("\n")
                #given the varaibles, provide the average accuracy and prediction times (early prediction)
                accuracy, kappa, prediction_time, _, _, _ , _ = run_expanding_classification(subjects, threshold, patience, confidence_type, initial_window_length, expansion_rate, sfreq, csp_components)
                best_itr, best_itr_patience, best_itr_threshold, best_confidence_type = calculate_best_itr(best_itr, best_itr_patience, best_itr_threshold, best_confidence_type, accuracy, patience, threshold, confidence_type)
                accuracy_row.append(accuracy)
                prediction_time_row.append(prediction_time)
                kappa_array_row.append(kappa)
            accuracy_array.append(accuracy_row)
            prediction_time_array.append(prediction_time_row)
            kappa_array.append(kappa_array_row)
        #Plotting the average accuracy and prediction times (early prediction) as well as the different threshold and patience values across subjects for each of the confidence types
        accuracy_array = np.array(accuracy_array)
        prediction_time_array = np.array(prediction_time_array)
        kappa_array = np.array(kappa_array)
    write(accuracy_array, prediction_time_array, kappa_array)
    print(f"Classification of best patience and threshold based on itr")
    accuracy, kappa, prediction_time, cm, subjects_accuracies, subjects_prediction_times, subjects_kappa = run_expanding_classification(subjects, best_itr_threshold, best_itr_patience, confidence_type, initial_window_length, expansion_rate, sfreq, csp_components)
    #evaluate_and_plot(accuracy_array, prediction_time_array, kappa_array, threshold_values, patience_values, initial_window_length, sfreq, confidence_type)
    plot_confusion_matrix(cm)
    print(accuracy)
    print(kappa)
    print(prediction_time)
    print(subjects_accuracies)
    print(subjects_kappa)
    print(subjects_prediction_times)
    return accuracy, kappa, prediction_time, subjects_accuracies, subjects_prediction_times, subjects_kappa  #best expanding model
if __name__ == "__main__": 
    main_lda_dynamic_expanding()                