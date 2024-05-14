import os
import sys
import numpy as np
from math import log2
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import entropy
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
from mne.decoding import CSP

from sklearn.model_selection import ParameterSampler
import numpy as np

current_directory = os.path.abspath('')

project_root = current_directory

sys.path.append(project_root)
print("ROOT:", project_root)

VALID_CONFIDENCE_TYPES = {'highest_prob', 'difference_two_highest', 'neg_norm_shannon'}
#Static hyperparameter tuning
from Early_predict_UQ.data.make_dataset import make_data
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
def run_expanding_classification_dynamic(subjects, threshold, patience, confidence_type, initial_window_length, expansion_rate, sfreq, csp_components):
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

#Expanding window - classification - tuning using kfold cross validation
   ##calculate kappa and accuracy at each window step
def run_expanding_classification_tuning_static(subjects, initial_window_length, expansion_rate, csp_components):
    scores_across_subjects = []
    kappa_across_subjects = []

    subjects_accuracies =[]
    subjects_kappa = []
    current_person = 0
    for person in subjects:
        current_person += 1
        print("Person %d" % (person))
        subject= [person]
        epochs, labels = make_data(subject)

        #get the training set - first session of the data for each subject
        train_indexes = [i for i, epoch in enumerate(epochs) if epochs[i].metadata['session'].iloc[0] == '0train']
        epochs_data = epochs.get_data(copy = False)
        train_data = epochs_data[train_indexes]
        train_labels = labels[train_indexes]


        cv = KFold(n_splits=10, shuffle = True, random_state=42)

        scores_cv_splits = []
        kappa_cv_splits  = []

        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components= csp_components, reg=None, log=True, norm_trace=False)
        current_cv = 0 
        for train_idx, test_idx in cv.split(train_data):
            current_cv += 1
            y_train, y_test = train_labels[train_idx], train_labels[test_idx]
            X_train = csp.fit_transform(train_data[train_idx], y_train)
            lda.fit(X_train, y_train)
            w_start = np.arange(0, train_data.shape[2] - initial_window_length, expansion_rate) 
            scores_across_epochs = []
            kappa_across_epochs = []
            for n, window_start in enumerate(w_start):
                window_length = initial_window_length + n * expansion_rate
                X_test_window = csp.transform(train_data[test_idx][:, :,  w_start[0]:window_length])
                #accuracy
                score = lda.score(X_test_window, y_test)
                scores_across_epochs.append(score)

                #kappa
                kappa = cohen_kappa_score(lda.predict(X_test_window), y_test) 
                kappa_across_epochs.append(kappa)
            if current_cv == 1:
                scores_cv_splits = np.array(scores_across_epochs)
                kappa_cv_splits = np.array(kappa_across_epochs)
            else:
                scores_cv_splits = np.vstack((scores_cv_splits,np.array(scores_across_epochs)))
                kappa_cv_splits = np.vstack((kappa_cv_splits,np.array(kappa_across_epochs)))

        mean_scores_across_cv = np.mean(scores_cv_splits, axis=0)
        mean_kappa_across_cv = np.mean(kappa_cv_splits, axis=0)
        if current_person == 1:
            scores_across_subjects  = np.array(mean_scores_across_cv)
            kappa_across_subjects  = np.array(mean_kappa_across_cv)
        else:
            scores_across_subjects = np.vstack((scores_across_subjects,np.array(mean_scores_across_cv)))
            kappa_across_subjects = np.vstack((kappa_across_subjects,np.array(mean_kappa_across_cv)))

        subjects_accuracies.append(np.mean(mean_scores_across_cv))
        subjects_kappa.append(np.mean(mean_kappa_across_cv))

    mean_scores_across_subjects = np.mean(scores_across_subjects, axis=0)
    mean_kappa_across_subjects = np.mean(kappa_across_subjects, axis=0)
    accuracy = mean_scores_across_subjects
    kappa = mean_kappa_across_subjects

    return subjects_accuracies, scores_across_subjects, subjects_kappa, kappa_across_subjects, accuracy, kappa

def create_parameterslist(sfreq):
    rng = np.random.RandomState(42)

    #max intial_length and epxansion rate to be 1/4 of the trial during the MI task, or 1 seconds
    initial_window_length = np.round(rng.uniform(0.1, 2, 10), 1) 
    expansion_rate = np.round(rng.uniform(0.1, 2, 10), 1)

    parameters = {  
    # 1, 2 or 3 filters for each class, as we have 4 classes.
    'csp_components': [4,8], # more than 8 was reading into noise - occipital lobe pattern
    'initial_window_length': np.round(sfreq * initial_window_length).astype(int), 
    'expansion_rate':  np.round(sfreq * expansion_rate).astype(int)
    }

    #Random search parameters - n_tier sets of parameter values
    parameters_list = list(ParameterSampler(parameters, n_iter= 10, random_state=rng))
    return parameters_list

#Random search and return the best parameter values and its accuracy
def hyperparameter_tuning (parameters_list, subjects):
    mean_accuracy = 0
    best_accuracy = 0
    for n, param_set in enumerate(parameters_list):
        csp_components = param_set['csp_components']
        initial_window_length = param_set['initial_window_length'] 
        expansion_rate =  param_set['expansion_rate']

        _, _, _, _, accuracy, _ = run_expanding_classification_tuning_static(subjects, initial_window_length, expansion_rate, csp_components)
        #only optimized for best accuracy here, maybe kappa too?
        mean_accuracy = np.mean(accuracy)

        print(f"Iteration {n+1}/{len(parameters_list)}: Mean accuracy for parameters {param_set} is {mean_accuracy}")

        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_params_expanding = param_set

    return best_params_expanding, best_accuracy

#Access general epoch information
def epochs_info(labels=False, tmin=False, length=False, info = False):
    global epochs_data
    global labels_data
    if labels or tmin or length or info:
        epochs, labels_data = make_data([1])
        epochs_data = epochs.get_data(copy=False)

    if labels and tmin:
        return labels_data, epochs.tmin
    elif labels and length:
        return labels_data, epochs_data.shape[2]
    elif tmin and length:
        return epochs.tmin, epochs_data.shape[2]
    elif labels:
        return labels_data
    elif tmin:
        return epochs.tmin
    elif length:
        return epochs_data.shape[2]
    elif info:
        return epochs.info
    else:
        raise ValueError("At least one of 'labels', 'tmin', 'length', or 'info' must be True.")
    
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

def main():
    subjects = [1,2,3,4,5,6,7,8,9]  # 9 subjects
    sfreq = 250    # Sampling frequency - 250Hz
    '''
    Expanding phase
    '''
    #Hyperparameter_tuning for csp, and expanding window parameters using the static model
    print("\n\n Hyperparameter tuning: \n\n")
    parameters_list = create_parameterslist(sfreq)
    best_params_expanding, best_accuracy = hyperparameter_tuning(parameters_list, subjects)

    #Find optimal confidence type and patience from the expanding dynamic model using itr
    csp_components = best_params_expanding['csp_components']
    initial_window_length = best_params_expanding['initial_window_length']
    expansion_rate = best_params_expanding['expansion_rate']
    w_start= np.arange(0, epochs_info(length= True) - initial_window_length, expansion_rate) 
    confidence_types = ['highest_prob','difference_two_highest', 'neg_norm_shannon' ]
    patience_values = np.arange(1, len(w_start)) 
    threshold_values = np.arange(0.1, 1, 0.1)
    tmin, tmax = epochs_info(tmin = True, tmax = True)

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
                accuracy, kappa, prediction_time, _, _, _ , _ = run_expanding_classification_dynamic(subjects, threshold, patience, confidence_type, initial_window_length, expansion_rate, sfreq, csp_components)
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

    #vary the treshold and find accuracy kappa and pred time for the dynamic model
    accuracy_dynamic = []
    kappa_dynamic = []
    prediction_time_dynamic = []
    print("Finding dynamic model accuracy, kappa, and pred time, across thresholds")
    for n, threshold in enumerate(threshold_values):
        print(f"Threshold:{n+1}/{len(threshold_values)}")
        #dynamic model evaluation
        accuracy, kappa, prediction_time, _, _, _ , _ = run_expanding_classification_dynamic(subjects, threshold, patience, confidence_type, initial_window_length, expansion_rate, sfreq, csp_components)
        accuracy_dynamic.append(accuracy)
        kappa_dynamic.append(kappa)
        prediction_time_dynamic.append(prediction_time)

    print("Finding static model accuracy, and kappa for the given prediction times from the dynamic model")
    #for each prediction time from the dynamic model, find the accuracy and kappa from the static model
    for n, prediction_time in enumerate(prediction_time_dynamic):
        print(f"Prediction time:{n+1}/{len(prediction_time_dynamic)}")
        