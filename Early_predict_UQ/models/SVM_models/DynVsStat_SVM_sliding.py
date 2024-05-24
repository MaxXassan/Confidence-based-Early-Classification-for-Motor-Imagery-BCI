from cProfile import label
from locale import windows_locale
import os
import sys
import numpy as np
from math import log2
import matplotlib.pyplot as plt
from sklearn.svm import SVC
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
def run_sliding_classification_dynamic(subjects, threshold, patience, confidence_type, w_length, w_step, sfreq, csp_components, c , kernel, gamma = None):
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

        if kernel == 'linear':
            svm = SVC(C = c, kernel =kernel, probability = True)
        else:
            svm = SVC(C = c, kernel =kernel, gamma = gamma, probability = True)
        csp = CSP(n_components=csp_components, reg=None, log=True, norm_trace=False)
        current_cv = 0 
        
        #Training
        current_cv += 1
        X_train = csp.fit_transform(train_data, train_labels)
        svm.fit(X_train, train_labels)
        w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step) 
        #Testing/inference
        scores_across_epochs = []
        kappa_across_epochs = []
        predict_time_across_epochs = []
        for epoch_idx in range(len(test_indexes)):
            previous_class_index = None
            predict = False
            numTimesThresholdCrossed = 0
            for n in w_start:
                X_test_window = csp.transform(test_data[:, :, n:(n + w_length)])
                X_test_epoch_window = X_test_window[epoch_idx]
                probabilities = svm.predict_proba([X_test_epoch_window])
                probabilities = np.array(probabilities)
                probabilities = probabilities.flatten()
                predict, numTimesThresholdCrossed,  previous_class_index = early_pred(
                    probabilities, predict, numTimesThresholdCrossed, patience, confidence_type, threshold,  previous_class_index
                )
                if predict:
                    #score
                    score = svm.score(X_test_epoch_window.reshape(1, -1), [test_labels[epoch_idx]])
                    scores_across_epochs.append(score)

                    # Calculate kappa for the window
                    kappa = cohen_kappa_score(svm.predict(X_test_window), test_labels)
                    kappa_across_epochs.append(kappa)

                    #prediction time
                    predict_time = n/sfreq + epochs.tmin
                    #predict_time = (predict_time + window_length) / sfreq + epochs.tmin
                    predict_time_across_epochs.append(predict_time)

                    #Confusion matrix
                    predictions = svm.predict(X_test_window)
                    cm = np.array(cm) + np.array(confusion_matrix(test_labels, predictions, labels = ['left_hand', 'right_hand', 'tongue', 'feet']))
                    number_cm +=1
                    break
            else:
                    #score
                    score = svm.score(X_test_epoch_window.reshape(1, -1), [test_labels[epoch_idx]])
                    scores_across_epochs.append(score)

                    # Calculate kappa for the window
                    kappa = cohen_kappa_score(svm.predict(X_test_window), test_labels)
                    kappa_across_epochs.append(kappa)

                    #prediction time
                    #predict_time = n
                    predict_time = n/sfreq + epochs.tmin
                    predict_time_across_epochs.append(predict_time)
                    
                    #Confusion matrix
                    predictions = svm.predict(X_test_window)
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

#Sldiding window - classification - tuning using kfold cross validation
   ##calculate kappa and accuracy at each window step
def run_sliding_classification_tuning_static(subjects, w_length, w_step, csp_components, c, kernel, gamma, pred_times = False):
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


        cv = KFold(n_splits=5, shuffle = True, random_state=42)

        scores_cv_splits = []
        kappa_cv_splits  = []

        if kernel == 'linear':
            svm = SVC(C = c, kernel =kernel, probability = True)
        else:
            svm = SVC(C = c, kernel =kernel, gamma = gamma, probability = True)
        csp = CSP(n_components= csp_components, reg=None, log=True, norm_trace=False)
        current_cv = 0 
        for train_idx, test_idx in cv.split(train_data):
            current_cv += 1
            y_train, y_test = train_labels[train_idx], train_labels[test_idx]
            X_train = csp.fit_transform(train_data[train_idx], y_train)
            svm.fit(X_train, y_train)
            w_start = np.arange(0, train_data.shape[2] - w_length, w_step) 
            scores_across_epochs = []
            kappa_across_epochs = []
            if pred_times:
                #loop over the given prediction times by the dynamic model
                for n in pred_times:
                    n = np.round(250 * n).astype(int)  #250 is the sample frequency
                    X_test_window = csp.transform(train_data[test_idx][:, :, n:(n + w_length)])
                    #accuracy
                    score = svm.score(X_test_window, y_test)
                    scores_across_epochs.append(score)
                    #kappa
                    kappa = cohen_kappa_score(svm.predict(X_test_window), y_test) 
                    kappa_across_epochs.append(kappa)
                if current_cv == 1:
                    scores_cv_splits = np.array(scores_across_epochs)
                    kappa_cv_splits = np.array(kappa_across_epochs)
                else:
                    scores_cv_splits = np.vstack((scores_cv_splits,np.array(scores_across_epochs)))
                    kappa_cv_splits = np.vstack((kappa_cv_splits,np.array(kappa_across_epochs)))
            else:
                for n in w_start:
                    X_test_window = csp.transform(train_data[test_idx][:, :, n:(n + w_length)])
                    #accuracy
                    score = svm.score(X_test_window, y_test)
                    scores_across_epochs.append(score)

                    #kappa
                    kappa = cohen_kappa_score(svm.predict(X_test_window), y_test) 
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
 
def run_sliding_classification_static(subjects, w_length, w_step, csp_components, c , kernel, gamma = None, pred_times = False):
    scores_across_subjects = []
    kappa_across_subjects = []

    subjects_accuracies = []
    subjects_kappa = []
    current_person = 0

    #confusion matrix
    number_cm = 0 
    cm = np.zeros((4,4))
    for person in subjects:
        current_person += 1
        subject= [person]
        epochs, labels = make_data(subject)
        epochs_data = epochs.get_data(copy=False)
        
        #get the training set - first session of the data
        train_indexes = [i for i, epoch in enumerate(epochs) if epochs[i].metadata['session'].iloc[0] == '0train']
        train_data = epochs_data[train_indexes]
        train_labels = labels[train_indexes]

         #get the test set - second session of the data
        test_indexes = [i for i, epoch in enumerate(epochs) if epochs[i].metadata['session'].iloc[0] == '1test']
        test_data = epochs_data[test_indexes]
        test_labels = labels[test_indexes]

        if kernel == 'linear':
            svm = SVC(C = c, kernel =kernel, probability = True)
        else:
            svm = SVC(C = c, kernel =kernel, gamma = gamma, probability = True)
        csp = CSP(n_components=csp_components, reg=None, log=True, norm_trace=False)

        # Fit CSP on training data
        X_train = csp.fit_transform(train_data, train_labels)

        # Fit SVM on training data
        svm.fit(X_train, train_labels)

        # Initialize lists to store scores and kappa values for each window
        scores_across_epochs = []
        kappa_across_epochs = []
        w_start = np.arange(0, train_data.shape[2] - w_length, w_step)
        if pred_times:
            for n in pred_times:
                n = np.round(250 * n).astype(int)  #250 is the sample frequency
                X_test_window = csp.transform(test_data[:, :, n:(n + w_length)])
                
                # Calculate accuracy for the window
                score = svm.score(X_test_window, test_labels)
                scores_across_epochs.append(score)

                # Calculate kappa for the window
                kappa = cohen_kappa_score(svm.predict(X_test_window), test_labels)
                kappa_across_epochs.append(kappa)
                
                #Confusion matrix
                predictions = svm.predict(X_test_window)
                cm = np.array(cm) + np.array(confusion_matrix(test_labels, predictions, labels = ['left_hand', 'right_hand', 'tongue', 'feet']))
                number_cm +=1
            if current_person == 1:
                scores_across_subjects  = np.array(scores_across_epochs)
                kappa_across_subjects = np.array(kappa_across_epochs)
            else:
                scores_across_subjects = np.vstack((scores_across_subjects,np.array(scores_across_epochs)))
                kappa_across_subjects = np.vstack((kappa_across_subjects,np.array(kappa_across_epochs)))
        else:
            for n in w_start:
                X_test_window = csp.transform(test_data[:, :, n:(n + w_length)])
                
                # Calculate accuracy for the window
                score = svm.score(X_test_window, test_labels)
                scores_across_epochs.append(score)

                # Calculate kappa for the window
                kappa = cohen_kappa_score(svm.predict(X_test_window), test_labels)
                kappa_across_epochs.append(kappa)
                
                #Confusion matrix
                predictions = svm.predict(X_test_window)
                cm = np.array(cm) + np.array(confusion_matrix(test_labels, predictions, labels = ['left_hand', 'right_hand', 'tongue', 'feet']))
                number_cm +=1
            if current_person == 1:
                scores_across_subjects  = np.array(scores_across_epochs)
                kappa_across_subjects = np.array(kappa_across_epochs)
            else:
                scores_across_subjects = np.vstack((scores_across_subjects,np.array(scores_across_epochs)))
                kappa_across_subjects = np.vstack((kappa_across_subjects,np.array(kappa_across_epochs)))
        #mean accuracy and kappa for each subject
        subjects_accuracies.append(np.mean(scores_across_epochs))
        subjects_kappa.append(np.mean(kappa_across_epochs))

    #mean score for each window for each subject
    mean_score_across_subjects = np.mean(scores_across_subjects, axis=0)
    mean_kappa_across_subjects = np.mean(kappa_across_subjects, axis=0)
    # list of mean accuracy for each window
    accuracy = mean_score_across_subjects
    kappa = mean_kappa_across_subjects
    # confusion matrix 
    cm = np.divide(cm, number_cm)
    return subjects_accuracies, scores_across_subjects, subjects_kappa, kappa_across_subjects, accuracy, kappa, cm, csp

'''Create parameter search space
    - Sample n_iter sets of parameters - set to 60 iterations. 
    - 60 iterations lets us reach the 95% confidence interval that we have found a set of parameters with "good values",
      if the good space covers only 5% of the search space.
    - 1-0.95^(60) = 0.953 > 0.95
    (Random Search for Hyper-Parameter Optimization - Bergstra & Bengio (2012))
'''

def create_parameterslist(sfreq):
    rng = np.random.RandomState(42)
    
    w_length_values = np.round(rng.uniform(0.1, 1, 10), 2)

    w_step_values = []
    for w_length in w_length_values:
        max_step = max(0.1, 0.99 * w_length)
        w_step = np.round(rng.uniform(0.1, max_step), 1) 
        w_step_values.append(w_step)  
        
    w_length_values = np.round(np.array(w_length_values) * sfreq).astype(int)
    w_step_values = np.round(np.array(w_step_values) * sfreq).astype(int)
    
    parameters = {  
        'csp_components': [4, 8], 
        'w_length': w_length_values, 
        'w_step': w_step_values,
        'C': [0.1, 1, 10, 100, 1000],  
        'kernel': ['linear', 'rbf'] 
    }

    parameters_list = list(ParameterSampler(parameters, n_iter=20, random_state=rng))
    for param_set in parameters_list:
        if param_set['kernel'] == 'rbf':
            param_set['gamma'] = rng.choice([1, 0.1, 0.01, 0.001, 0.0001])

    return parameters_list



#Random search and return the best parameter values and its accuracy
def hyperparameter_tuning (parameters_list, subjects):
    mean_accuracy = 0
    best_accuracy = 0
    for n, param_set in enumerate(parameters_list):
        csp_components = param_set['csp_components']
        w_length = param_set['w_length'] 
        w_step =  param_set['w_step']
        c = param_set['C']
        kernel = param_set['kernel']
        if kernel == 'rbf':
            gamma = param_set['gamma']
            _ , accuracy = run_sliding_classification_tuning_static(subjects, w_length, w_step, csp_components, c, kernel, gamma)
        else:
            _ , accuracy = run_sliding_classification_tuning_static(subjects, w_length, w_step, csp_components, c, kernel, gamma = None)
        
        mean_accuracy = np.mean(accuracy)

        print(f"Iteration {n+1}/{len(parameters_list)}: Mean accuracy for parameters {param_set} is {mean_accuracy}")

        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_params = param_set

    return best_params, best_accuracy

#Access general epoch information
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
#calculate the information transfer rate
def calculate_best_itr_dyn_tune(best_itr, best_itr_patience, best_itr_threshold, best_confidence_type, accuracy, prediction_time, patience, threshold, confidence_type):
    number_classes = 4 # we have 4 classes in this dataset
    current_B = log2(number_classes) + accuracy*log2(accuracy)+(1-accuracy)*log2((1-accuracy)/(number_classes-1))
    current_Q = prediction_time
    current_itr = current_B/current_Q
    if current_itr > best_itr:
        best_itr_patience = patience
        best_itr_threshold = threshold
        best_itr = current_itr
        best_confidence_type = confidence_type
    return best_itr, best_itr_patience, best_itr_threshold, best_confidence_type
def calculate_best_itr_dyn(best_itr, accuracy, prediction_time, best_subjects_accuracies_dyn, best_subjects_prediction_times_dyn, best_subjects_kappa_dyn, best_cm_dyn, subjects_accuracies_dyn, subjects_prediction_times_dyn, subjects_kappa_dyn, cm_dyn):
    number_classes = 4 # we have 4 classes in this dataset
    current_B = log2(number_classes) + accuracy*log2(accuracy)+(1-accuracy)*log2((1-accuracy)/(number_classes-1))
    current_Q = prediction_time
    current_itr = current_B/current_Q
    if current_itr > best_itr:
        best_subjects_accuracies_dyn = subjects_accuracies_dyn
        best_subjects_kappa_dyn  = subjects_kappa_dyn 
        best_subjects_prediction_times_dyn = subjects_prediction_times_dyn
        best_cm_dyn = cm_dyn
        best_itr = current_itr
    return best_itr, best_subjects_accuracies_dyn, best_subjects_prediction_times_dyn, best_subjects_kappa_dyn, best_cm_dyn

def plot_confusion_matrix(cm_stat, cm_dyn):
    plt.figure()
    #confusion matrix
    displaycm = ConfusionMatrixDisplay(cm_dyn, display_labels=['left_hand', 'right_hand', 'tongue', 'feet'])
    plt.grid(False)
    displaycm.plot()
    plt.title(f"Confusion Matrix : SVM - Dynamic - Sliding model", fontsize=12)
    plt.grid(False)
    plt.savefig(project_root + '/reports/figures/cumulative/SVM/dynamicVSstatic/Sliding_dynamic_ConfusionMatrix.png')

    plt.figure()
    #confusion matrix
    displaycm = ConfusionMatrixDisplay(cm_stat, display_labels=['left_hand', 'right_hand', 'tongue', 'feet'])
    plt.grid(False)
    displaycm.plot()
    plt.title(f"Confusion Matrix : SVM - Static - Sliding model", fontsize=12)
    plt.grid(False)
    plt.savefig(project_root + '/reports/figures/cumulative/SVM/dynamicVSstatic/Sliding_static_ConfusionMatrix.png')
    
    

#Sliding - SVM
def main_svm_sliding():
    subjects = [1,2,3,4,5,6,7,8,9]  # 9 subjects
    sfreq = 250    # Sampling frequency - 250Hz
    '''
    Hyperparameter tuning
    '''
    #Hyperparameter_tuning for csp, and sliding window parameters using the static model
    print("\n\n Hyperparameter tuning (1): csp and window parameters \n\n")
    parameters_list = create_parameterslist(sfreq)
    best_params_sliding, best_accuracy = hyperparameter_tuning(parameters_list, subjects) #tuned on accuracy as we dont have pred time for itr
    print("\n\n Hyperparameter tuning (1): completed \n\n")
    csp_components = best_params_sliding['csp_components']
    w_length = best_params_sliding['w_length']
    w_step = best_params_sliding['w_step']
    c = best_params_sliding['C']
    kernel = best_params_sliding['kernel']
    if kernel == 'rbf':
        gamma = best_params_sliding['gamma']
    else:
        gamma = None

    w_start= np.arange(0, epochs_info(length= True) -  w_length,  w_step) 
    print("Length: 4 or number total samples-1000 something? ", epochs_info(length= True))

    confidence_types = ['highest_prob','difference_two_highest', 'neg_norm_shannon' ]
    patience_values = np.arange(1, len(w_start)) #need to control this by itr 
    threshold_values = np.arange(0.1, 1, 0.1)
    
    #Find optimal confidence type and patience from the sliding dynamic model using itr
    print("\n\n Hyperparameter tuning (2): patience and confidence type \n\n")
    # over confidence values
    for confidence_type in confidence_types:
        accuracy_array = []
        prediction_time_array = []
        kappa_array = []
        best_itr = 0
        best_itr_threshold = 0
        best_itr_patience = 0
        best_confidence_type = 'none'
        # over threshold values
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
                accuracy, kappa, prediction_time, _, _, _ , _ = run_sliding_classification_dynamic(subjects, threshold, patience, confidence_type, w_length, w_step, sfreq, csp_components, c , kernel, gamma = None)
                best_itr, best_itr_patience, best_itr_threshold, best_confidence_type = calculate_best_itr_dyn_tune(best_itr, best_itr_patience, best_itr_threshold, best_confidence_type, accuracy, prediction_time, patience, threshold, confidence_type)
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
    
    #best_itr_patience = 5
    #best_confidence_type = 'highest_prob'
    print("\n\n Hyperparameter tuning (2): completed\n\n")
    print(f"chosen patience: {best_itr_patience}, chosen confidence_type: {best_confidence_type}")

    '''
    Evaluation
    '''
    #vary the treshold and find accuracy, kappa, and pred time for the dynamic model
    accuracy_dynamic = []
    kappa_dynamic = []
    prediction_time_dynamic = []
    best_subjects_accuracies_dyn = None
    best_subjects_prediction_times_dyn = None
    best_subjects_kappa_dyn = None
    best_cm_dyn = None
    best_itr = 0
    #take the average subjects accuracies
    print("Evaluation (1): Finding dynamic model accuracy, kappa, and pred time, across thresholds")
    for n, threshold in enumerate(threshold_values):
        print(f"Threshold:{n+1}/{len(threshold_values)}")
        #dynamic model evaluation
        accuracy, kappa, prediction_time, cm_dyn, subjects_accuracies_dyn, subjects_prediction_times_dyn, subjects_kappa_dyn = run_sliding_classification_dynamic(subjects, threshold, best_itr_patience, best_confidence_type, w_length, w_step, sfreq, csp_components, c , kernel, gamma = None)
        best_itr, best_subjects_accuracies_dyn, best_subjects_prediction_times_dyn, best_subjects_kappa_dyn, best_cm_dyn = calculate_best_itr_dyn(best_itr, accuracy, prediction_time, best_subjects_accuracies_dyn, best_subjects_prediction_times_dyn, best_subjects_kappa_dyn, best_cm_dyn, subjects_accuracies_dyn, subjects_prediction_times_dyn, subjects_kappa_dyn, cm_dyn)
        accuracy_dynamic.append(accuracy)
        kappa_dynamic.append(kappa)
        prediction_time_dynamic.append(prediction_time)
    
    print("Evaluation (2):Finding static model accuracy, and kappa for the given prediction times from the dynamic model")
    #for each prediction time from the dynamic model, find the accuracy and kappa from the static model
    accuracy_static = []
    kappa_static= []
    subjects_accuracies_stat, scores_across_subjects_stat, subjects_kappa_stat, kappa_across_subjects_stat, accuracy, kappa, cm_stat, _ = run_sliding_classification_static(subjects, w_length, w_step, csp_components, c , kernel, gamma, prediction_time_dynamic)
    #accuracy and kappa are not single numbers here
    accuracy_static = accuracy
    kappa_static = kappa

    '''
    Plotting and storing data
    '''
    #Write the optimal parameters
    print(f"Chosen parameters: \n - csp: {csp_components}, \n - w_length: {w_length},\n - w_step:  {w_step}, \n - confidence_type: {best_confidence_type}, \n - patience: {best_itr_patience} out of {patience_values}")
    h = open(project_root + "/reports/figures/cumulative/SVM/dynamicVSstatic/sliding_model_optimal_parameters", "w")
    h.write(f"Chosen parameters: \n - csp: {csp_components}, \n - w_length: {w_length},\n - w_step:  {w_step}, \n - confidence_type: {best_confidence_type}, \n - patience: {best_itr_patience} out of {patience_values}")
    h.close()

    #write per subject accuracies - dynamic
    subject_tuples = [(i+1, acc) for i, acc in enumerate(best_subjects_accuracies_dyn)]
    sorted_subjects = sorted(subject_tuples, key=lambda x: x[1], reverse=True)
    f = open(project_root + "/reports/figures/cumulative/SVM/dynamicVSstatic/sliding_dynamic_model_accuracy_by_subject", "w")
    f.write(f"Classification accuracy: {np.mean(accuracy_dynamic)}\n")
    for subject, subject_accuracy in sorted_subjects:
        f.write(f"Subject {subject}: Accuracy: {subject_accuracy} \n")
        print(f"Subject {subject}: Accuracy: {subject_accuracy}")
    f.close()

    #write per subject accuracies - static
    subject_tuples = [(i+1, acc) for i, acc in enumerate(subjects_accuracies_stat)]
    sorted_subjects = sorted(subject_tuples, key=lambda x: x[1], reverse=True)

    f = open(project_root + "/reports/figures/cumulative/SVM/dynamicVSstatic/sliding_static_model_accuracy_by_subject", "w")
    f.write(f"Classification accuracy: {np.mean(accuracy)}\n")
    for subject, subject_accuracy in sorted_subjects:
        f.write(f"Subject {subject}: Accuracy: {subject_accuracy} \n")
        print(f"Subject {subject}: Accuracy: {subject_accuracy}")
    f.close()

    #write per subject kappa - dynamic
    subject_tuples = [(i+1, acc) for i, acc in enumerate(best_subjects_kappa_dyn)]
    sorted_subjects = sorted(subject_tuples, key=lambda x: x[1], reverse=True)
    f = open(project_root + "/reports/figures/cumulative/SVM/dynamicVSstatic/sliding_dynamic_model_kappa_by_subject", "w")
    f.write(f"Average kappa: {np.mean(kappa_dynamic)}\n")
    for subject, subject_kappa in sorted_subjects:
        f.write(f"Subject {subject}: Kappa: {subject_kappa} \n")
        print(f"Subject {subject}: Kappa: {subject_kappa}")
    f.close()

    #write per subject kappa - static
    subject_tuples = [(i+1, acc) for i, acc in enumerate(subjects_kappa_stat)]
    sorted_subjects = sorted(subject_tuples, key=lambda x: x[1], reverse=True)

    f = open(project_root + "/reports/figures/cumulative/SVM/dynamicVSstatic/sliding_static_model_kappa_by_subject", "w")
    f.write(f"Average kappa: {np.mean(kappa)}\n")
    for subject, subject_kappa in sorted_subjects:
        f.write(f"Subject {subject}: Kappa: {subject_kappa} \n")
        print(f"Subject {subject}: Kappa: {subject_kappa}")
    f.close()

    #write per subject prediction time - dynamic
    subject_tuples = [(i+1, acc) for i, acc in enumerate(best_subjects_prediction_times_dyn)]
    sorted_subjects = sorted(subject_tuples, key=lambda x: x[1], reverse=True)
    f = open(project_root + "/reports/figures/cumulative/SVM/dynamicVSstatic/sliding_dynamic_model_predtime_by_subject", "w")
    f.write(f"Average prediction time: {np.mean(prediction_time_dynamic)}\n")
    for subject, subject_prediction_time in sorted_subjects:
        f.write(f"Subject {subject}: Prediction time: {subject_prediction_time} \n")
        print(f"Subject {subject}: Prediction time: {subject_prediction_time}")
    f.close()

    #plot accuracies
    plt.figure()
    plt.xlabel('Prediction time')
    plt.ylabel('Accuracy')
    plt.grid(True)
    tmin, tmax = epochs_info(tmin = True, tmax = True)
    onset = tmin
    offset = tmax
    plt.axvline(onset, linestyle="--", color="r", label="Onset")
    plt.axvline(offset, linestyle="--", color="b", label="Offset")
    plt.plot(prediction_time_dynamic,accuracy_dynamic, label="Dynamic model", linestyle='-', marker='o')
    plt.plot(prediction_time_dynamic, accuracy_static, label = "Static model", linestyle='-', marker='o')
    plt.title("Sliding model: Accuracy vs Pred time")
    plt.axhline(0.25, label= "chance")
    plt.legend()
    plt.show()
    plt.savefig(project_root + '/reports/figures/cumulative/SVM/dynamicVSstatic/SlidingAccuracy.png')

    #plot accuracies
    plt.figure()
    plt.xlabel('Prediction time')
    plt.ylabel('Kappa')
    plt.grid(True)
    tmin, tmax = epochs_info(tmin = True, tmax = True)
    onset = tmin
    offset = tmax
    plt.axvline(onset, linestyle="--", color="r", label="Onset")
    plt.axvline(offset, linestyle="--", color="b", label="Offset")
    plt.plot(prediction_time_dynamic,kappa_dynamic, label="Dynamic model", linestyle='-', marker='o')
    plt.plot(prediction_time_dynamic, kappa_static, label = "Static model",linestyle='-', marker='o')
    plt.title("Sliding model: Kappa vs Pred time")
    plt.show()
    plt.legend()
    plt.savefig(project_root + '/reports/figures/cumulative/SVM/dynamicVSstatic/SlidingKappa.png')

    plot_confusion_matrix(cm_stat, best_cm_dyn)
    #for comparison with the sliding model to decide and choose the one that performs better
    return best_itr
'''
to do:
plot the accuracy and kappa plots 
print the chosen traversal methods, csp, window parameters, confidence type, patience, 
introduce way of finding best between sliding vs expanding - make another file for sliding, and output best_itr and use the file that has the best itr
table (dynamic vs static) with pr subject accuracy and kappa(and maybe pred time, but only dynamic has that)
confusion matrix
'''

if __name__ == "__main__":
    main_svm_sliding()
    