import os
import sys
import numpy as np
from math import log2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy
from scipy.stats import entropy
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import mne
import matplotlib.pyplot as plt
import logging
# Set the logging level to ERROR to reduce verbosity
mne.set_log_level(logging.ERROR)
from mne.decoding import CSP
from sklearn.model_selection import ParameterSampler
import seaborn as sns
current_directory = os.path.abspath('')

project_root = current_directory

sys.path.append(project_root)
print("ROOT:", project_root)

VALID_CONFIDENCE_TYPES = {'highest_prob', 'difference_two_highest', 'neg_norm_shannon'}
#Static hyperparameter tuning
from Early_predict_UQ.data.make_dataset import make_data
from Tune_LDA_expanding import tune_lda_expanding
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

    #Stopping rule: If confidence > threshold, and threshold reached n = patience times in a row -> Predict early
    if np.round(confidence, 2) > threshold and not predict:
        numTimesThresholdCrossed += 1
        #Predict early
        if numTimesThresholdCrossed == patience:
            predict = True
    return predict, numTimesThresholdCrossed, previous_class_index

#Given sliding window and stopping values, we average the accuracy and prediction time for the model
def run_expanding_classification_dynamic(subjects, threshold, patience, confidence_type, initial_window_length, expansion_rate, sfreq, csp_components, solver, shrinkage, n_components, store_covariance, tol):
    scores_across_subjects = []
    kappa_across_subjects = []
    prediction_time_across_subjects = []
    itrs_across_subjects = []

    subjects_accuracies = []
    subjects_kappa = []
    subjects_prediction_times = []
    subjects_itrs = []
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
        #if svd vs if not ; tol shrinkage store cov
        if solver == 'svd':
            lda = LinearDiscriminantAnalysis(solver = solver, tol = tol, n_components = n_components)
        else:
            lda = LinearDiscriminantAnalysis(solver = solver, shrinkage = shrinkage, n_components = n_components)
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
        itrs_across_epochs = []
        predictions = []
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

                    #prediction time
                    predict_time = window_length
                    predict_time_across_epochs.append(predict_time)

                    #For kappa, and confusion matrix
                    prediction = lda.predict(X_test_epoch_window.reshape(1, -1))
                    predictions.append(prediction)

                    break
            else:
                    #score
                    score = lda.score(X_test_epoch_window.reshape(1, -1), [test_labels[epoch_idx]])
                    scores_across_epochs.append(score)

                    #prediction time
                    predict_time_last =  window_length#end of the last window
                    predict_time_across_epochs.append(predict_time_last)
                    #For kappa, and confusion matrix
                    prediction = lda.predict(X_test_epoch_window.reshape(1, -1))
                    predictions.append(prediction)
        #Information transfer rate
        _, _, _, _, _, _, itr = calculate_best_itr_dyn(best_itr = 0, accuracy = np.mean(scores_across_epochs), prediction_time = np.mean(predict_time_across_epochs), best_subjects_accuracies_dyn= None, best_subjects_prediction_times_dyn= None, best_subjects_kappa_dyn= None, best_subjects_itrs_dyn= None, best_cm_dyn= None, subjects_accuracies_dyn= None, subjects_prediction_times_dyn= None, subjects_kappa_dyn= None, subjects_itrs_dyn = None, cm_dyn = None)
        itrs_across_epochs = itr #single number
        itrs_across_subjects.append(itr)
        #Kappa
        kappa_score = cohen_kappa_score(predictions, test_labels)
        kappa_across_epochs =  kappa_score #single number
        kappa_across_subjects.append(kappa_score)
        #Confusion matrix
        cm = np.array(cm) + np.array(confusion_matrix(test_labels, predictions, labels = ['left_hand', 'right_hand', 'tongue', 'feet']))
        number_cm +=1
        if current_person == 1:
            scores_across_subjects  = scores_across_epochs
            prediction_time_across_subjects = predict_time_across_epochs
        else:
            scores_across_subjects = np.vstack((scores_across_subjects, scores_across_epochs))
            prediction_time_across_subjects = np.vstack((prediction_time_across_subjects, predict_time_across_epochs))
        subjects_accuracies.append(np.mean(scores_across_epochs))
        subjects_prediction_times.append(np.mean(predict_time_across_epochs))
        subjects_kappa = np.append(subjects_kappa,kappa_across_epochs)
        subjects_itrs = np.append(subjects_itrs, itrs_across_epochs)
    #accuracy
    mean_scores_across_subjects = np.mean(scores_across_subjects, axis=1)
    accuracy = np.mean(mean_scores_across_subjects) #single number
    #kappa
    mean_kappa_across_subjects = np.array(kappa_across_subjects)
    kappa = np.mean(mean_kappa_across_subjects)# single number
    #prediction time
    mean_prediction_time_across_subjects = np.mean(prediction_time_across_subjects, axis=1) #issue here i think
    prediction_time = np.mean(mean_prediction_time_across_subjects) #single number
    #itr
    mean_itr_across_subjects = np.array(itrs_across_subjects)
    itr = np.mean(mean_itr_across_subjects)
    # calculate average confusion
    cm = np.divide(cm, number_cm)
    return accuracy, kappa, prediction_time, itr, cm, subjects_accuracies, subjects_prediction_times, subjects_kappa, subjects_itrs, mean_scores_across_subjects, mean_kappa_across_subjects, mean_prediction_time_across_subjects, mean_itr_across_subjects

def run_expanding_classification_static(subjects, initial_window_length, expansion_rate, csp_components, pred_times, solver, shrinkage, n_components, tol):
    scores_across_subjects = []
    kappa_across_subjects = []
    itrs_across_subjects = []

    subjects_accuracies = []
    subjects_kappa = []
    subjects_itrs = []
    current_person = 0
    itr_trial = []

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
        #if svd vs if not ; tol shrinkage store cov
        if solver == 'svd':
            lda = LinearDiscriminantAnalysis(solver = solver, tol = tol, n_components = n_components)
        else:
            lda = LinearDiscriminantAnalysis(solver = solver, shrinkage = shrinkage, n_components = n_components)
        csp = CSP(n_components=csp_components, reg=None, log=True, norm_trace=False)

        # Fit CSP on training data
        X_train = csp.fit_transform(train_data, train_labels)

        # Fit LDA on training data
        lda.fit(X_train, train_labels)

        # Initialize lists to store scores and kappa values for each window
        scores_across_epochs = []
        kappa_across_epochs = []
        itrs_across_epochs = []
        w_start = np.arange(0, test_data.shape[2] - initial_window_length, expansion_rate)
        pred_times = np.round(np.array(pred_times)).astype(int)
        for n, window_start in enumerate(pred_times):
            window_length = pred_times[n]
            X_test_window = csp.transform(test_data[:, :, w_start[0]: window_length])

            # Calculate accuracy for the window
            score = lda.score(X_test_window, test_labels)
            scores_across_epochs.append(score)

            # Calculate kappa for the window
            kappa = cohen_kappa_score(lda.predict(X_test_window), test_labels)
            kappa_across_epochs.append(kappa)

            #Confusion matrix
            predictions = lda.predict(X_test_window)
            cm = np.array(cm) + np.array(confusion_matrix(test_labels, predictions, labels = ['left_hand', 'right_hand', 'tongue', 'feet']))
            number_cm +=1
            #ITR across epochs
            _, _, _, _, _, _, itr = calculate_best_itr_dyn(best_itr = 0, accuracy = score, prediction_time = pred_times[n], best_subjects_accuracies_dyn= None, best_subjects_prediction_times_dyn= None, best_subjects_kappa_dyn= None, best_subjects_itrs_dyn= None, best_cm_dyn= None, subjects_accuracies_dyn= None, subjects_prediction_times_dyn= None, subjects_kappa_dyn= None, subjects_itrs_dyn = None, cm_dyn = None)
            itrs_across_epochs.append(itr)
        if current_person == 1:
            scores_across_subjects  = np.array(scores_across_epochs)
            kappa_across_subjects = np.array(kappa_across_epochs)
            itrs_across_subjects = np.array(itrs_across_epochs)
        else:
            scores_across_subjects = np.vstack((scores_across_subjects,np.array(scores_across_epochs)))
            kappa_across_subjects = np.vstack((kappa_across_subjects,np.array(kappa_across_epochs)))
            itrs_across_subjects = np.vstack((itrs_across_subjects,np.array(itrs_across_epochs)))
        #mean accuracy and kappa for each subject
        subjects_accuracies.append(np.mean(scores_across_epochs))
        subjects_kappa.append(np.mean(kappa_across_epochs))
        subjects_itrs.append(np.mean(itrs_across_epochs))

    #mean score for each window for each subject
    mean_score_across_subjects = np.mean(scores_across_subjects, axis=0)
    mean_kappa_across_subjects = np.mean(kappa_across_subjects, axis=0)
    mean_itr_across_subjects = np.mean(itrs_across_subjects, axis = 0)
    # list of mean accuracy for each window
    accuracy = mean_score_across_subjects
    kappa = mean_kappa_across_subjects
    itr = mean_itr_across_subjects

    # confusion matrix
    cm = np.divide(cm, number_cm)
    return subjects_accuracies, scores_across_subjects, subjects_kappa, kappa_across_subjects, subjects_itrs, itrs_across_subjects, accuracy, kappa, itr, cm, csp

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
#current Source but actually details the problems with the method: Yuan et.al.  https://iopscience-iop-org.proxy-ub.rug.nl/article/10.1088/1741-2560/10/2/026014/pdf
# B = log2 N + P log2 P + (1 − P)log2[(1 − P)/(N − 1)]
# Q =  (60/T)
# Bt = ITR  = B * Q (bits/min)
def calculate_best_itr_dyn_tune(best_itr, best_itr_patience, best_itr_threshold, best_confidence_type, accuracy, prediction_time, patience, threshold, confidence_type):
    number_classes = 4 # we have 4 classes in this dataset
    current_B = log2(number_classes) + accuracy*log2(accuracy)+(1-accuracy)*log2((1-accuracy)/(number_classes-1))

    prediction_time = prediction_time/250 #turning prediction time to seconds first
    current_Q =  60/prediction_time 

    current_itr = current_B * current_Q
    if current_itr > best_itr:
        best_itr_patience = patience
        best_itr_threshold = threshold
        best_itr = current_itr
        best_confidence_type = confidence_type
    return best_itr, best_itr_patience, best_itr_threshold, best_confidence_type
def calculate_best_itr_dyn(best_itr, accuracy, prediction_time, best_subjects_accuracies_dyn, best_subjects_prediction_times_dyn, best_subjects_kappa_dyn, best_subjects_itrs_dyn, best_cm_dyn, subjects_accuracies_dyn, subjects_prediction_times_dyn, subjects_kappa_dyn, subjects_itrs_dyn, cm_dyn):
    number_classes = 4 # we have 4 classes in this dataset
    current_B = log2(number_classes) + accuracy*log2(accuracy)+(1-accuracy)*log2((1-accuracy)/(number_classes-1))

    prediction_time = prediction_time/250 #turning prediction time to seconds first
    current_Q =  60/prediction_time 

    current_itr = current_B * current_Q
    if current_itr > best_itr:
        best_subjects_accuracies_dyn = subjects_accuracies_dyn
        best_subjects_kappa_dyn  = subjects_kappa_dyn 
        best_subjects_prediction_times_dyn = subjects_prediction_times_dyn
        best_cm_dyn = cm_dyn
        best_itr = current_itr
        best_subjects_itrs_dyn = subjects_itrs_dyn
    return best_itr, best_subjects_accuracies_dyn, best_subjects_prediction_times_dyn, best_subjects_kappa_dyn, best_subjects_itrs_dyn, best_cm_dyn, current_itr

def plot_confusion_matrix(cm_stat, cm_dyn):
    plt.figure()  # Increase the size of the plot
    plt.title("Confusion Matrix: LDA - Dynamic - Expanding model", fontsize=12)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    s = sns.heatmap(cm_dyn, annot=True, fmt=".1f", cmap='magma', xticklabels=['Left hand', 'Right hand', 'Tongue', 'Feet'], yticklabels=['Left hand', 'Right hand', 'Tongue', 'Feet'])
    s.set(xlabel='Predicted Label', ylabel='True Label')
    plt.savefig(project_root + '/reports/figures/cumulative/LDA/dynamicVSstatic/Expanding_dynamic_ConfusionMatrix.png')

    # Plot confusion matrix for static model
    plt.figure() # Increase the size of the plot
    plt.title("Confusion Matrix: LDA - Static - Expanding model", fontsize=12)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    s = sns.heatmap(cm_stat, annot=True, fmt=".1f", cmap='magma', xticklabels=['Left hand', 'Right hand', 'Tongue', 'Feet'], yticklabels=['Left hand', 'Right hand', 'Tongue', 'Feet'])
    s.set(xlabel='Predicted Label', ylabel='True Label')
    plt.savefig(project_root + '/reports/figures/cumulative/LDA/dynamicVSstatic/Expanding_static_ConfusionMatrix.png')
#Expanding model - LDA
def main_lda_expanding():
    subjects = [1,2,3,4,5,6,7,8,9]  # 9 subjects
    sfreq = 250    # Sampling frequency - 250Hz
  
    best_params_expanding, best_itr_patience, best_confidence_type = tune_lda_expanding()
    print("\n\n Hyperparameter tuning (1): completed \n\n")
    
    csp_components = best_params_expanding['csp_components']
    initial_window_length = best_params_expanding['initial_window_length']
    expansion_rate = best_params_expanding['expansion_rate']
    solver = best_params_expanding['solver']
    n_components = best_params_expanding['n_components']
    tol = best_params_expanding['tol']
    shrinkage = best_params_expanding['shrinkage']

    w_start= np.arange(0, epochs_info(length= True) - initial_window_length, expansion_rate) 
    # Patience values = 1 - len(nr of windows)
    patience_values = np.arange(1, len(w_start)) #need to control this by itr 
    # Threshold values -> [0.001, 0.01, 0.1 - 0.9, 0.99, 0.999]
    threshold_values = np.array(np.arange(0.1, 1, 0.1)) #Have thresholds that are super close to 0 and super close 1, that might expand the plot
    threshold_values =np.concatenate(([0.001, 0.01], threshold_values))
    threshold_values = np.append(threshold_values, [0.99, 0.999])

    '''
    Evaluation
    '''
    #vary the treshold and find accuracy, kappa, and pred time for the dynamic model
    accuracy_dynamic = []
    accuracy_dynamic_total = []
    kappa_dynamic = []
    kappa_dynamic_total =  []
    prediction_time_dynamic = []
    prediction_time_dynamic_total = []
    itr_dynamic = []
    itr_dynamic_total = []
    best_subjects_accuracies_dyn = None
    best_subjects_prediction_times_dyn = None
    best_subjects_kappa_dyn = None
    best_cm_dyn = None
    best_subjects_itrs_dyn = None
    best_itr = 0
    #take the average subjects accuracies
    print("Evaluation (1): Finding dynamic model accuracy, kappa, and pred time, across thresholds")
    
    for n, threshold in enumerate(threshold_values):
        print(f"Threshold:{n+1}/{len(threshold_values)}")
        #dynamic model evaluation
        accuracy, kappa, prediction_time, itr, cm_dyn, subjects_accuracies_dyn, subjects_prediction_times_dyn, subjects_kappa_dyn, subjects_itrs_dyn, mean_scores_across_subjects, mean_kappa_across_subjects,mean_prediction_time_across_subjects, mean_itr_across_subjects = run_expanding_classification_dynamic(subjects, threshold, best_itr_patience, best_confidence_type, initial_window_length, expansion_rate, sfreq, csp_components, solver, shrinkage, n_components, tol)
        best_itr, best_subjects_accuracies_dyn, best_subjects_prediction_times_dyn, best_subjects_kappa_dyn, best_subjects_itrs_dyn, best_cm_dyn, _= calculate_best_itr_dyn(best_itr, accuracy, prediction_time, best_subjects_accuracies_dyn, best_subjects_prediction_times_dyn, best_subjects_kappa_dyn, best_subjects_itrs_dyn, best_cm_dyn, subjects_accuracies_dyn, subjects_prediction_times_dyn, subjects_kappa_dyn, subjects_itrs_dyn, cm_dyn)
        accuracy_dynamic.append(accuracy)
        kappa_dynamic.append(kappa)
        prediction_time_dynamic.append(prediction_time)
        accuracy_dynamic_total.append(mean_scores_across_subjects)
        kappa_dynamic_total.append(mean_kappa_across_subjects)
        prediction_time_dynamic_total.append(mean_prediction_time_across_subjects)
        itr_dynamic.append(itr)
        itr_dynamic_total.append(mean_itr_across_subjects)
    
    print("Evaluation (2):Finding static model accuracy, and kappa for the given prediction times from the dynamic model")
    #for each prediction time from the dynamic model, find the accuracy and kappa from the static model
    accuracy_static = []
    kappa_static= []
    subjects_accuracies_stat = []
    subjects_kappa_stat = []
    scores_across_subjects_stat = []
    kappa_across_subjects_stat = []
    accuracy = []
    kappa = []
    cm_stat = None

    subjects_accuracies_stat, scores_across_subjects_stat, subjects_kappa_stat, kappa_across_subjects_stat, subjects_itrs_stat, itrs_across_subjects_stat,  accuracy, kappa, itr, cm_stat, csp_stat= run_expanding_classification_static(subjects, initial_window_length, expansion_rate, csp_components, prediction_time_dynamic, solver, shrinkage)
    #accuracy and kappa are not single numbers here
    accuracy_static = accuracy
    kappa_static = kappa
    itr_static = itr

    #Turn pred times back to seconds
    tmin, tmax = epochs_info(tmin = True, tmax = True)

    prediction_time_dynamic_total = np.array(prediction_time_dynamic_total) /sfreq + tmin
    prediction_time_dynamic = np.array(prediction_time_dynamic) /sfreq + tmin
    best_subjects_prediction_times_dyn = np.array(best_subjects_prediction_times_dyn) / sfreq + tmin

    '''
    Plotting and storing data
    '''
    #Write the optimal parameters
    print(f"Chosen parameters: \n - csp: {csp_components}, \n - initial_window_length: {initial_window_length}\n - expansion-rate:  {expansion_rate} \n - confidence_type: {best_confidence_type}, \n - patience: {best_itr_patience} out of {patience_values}")
    h = open(project_root + "/reports/figures/cumulative/LDA/dynamicVSstatic/expanding_model_optimal_parameters.txt", "w")
    h.write(f"Chosen parameters: \n - csp: {csp_components}, \n - initial_window_length: {initial_window_length}\n - expansion-rate:  {expansion_rate} \n - confidence_type: {best_confidence_type}, \n - patience: {best_itr_patience} out of {patience_values}")
    h.close()

    #write per subject accuracies - dynamic
    subject_tuples = [(i+1, acc) for i, acc in enumerate(best_subjects_accuracies_dyn)]
    sorted_subjects = sorted(subject_tuples, key=lambda x: x[1], reverse=True)
    f = open(project_root + "/reports/figures/cumulative/LDA/dynamicVSstatic/expanding_dynamic_model_accuracy_by_subject.txt", "w")
    f.write(f"Classification accuracy - Dynamic Model: {np.mean(accuracy_dynamic)}\n")
    for subject, subject_accuracy in sorted_subjects:
        f.write(f"Subject {subject}: Accuracy: {subject_accuracy} \n")
        print(f"Subject {subject}: Accuracy: {subject_accuracy}")
    f.close()

    #write per subject accuracies - static
    subject_tuples = [(i+1, acc) for i, acc in enumerate(subjects_accuracies_stat)]
    sorted_subjects = sorted(subject_tuples, key=lambda x: x[1], reverse=True)

    f = open(project_root + "/reports/figures/cumulative/LDA/dynamicVSstatic/expanding_static_model_accuracy_by_subject.txt", "w")
    f.write(f"Classification accuracy - Static Model: {np.mean(accuracy_static)}\n")
    for subject, subject_accuracy in sorted_subjects:
        f.write(f"Subject {subject}: Accuracy: {subject_accuracy} \n")
        print(f"Subject {subject}: Accuracy: {subject_accuracy}")
    f.close()

    #write per subject kappa - dynamic
    subject_tuples = [(i+1, acc) for i, acc in enumerate(best_subjects_kappa_dyn)]
    sorted_subjects = sorted(subject_tuples, key=lambda x: x[1], reverse=True)
    f = open(project_root + "/reports/figures/cumulative/LDA/dynamicVSstatic/expanding_dynamic_model_kappa_by_subject.txt", "w")
    f.write(f"Average kappa - Dynamic Model: {np.mean(kappa_dynamic)}\n")
    for subject, subject_kappa in sorted_subjects:
        f.write(f"Subject {subject}: Kappa: {subject_kappa} \n")
        print(f"Subject {subject}: Kappa: {subject_kappa}")
    f.close()

    #write per subject kappa - static
    subject_tuples = [(i+1, acc) for i, acc in enumerate(subjects_kappa_stat)]
    sorted_subjects = sorted(subject_tuples, key=lambda x: x[1], reverse=True)

    f = open(project_root + "/reports/figures/cumulative/LDA/dynamicVSstatic/expanding_static_model_kappa_by_subject.txt", "w")
    f.write(f"Average kappa - Static Model: {np.mean(kappa_static)}\n")
    for subject, subject_kappa in sorted_subjects:
        f.write(f"Subject {subject}: Kappa: {subject_kappa} \n")
        print(f"Subject {subject}: Kappa: {subject_kappa}")
    f.close()

   #write per subject itr - dynamic
    subject_tuples = [(i+1, acc) for i, acc in enumerate(best_subjects_itrs_dyn)]
    sorted_subjects = sorted(subject_tuples, key=lambda x: x[1], reverse=True)
    f = open(project_root + "/reports/figures/cumulative/LDA/dynamicVSstatic/expanding_dynamic_model_itrs_by_subject.txt", "w")
    f.write(f"Average itr - Dynamic Model: {np.mean(itr_dynamic)}\n")
    for subject, subject_itr in sorted_subjects:
        f.write(f"Subject {subject}: ITR: {subject_itr} \n")
        print(f"Subject {subject}: ITR: {subject_itr}")
    f.close()

    #write per subject itr - static
    subject_tuples = [(i+1, acc) for i, acc in enumerate(subjects_itrs_stat)]
    sorted_subjects = sorted(subject_tuples, key=lambda x: x[1], reverse=True)

    f = open(project_root + "/reports/figures/cumulative/LDA/dynamicVSstatic/expanding_static_model_itrs_by_subject.txt", "w")
    f.write(f"Average itr - Static Model: {np.mean(itr_static)}\n")
    for subject, subject_itr in sorted_subjects:
        f.write(f"Subject {subject}: ITR: {subject_itr} \n")
        print(f"Subject {subject}: ITR: {subject_itr}")
    f.close()
    #write per subject prediction time - dynamic
    subject_tuples = [(i+1, acc) for i, acc in enumerate(best_subjects_prediction_times_dyn)]
    sorted_subjects = sorted(subject_tuples, key=lambda x: x[1], reverse=True)
    f = open(project_root + "/reports/figures/cumulative/LDA/dynamicVSstatic/expanding_dynamic_model_predtime_by_subject.txt", "w")
    f.write(f"Average prediction time - Dynamic Model: {np.mean(prediction_time_dynamic)}\n")
    for subject, subject_prediction_time in sorted_subjects:
        f.write(f"Subject {subject}: Prediction time: {subject_prediction_time} \n")
        print(f"Subject {subject}: Prediction time: {subject_prediction_time}")
    f.close()

    #plot accuracies
        # Calculate Standard Error of the Mean (SEM) based on individual accuracies
    sem_accuracy_dynamic = [np.std(scores) / np.sqrt(len(scores)) for scores in np.array(accuracy_dynamic_total)] #(accuracy_dyn_tot: shape(thresholds, test_epochs)) - accuracies over x threshold, 288 epochs, averaged across the subjects. Finding SEM for each threshold value
    sem_pred_time_dynamic = [np.std(times) / np.sqrt(len(times)) for times in np.array(prediction_time_dynamic_total)]
    sem_accuracy_stat =  [np.std(scores) / np.sqrt(len(scores)) for scores in np.array(scores_across_subjects_stat).T] #(scores_across_subjects_stat: (subjects, predtimes)) - 9 subjects, statically for each given prediction time from the dyn model(correlates to a specific threshold value)
    plt.figure(figsize=(8, 6)) 
    plt.style.use('ggplot')
    plt.xlabel('Prediction time (sec)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    tmin, tmax = epochs_info(tmin = True, tmax = True)
    onset = tmin
    offset = tmax
    plt.axvline(onset, linestyle="--", color="r", label="Onset")
    plt.axvline(offset, linestyle="--", color="b", label="Offset")
    plt.errorbar(prediction_time_dynamic, accuracy_dynamic, xerr = sem_pred_time_dynamic,yerr= sem_accuracy_dynamic, label = "Dynamic model", fmt='o', color='blue', ecolor='black', linestyle='-', linewidth=0.9, elinewidth=0.65, capsize=0.65)
    plt.errorbar(prediction_time_dynamic, accuracy_static, yerr= sem_accuracy_stat, label = "Static model", fmt='s', color='green', ecolor='black', linestyle='-', linewidth=0.9, elinewidth=0.65, capsize=0.65)
    plt.title("Expanding LDA models: Accuracy vs Pred time")
    plt.axhline(0.25, label= "Chance")
    plt.legend()
    plt.savefig(project_root + '/reports/figures/cumulative/LDA/dynamicVSstatic/ExpandingAccuracy.png')
    plt.show()

    #plot kappa
    sem_kappa_dynamic = [np.std(scores) / np.sqrt(len(scores)) for scores in np.array(kappa_dynamic_total)]
    sem_kappa_stat =  [np.std(scores) / np.sqrt(len(scores)) for scores in np.array(kappa_across_subjects_stat).T]
    plt.figure(figsize=(8, 6)) 
    plt.style.use('ggplot')
    plt.xlabel('Prediction time (sec)')
    plt.ylabel('Kappa')
    plt.grid(True)
    tmin, tmax = epochs_info(tmin = True, tmax = True)
    onset = tmin
    offset = tmax
    plt.axvline(onset, linestyle="--", color="r", label="Onset")
    plt.axvline(offset, linestyle="--", color="b", label="Offset")
    plt.errorbar(prediction_time_dynamic, kappa_dynamic, xerr = sem_pred_time_dynamic,yerr= sem_kappa_dynamic , label = "Dynamic model", fmt='o', color='blue', ecolor='black', linestyle='-', linewidth=0.9, elinewidth=0.65, capsize=0.65)
    plt.errorbar(prediction_time_dynamic, kappa_static, yerr= sem_kappa_stat, label = "Static model", fmt='s', color='green', ecolor='black', linestyle='-', linewidth=0.9, elinewidth=0.65, capsize=0.65)
    plt.title("Expanding LDA models: Kappa vs Pred time")
    plt.legend()
    plt.ylim(bottom=0)
    plt.savefig(project_root + '/reports/figures/cumulative/LDA/dynamicVSstatic/ExpandingKappa.png')
    plt.show()

    #plot itrs
    sem_itr_dynamic = [np.std(scores) / np.sqrt(len(scores)) for scores in np.array(itr_dynamic_total)]
    sem_itr_stat =  [np.std(scores) / np.sqrt(len(scores)) for scores in np.array(itrs_across_subjects_stat).T]
    plt.figure(figsize=(8, 6)) 
    plt.style.use('ggplot')
    plt.xlabel('Prediction time (sec)')
    plt.ylabel('Information transfer rate (bits/min)')
    plt.grid(True)
    tmin, tmax = epochs_info(tmin = True, tmax = True)
    onset = tmin
    offset = tmax
    plt.axvline(onset, linestyle="--", color="r", label="Onset")
    plt.axvline(offset, linestyle="--", color="b", label="Offset")
    plt.errorbar(prediction_time_dynamic, itr_dynamic, xerr = sem_pred_time_dynamic,yerr= sem_itr_dynamic , label = "Dynamic model", fmt='o', color='blue', ecolor='black', linestyle='-', linewidth=0.9, elinewidth=0.65, capsize=0.65)
    plt.errorbar(prediction_time_dynamic, itr_static, yerr= sem_itr_stat, label = "Static model", fmt='s', color='green', ecolor='black', linestyle='-', linewidth=0.9, elinewidth=0.65, capsize=0.65)
    plt.title("Expanding LDA models: Information Transfer rate vs Pred time")
    plt.legend()
    plt.ylim(bottom=0)
    plt.savefig(project_root + '/reports/figures/cumulative/LDA/dynamicVSstatic/ExpandingItr.png')
    plt.show()

    plot_confusion_matrix(cm_stat, best_cm_dyn)
    #for comparison with the sliding model to decide and choose the one that performs better
    itr_dyn = np.mean(itr_dynamic)
    itr_stat = np.mean(itr_static)
    return itr_dyn, itr_stat


if __name__ == "__main__":
    main_lda_expanding()
    