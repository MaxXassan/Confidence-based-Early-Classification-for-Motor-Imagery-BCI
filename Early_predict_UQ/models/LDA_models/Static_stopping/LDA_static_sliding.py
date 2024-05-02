import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
from mne.decoding import CSP

from sklearn.model_selection import ParameterSampler
import numpy as np


current_directory = os.path.abspath('')

project_root = current_directory

sys.path.append(project_root)

print("ROOT:", project_root)
from Early_predict_UQ.data.make_dataset import make_data

#Sldiding window - classification - tuning using kfold cross validation
   ##calculate kappa and accuracy at each window step
def run_sliding_classification_tuning(subjects, w_length, w_step, csp_components):
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
        
        #get the training set - first session of the data
        train_indexes = [i for i, epoch in enumerate(epochs) if epochs[i].metadata['session'].iloc[0] == '0train']
        epochs_data = epochs.get_data(copy = False)
        train_data = epochs_data[train_indexes]
        train_labels = labels[train_indexes]


        cv = KFold(n_splits=3, shuffle = True, random_state=42)
        scores_cv_splits = []
        kappa_cv_splits  = []

        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components= csp_components, reg=None, log=True, norm_trace=False)

        current_cv = 0 
        for train_idx, test_idx in cv.split( train_data):
            current_cv += 1
            y_train, y_test = train_labels[train_idx], train_labels[test_idx]
            X_train = csp.fit_transform(train_data[train_idx], y_train)
            lda.fit(X_train, y_train)

            w_start = np.arange(0, train_data.shape[2] - w_length, w_step) 
            scores_across_epochs = []
            kappa_across_epochs = []
            for n in w_start:
                X_test_window = csp.transform(train_data[test_idx][:, :, n:(n + w_length)])
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
 
def run_sliding_classification(subjects, w_length, w_step, csp_components):
    scores_across_subjects = []
    kappa_across_subjects = []

    subjects_accuracies =[]
    subjects_kappa = []
    current_person = 0

    #confusion matrix
    number_cm = 0 
    cm = np.zeros((4,4))
    for person in subjects:
        current_person += 1
        print("Person %d" % (person))
        subject= [person]
        epochs, labels = make_data(subject)
        epochs_data = epochs.get_data(copy = False)


        ##change labels!!  maybe

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


        # Fit CSP on training data
        X_train = csp.fit_transform(train_data, train_labels)

        # Fit LDA on training data
        lda.fit(X_train, train_labels)

        # Initialize lists to store scores and kappa values for each window
        scores_across_epochs = []
        kappa_across_epochs = []

        w_start = np.arange(0, test_data.shape[2] - w_length, w_step) 
        #windows in the trial
        for n in w_start:
            X_test_window = csp.transform(train_data[:, :, n:(n + w_length)])
            #accuracy
            score = lda.score(X_test_window, test_labels)
            scores_across_epochs.append(score)

            #kappa
            kappa = cohen_kappa_score(lda.predict(X_test_window), test_labels) 
            kappa_across_epochs.append(kappa)

            #Confusion matrix
            predictions = lda.predict(X_test_window)
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

    cm = np.divide(cm, number_cm)
    print("confusion matrix")
    print(cm)
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

    #max intial_length and epxansion rate to be 1/4 of the trial during the MI task, or 1 seconds
    w_length_values = np.round(rng.uniform(0.1, 2, 10), 2)

    #max step is the length of the window. 
    w_step_values = []
    for w_length in w_length_values:
        max_step = max(0.1, 0.99 * w_length)
        w_step = np.round(rng.uniform(0.1, max_step), 1) 
        w_step_values.append(w_step)  
        
    w_length_values = np.round(np.array(w_length_values) * sfreq).astype(int)
    w_step_values = np.round(np.array(w_step_values) * sfreq).astype(int)
    
    parameters = {  
        'csp_components': [4, 8, 12], 
        'w_length': w_length_values, 
        'w_step': w_step_values
    }


    parameters_list = list(ParameterSampler(parameters, n_iter=3, random_state=rng))
    return parameters_list



def hyperparameter_tuning (parameters_list, subjects):
    mean_accuracy = 0
    best_accuracy = 0
    for n, param_set in enumerate(parameters_list):
        csp_components = param_set['csp_components']
        w_length = param_set['w_length'] 
        w_step =  param_set['w_step']
        _, _, _, _, accuracy, _ = run_sliding_classification_tuning(subjects, w_length, w_step, csp_components)
        
        mean_accuracy = np.mean(accuracy)

        print(f"Iteration {n+1}/{len(parameters_list)}: Mean accuracy for parameters {param_set} is {mean_accuracy}")

        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_params = param_set

    return best_params, best_accuracy

def evaluate_and_plot(best_params, best_accuracy, accuracy, subjects_accuracies, scores_across_subjects, kappa, subjects_kappa, kappa_across_subjects, sfreq, cm, csp):
    print("Best params:", best_params)
    print("Best accuracy", best_accuracy)
    print("subjects_kappa", subjects_kappa)
    print("subjects_accuracy", subjects_accuracies)
    h = open(project_root + "/reports/figures/cumulitive/LDA/static/sliding/lda_static_sliding_best_params.txt", "w")
    h.write(f"Best params: {best_params}\n")
    h.close()
    print("Classification accuracy:", np.mean(accuracy))
    print("Kappa:", np.mean(kappa))

    labels, tmin = epochs_info(labels=True, tmin=True)

    #Mean accuracy for each subject
    subject_tuples = [(i+1, acc) for i, acc in enumerate(subjects_accuracies)]
    sorted_subjects = sorted(subject_tuples, key=lambda x: x[1], reverse=True)

    f = open(project_root + "/reports/figures/cumulitive/LDA/static/sliding/lda_static_sliding_accuracy_by_subject.txt", "w")
    f.write(f"Classification accuracy: {np.mean(accuracy)}\n")
    for subject, subject_accuracy in sorted_subjects:
        f.write(f"Subject {subject}: Accuracy {subject_accuracy} \n")
        print(f"Subject {subject}: Accuracy {subject_accuracy}")
    f.close()

    #Mean kappa for each subject
    g = open(project_root + "/reports/figures/cumulitive/LDA/static/sliding/lda_static_sliding_kappa_by_subject.txt", "w")
    g.write(f"Kappa: {np.mean(kappa)}\n")
    subject_tuples = [(i+1, acc) for i, acc in enumerate(subjects_kappa)]
    sorted_subjects = sorted(subject_tuples, key=lambda x: x[1], reverse=True)
    for subject, subject_kappa in sorted_subjects:
        g.write(f"Subject {subject}: Kappa {subject_kappa} \n")
        print(f"Subject {subject}: Kappa {subject_kappa}")
    g.close()

    classes = ['left_hand', 'right_hand', 'tongue', 'feet']
    class_balance = np.zeros(4)
    for i, class_type in enumerate(classes):
        class_balance[i] = np.mean(labels == class_type)
    class_balance = np.max(class_balance)
    print("class balance",  class_balance)

    prediction_times = np.arange(0, epochs_data.shape[2] - best_params['w_length'], best_params['w_step'])
    prediction_times  = (prediction_times + (best_params['w_length'] / tmin)) / sfreq + tmin


    ## csp patterns
    info = epochs_info(info = True)
    csp.plot_patterns(info, ch_type="eeg", units="Patterns (AU)", size=1.5)
    plt.savefig(project_root + '/reports/figures/cumulitive/LDA/static/sliding/lda_static_sliding_cspPatterns.png')
    plt.show()

    ##confusion  matrix
    displaycm = ConfusionMatrixDisplay(cm, display_labels = classes)
    displaycm.plot()
    plt.title('Confusion Matrix: LDA - Static model - Expanding window')
    plt.grid(False)
    plt.savefig(project_root + '/reports/figures/cumulitive/LDA/static/sliding/lda_static_sliding_confusionMatrix.png')
    plt.show()


    #Accuracy for each subject
    plt.xlabel("Prediction time")
    plt.ylabel("Classification accuracy")

    for i in range(len(scores_across_subjects)):
        plt.plot(prediction_times, scores_across_subjects[i], label= f"Subject: {i+1}")

    plt.axhline(class_balance, linestyle="-", color="k", label="Chance")
    plt.axvline(tmin, linestyle="--", color="k", label="Onset")
    plt.title('Accuracy over time for each subject: LDA - Static model - Sliding window')
    plt.legend()
    plt.grid(True)
    plt.savefig(project_root + '/reports/figures/cumulitive/LDA/static/sliding/lda_static_sliding_accuracy_over_time_subjects.png')
    plt.show()  

    #Kappa for each subject
    plt.xlabel("Prediction time")
    plt.ylabel("Kappa")

    for i in range(len(kappa_across_subjects)):
        plt.plot(prediction_times, kappa_across_subjects[i], label= f"Subject: {i+1}")
    #whats a bad line?
    plt.axvline(tmin, linestyle="--", color="k", label="Onset")
    plt.title('Kappa over time for each subject: LDA - Static model - Sliding window')
    plt.legend()
    plt.grid(True)
    plt.savefig(project_root + '/reports/figures/cumulitive/LDA/static/sliding/lda_static_sliding_accuracy_over_time_subjects.png')
    plt.show()

    #Accuracy
    accuracy_array = np.array(accuracy)
    plt.xlabel("Prediction time")
    plt.ylabel("Classification accuracy")
    plt.plot(prediction_times, accuracy_array)
    plt.axhline(class_balance, linestyle="-", color="k", label="Chance")
    plt.axvline(tmin, linestyle="--", color="k", label="Onset")
    plt.title('Accuracy over time: LDA - Static model - Sliding window')
    plt.legend()
    plt.grid(True)
    plt.savefig(project_root + '/reports/figures/cumulitive/LDA/static/sliding/lda_static_sliding_accuracy_over_time.png')
    plt.show()

    #Kappa
    kappa_array = np.array(kappa)
    plt.xlabel("Prediction time")
    plt.ylabel("Kappa")
    #whats a bad line?
    plt.plot(prediction_times, kappa_array)
    plt.axvline(tmin, linestyle="--", color="k", label="Onset")
    plt.title('Kappa over time: LDA - Static model - Sliding window')
    plt.legend()
    plt.grid(True)
    plt.savefig(project_root + '/reports/figures/cumulitive/LDA/static/sliding/lda_static_sliding_kappa_over_time.png')
    plt.show()

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

'''
    Main function: static, sliding, lda prediction at each time point(window step) of the csp transformed eeg signal (depending on the optimum window configuration) - LDA + CSP
        - Tune hyperparameters using randomsearch
        - Run with the optimal parameters
        - Evaluate and plot the result

'''
def main_static_sliding():
    subjects = [1, 3]   # 9 subjects
    sfreq = 250     # Sampling frequency - 250Hz

    #Hyperparameter_tuning
    print("\n\n Hyperparameter tuning: \n\n")
    parameters_list = create_parameterslist(sfreq)

    best_params, best_accuracy = hyperparameter_tuning(parameters_list, subjects)
    
    #classify
    print("\n\n Classification: \n\n")
    subjects_accuracies, scores_across_subjects, subjects_kappa, kappa_across_subjects, accuracy, kappa, cm, csp = run_sliding_classification(subjects, best_params['w_length'], best_params['w_step'], best_params['csp_components'])

    #evaluate and plot
    evaluate_and_plot(best_params, best_accuracy, accuracy, subjects_accuracies, scores_across_subjects, kappa, subjects_kappa, kappa_across_subjects, sfreq, cm, csp)

    return best_params

if __name__ == "__main__":
    main_static_sliding()