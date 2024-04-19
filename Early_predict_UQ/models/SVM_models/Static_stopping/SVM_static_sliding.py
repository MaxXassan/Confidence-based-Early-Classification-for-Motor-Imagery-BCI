import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from mne.decoding import CSP

from sklearn.model_selection import ParameterSampler
import numpy as np


current_directory = os.path.abspath('')

project_root = current_directory

sys.path.append(project_root)

print("ROOT:", project_root)
from Early_predict_UQ.data.make_dataset import make_data


# epoch tmin  = 2 and tmax = 6 , as the motor imagery task lasted in that time
def run_sliding_classification(subjects, w_length, w_step, csp_components, c, kernel, gamma = None):
    scores_across_subjects = []
    current_person = 0
    subjects_accuracies = []
    for person in subjects:
        current_person += 1
        print("Person %d" % (person))
        subject= [person]
        epochs, labels = make_data(subject)
        labels = epochs.events[:, -1] - 4
        epochs_data = epochs.get_data(copy=False)

        cv = ShuffleSplit(n_splits=10, test_size = 0.2, random_state=42)
        scores_cv_splits = []
        if gamma == None:
            svm = SVC(C = c, kernel = kernel, probability = True)
        else:
            svm = SVC(C = c, gamma = gamma,kernel = kernel, probability = True)
        csp = CSP(n_components= csp_components, reg=None, log=True, norm_trace=False)
        current_cv = 0 
        for train_idx, test_idx in cv.split(epochs_data):
            current_cv += 1
            y_train, y_test = labels[train_idx], labels[test_idx]
            X_train = csp.fit_transform(epochs_data[train_idx], y_train)
            svm.fit(X_train, y_train)

            w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step) 
            scores_across_epochs = []
            for n in w_start:
                X_test_window = csp.transform(epochs_data[test_idx][:, :, n:(n + w_length)])
                score = svm.score(X_test_window, y_test)
                scores_across_epochs.append(score)
            if current_cv == 1:
                scores_cv_splits = np.array(scores_across_epochs)
            else:
                scores_cv_splits = np.vstack((scores_cv_splits,np.array(scores_across_epochs)))

        mean_scores_across_cv = np.mean(scores_cv_splits, axis=0)
        if current_person == 1:
            scores_across_subjects  = np.array(mean_scores_across_cv)
        else:
            scores_across_subjects = np.vstack((scores_across_subjects,np.array(mean_scores_across_cv)))
        subjects_accuracies.append(np.mean(mean_scores_across_cv))
    mean_scores_across_subjects = np.mean(scores_across_subjects, axis=0)
    accuracy = mean_scores_across_subjects

    return subjects_accuracies, accuracy
 
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
        'csp_components': [2, 4, 6, 8, 10], 
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
            _ , accuracy = run_sliding_classification(subjects, w_length, w_step, csp_components, c, kernel, gamma)
        else:
            _ , accuracy = run_sliding_classification(subjects, w_length, w_step, csp_components, c, kernel, gamma = None)
        
        mean_accuracy = np.mean(accuracy)

        print(f"Iteration {n+1}/{len(parameters_list)}: Mean accuracy for parameters {param_set} is {mean_accuracy}")

        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_params = param_set

    return best_params, best_accuracy

def evaluate_and_plot(best_params, best_accuracy, accuracy,subjects_accuracies):
    print("Best params:", best_params)
    print("Best accuracy", best_accuracy)
    print("Classification accuracy:", np.mean(accuracy))
    accuracy_array = np.array(accuracy)
    subject_tuples = [(i+1, acc) for i, acc in enumerate(subjects_accuracies)]

    sorted_subjects = sorted(subject_tuples, key=lambda x: x[1], reverse=True)

    for subject, subject_accuracy in sorted_subjects:
        print(f"Subject {subject}: Accuracy {subject_accuracy}")

    labels, tmin = epochs_info(labels=True, tmin=True)

    class_balance = np.zeros(4)
    for i in range(4):
        class_balance[i] = np.mean(labels == i)
    class_balance = np.max(class_balance)

    plt.title("Accuracy over time")
    plt.xlabel("Prediction time")
    plt.ylabel("Classification accuracy")

    prediction_times = np.arange(0, epochs_data.shape[2] - best_params['w_length'], best_params['w_step'])
    prediction_times  = (prediction_times + (best_params['w_length'] / tmin)) / sfreq + tmin

    plt.plot(prediction_times, accuracy_array)
    plt.axhline(class_balance, linestyle="-", color="k", label="Chance")
    plt.axvline(tmin, linestyle="--", color="k", label="Onset")
    plt.title('Accuracy over time: SVM - Static model - Sliding window')
    plt.legend()
    plt.grid(True)
    plt.savefig(project_root + '/reports/figures/cumulitive/SVM/static/svm_static_sliding_accuracy_over_time.png')
    plt.show()


def epochs_info(labels=False, tmin=False, length=False):
    global epochs_data
    global labels_data
    if labels or tmin or length:
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
    else:
        raise ValueError("At least one of 'labels', 'tmin', or 'length' must be True.")


if __name__ == "__main__":
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]   # 9 subjects
    sfreq = 250      
    #Hyperparameter_tuning
    print("\n\n Hyperparameter tuning: \n\n")
    parameters_list = create_parameterslist(sfreq)

    best_params, best_accuracy = hyperparameter_tuning(parameters_list, subjects)
    
    #classify
    print("\n\n Classification: \n\n")
    if best_params['kernel'] == 'rbf':
        subjects_accuracies, accuracy = run_sliding_classification(subjects, best_params['w_length'], best_params['w_step'], best_params['csp_components'], best_params['C'], best_params['kernel'], best_params['gamma'])
    else:
        subjects_accuracies, accuracy = run_sliding_classification(subjects, best_params['w_length'], best_params['w_step'], best_params['csp_components'], best_params['C'], best_params['kernel'], gamma = None)
    accuracy_array = np.array(accuracy)

    #evaluate and plot
    evaluate_and_plot(best_params, best_accuracy, accuracy, subjects_accuracies)
