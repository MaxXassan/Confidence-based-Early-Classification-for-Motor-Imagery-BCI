import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

from scipy.stats import entropy

from mne.decoding import CSP

current_directory = os.path.abspath('')

project_root = os.path.abspath(os.path.join(current_directory, '..'))

sys.path.append(project_root)

print("ROOT:", project_root)
from Early_predict_UQ.data.make_dataset import make_data
from Early_predict_UQ.data.plots import plot_accuracy_over_time_and_epochs, plot_confidence_over_time_and_epochs #, plot_cost_over_time_and_epochs


# predicting with an expanding window
def predict_expanding(initial_window_length, w_start, w_step, y_test, epochs_data, test_idx, probs_this_window, chosen_epoch=None):
    score_this_window = []
    entropy_this_window = []
    confidence_this_window = []
    costs = []
    numTimesBelowThreshold = 0
    numberOfNs = 0
    predict_time = 0
    predict = False
    for n, window_start in enumerate(w_start):
        window_length = initial_window_length + n * w_step
        X_test = csp.transform(epochs_data[test_idx][:, :, window_start:(window_start + window_length)])
        
        if chosen_epoch is not None:
            X_test = X_test[chosen_epoch]
        
        score = lda.score(X_test, y_test)
        score_this_window.append(score)
        
        probabilities = lda.predict_proba(X_test)
        
        if len(probs_this_window) == 0:
            probs_this_window = probabilities
        else:
            probs_this_window = np.vstack((probs_this_window, probabilities))

        #predictive entropy - H_pred(p) 
        entropy_score = entropy(probabilities, axis=1)
        entropy_this_window.append(entropy_score)

        '''
        Confidence - as seen in: 
        Uncertainty Quantification in Machine Learning for Biosignal Applications - A Review, page 13.
        1 - H_pred(p) can be used as a confidence measure. Normalizing seems useful - 1 / (1- entropy-score)
        '''
        confidence = 1 - entropy_score
        confidence_this_window.append(confidence)
        probabilities = np.array(probabilities)

        #Early prediction
        probabilities = probabilities.flatten()
        sorted_probs = sorted(probabilities, reverse=True)
        print("sorted probabilities: \n", sorted_probs)
        #cost1 = 1/(1+(sorted_probs[0] - sorted_probs[1]))
        cost =1/(1+(sorted_probs[0] + (sorted_probs[0] - sorted_probs[1])))
        costs.append(cost)
        if cost < 0.5 and predict == False:
            numTimesBelowThreshold +=1
            if numTimesBelowThreshold == 2:
                predict = True
                predict_time = numberOfNs
                print("BELOW")
        numberOfNs+=1

    return score_this_window, probs_this_window, entropy_this_window, confidence_this_window, predict_time, costs


# predicting with a sliding window
def predict_sliding(w_length, w_start, w_step, y_test, epochs_data, test_idx, probs_this_window, chosen_epoch=None):
    score_this_window = []
    entropy_this_window = []
    confidence_this_window = []
    
    for n in w_start:
        X_test = csp.transform(epochs_data[test_idx][:, :, n:(n + w_length)])

        if chosen_epoch is not None:
            X_test = X_test[chosen_epoch]
        
        score = lda.score(X_test, y_test)
        score_this_window.append(score)
        
        probabilities = lda.predict_proba(X_test)
        
        if len(probs_this_window) == 0:
            probs_this_window = probabilities
        else:
            probs_this_window = np.vstack((probs_this_window, probabilities))

        entropy_score = entropy(probabilities, axis=1)
        entropy_this_window.append(entropy_score)

        confidence = 1 - entropy_score
        confidence_this_window.append(confidence)

    return score_this_window, probs_this_window, entropy_this_window, confidence_this_window

# Setting parameters
threshold = 0.7
subject_list = [1]


## (Might need to do cross session - session 1 as train, and session 2 as test. See dataset_structure.ipynb)
# load preprocessed epochs
epochs, labels = make_data(subject_list)

# Asserting the epochs and labels (last row of the events matrix) to be used for the classification
epochs_train = epochs.copy()
labels = epochs.events[:, -1] - 4

# Cross-validation 
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

# LDA and CSP pipeline
lda = LinearDiscriminantAnalysis()
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

# Class balance between the 4 classes
class_balance = np.max([np.mean(labels == i) for i in range(4)])

sfreq = 250  # Sampling frequency of 250 Hz, as per the BCI competion dataset 2a

# Classify the signal using a sliding window or expanding window
w_length = int(sfreq * 0.5)  # Window length/initial window length
w_step = int(sfreq * 0.1)  # Window step size
w_start = np.arange(0, epochs.get_data().shape[2] - w_length, w_step)
w_times = (w_start + w_length / 2.0) / sfreq + epochs.tmin

class_names = {
    1: "Left hand",
    2: "Right hand",
    3: "Both feet",
    4: "Tongue"
}

traversal_types = ['expanding', 'sliding', 'whole_data']
traversal_types_scores = []

# Loop through different traversal types
for traversal_type in traversal_types:
    probabilitites_windows = []
    scores_windows = []  
    confidence_windows = []
    entropy_windows = []
    predict_times = []
    
    # Handle the case of using the whole data
    if traversal_type == 'whole_data':
        print("WHOLE DATA METHOD")
        clf = Pipeline([("CSP", csp), ("LDA", lda)])
        score_this_window = cross_val_score(clf, epochs_train.get_data(), labels, cv=cv, n_jobs=None)
        print("Score_this_window whole, mean:\n",  np.mean(score_this_window))
    
    else:
        # Split data for cross-validation
        cv_split = cv.split(epochs_train.get_data())
        
        # Loop through cross-validation splits
        for train_idx, test_idx in cv_split:
            print("\n\n\nCV FOR:", traversal_type)
            y_train, y_test = labels[train_idx], labels[test_idx]
            
            # Transform data using CSP
            X_train = csp.fit_transform(epochs_train.get_data()[train_idx], y_train)
            X_test = csp.transform(epochs_train.get_data()[test_idx])
            
            # Fit classifier
            lda.fit(X_train, y_train)
            
            probs_this_window = []
            score_this_window = []
            confidence_this_window =  []
            
            # Predict based on traversal type
            if traversal_type == 'sliding':
                print("SLIDING METHOD")
                score_this_window, probs_this_window, entropy_this_window, confidence_this_window = predict_sliding(
                    w_length, 
                    w_start,
                    w_step,
                    y_test,
                    epochs.get_data(),
                    test_idx,
                    probs_this_window,
                    chosen_epoch=None
                )
                print("Score_this_window sliding, mean:\n", np.mean(score_this_window))
            #expanding window - hyperparamers - intial window lenght, expansion rate
            elif traversal_type == 'expanding':
                print("EXPANDING METHOD")
                score_this_window, probs_this_window, entropy_this_window, confidence_this_window, predict_time, costs = predict_expanding(
                    w_length, 
                    w_start,
                    w_step,
                    y_test,
                    epochs.get_data(),
                    test_idx,
                    probs_this_window,
                    chosen_epoch=None
                )
                print("Score_this_window expanding, mean:\n", np.mean(score_this_window))
            
            scores_windows.append(score_this_window)
            probabilitites_windows.append(probs_this_window)
            confidence_windows.append(np.mean(confidence_this_window, 1))
            entropy_windows.append(entropy_this_window)
            predict_times.append(predict_time)
            
            traversal_types_scores.append(np.mean(scores_windows))

        #print("Scores:\n", scores_windows)
        print("Scores.shape:\n", np.array(scores_windows).shape)
        print("w times:\n", np.array(w_times).shape)
        print("\n\n\nEND OF:", traversal_type)
        #cant plot in codespace
        #plot_cost_over_time_and_epochs(w_times, costs, predict_time)
        plot_accuracy_over_time_and_epochs(w_times, scores_windows, int(np.mean(predict_times)), class_balance)
        plot_confidence_over_time_and_epochs(w_times, confidence_windows, int(np.mean(predict_times)), threshold)

print(traversal_types_scores)

for i in range(len(traversal_types)):
    print(
        "Classification accuracy of %s: %f / Chance level: %f" % (traversal_types[i], traversal_types_scores[i], class_balance)
    )

''' In fix_csp_lda_current
To do: 
-  see if classification accuracy works and provides the right numbers
-  Plot costs with the average predict time (make it work for sliding and whole data too)
- start the hyperparameter tuning to maximize classification accuracy, and minimize predict_time 
- make it take into account all the subjects
- make it work using svm
- provide the plots for all the subjects for all subjects for each condition, let it just save the plots to a folder automatically
'''