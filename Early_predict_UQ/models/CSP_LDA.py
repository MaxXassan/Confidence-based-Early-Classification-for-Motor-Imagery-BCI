import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit
from scipy.stats import entropy

from mne.decoding import CSP

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

print("ROOT:", project_root)
from data.make_dataset import make_data
from data.plots import plot_accuracy_over_time_and_epochs, plot_confidence_over_time_and_epochs

#from feauterimport plot_entropy_over_time

#add for when only 1 epoch

def predict_expanding(initial_window_length, w_start, w_step, y_test, epochs_data, test_idx, probs_this_window, chosen_epoch = None):
    score_this_window = []
    #preds_this_window = []
    entropy_this_window = []
    confidence_this_window = []
    print("w_Start: ", len(w_start))
    for n, window_start in enumerate(w_start):
        window_length = initial_window_length + n * w_step
        X_test  = csp.transform(epochs_data[test_idx][:, :, window_start:(window_start + window_length)])
        #print("X_test  shape:\n",X_test.shape)
        if chosen_epoch != None:
            X_test = X_test [chosen_epoch] #Chooosing a specific epoch in the test set 
        #Accuracy
        score = lda.score(X_test, y_test)
        score_this_window.append(score)
        
        probabilities = lda.predict_proba(X_test)
        
        if len(probs_this_window) == 0:
            probs_this_window = probabilities
        else:
            probs_this_window = np.vstack((probs_this_window, probabilities))

       # print("Prediction for this time window: ", prediction)
        #print("prob shape: ", probabilities.shape)
        #print("probabilities: \n", probabilities)

        #predictive entropy - H_pred(p) 
        entropy_score = entropy(probabilities, axis = 1) #- see if entropy is better than probabilites
        entropy_this_window.append(entropy_score)

        '''
        Confidence - as seen in: 
        Uncertainty Quantification in Machine Learning for Biosignal Applications - A Review, page 13.
        1 - H_pred(p) can be used as a confidence measure. Normalizing seems useful - 1 / (1- entropy-score)
        '''
        #confidence
        confidence = 1 - entropy_score
        confidence_this_window.append(confidence)

    return score_this_window, probs_this_window, entropy_this_window, confidence_this_window


#def predict_sliding():
    
threshold = 0.7
subject_list = [1]

# Preprocessed epochs
epochs, labels = make_data(subject_list)


# Asserting the epochs and labels (last row of the events matrix) to be used for the classification
epochs_train = epochs.copy()
labels = epochs.events[:, -1] - 4

# Cross validation 
## (Might need to do cross session - session 1 as train, and session 2 as test. See dataset_structure.ipynb)
scores = []
epochs_data = epochs.get_data(copy=False)
epochs_data_train = epochs_train.get_data(copy=False)
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
cv_split = cv.split(epochs_data_train)

# LDA and CSP pipeline
lda = LinearDiscriminantAnalysis()
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

# Class balance between the 4 classes. 
class_balance = np.zeros(4)
for i in range(4):
    class_balance[i] = np.mean(labels == i)

class_balance = np.max(class_balance)

sfreq = 250 # Sampling frequency of 250 Hz as per the BCI competion dataset 2a

# Classify the signal using a sliding window or expanding window
w_length = int(sfreq * 0.5)  # Window length/initial window length
w_step = int(sfreq * 0.1)  # window step size
w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step) #
print("w start shape: ", w_start.shape)

class_names = {
        1: "Left hand",
        2: "Right hand",
        3: "Both feet",
        4: "Tongue"
}

probabilitites_windows = []
scores_windows = []  
confidence_windows = []
entropy_windows = []
traversal_type = 'expanding'
# Running classification across the signal
for train_idx, test_idx in cv_split:
    y_train, y_test = labels[train_idx], labels[test_idx] # Get the current labels and data
    
    # Exatract spatial filters and transform the data as a whole
    X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
    X_test = csp.transform(epochs_data_train[test_idx])

    # Fit the classifier on the training data
    lda.fit(X_train, y_train)
    w_times = (w_start + w_length / 2.0) / sfreq + epochs.tmin

    # Test the classifier on the windows. This is where we run over the signal
    probs_this_window = []
    #chosen_epoch = 80
    c = 0
    if traversal_type == 'sliding':
        #predict_sliding()
        c= 1
    elif traversal_type == 'expanding':
        score_this_window, probs_this_window, entropy_this_window, confidence_this_window = predict_expanding(
            w_length, 
            w_start,
            w_step,
            y_test,
            epochs_data,
            test_idx,
            probs_this_window,
            chosen_epoch = None
        )
    scores_windows.append(score_this_window)
    probabilitites_windows.append(probs_this_window)
    confidence_windows.append(confidence_this_window)
    entropy_windows.append(entropy_this_window)

# Plot the scores over time
print("accuracy_all shape: ", scores_windows.shape)
print("confindence_all shape: ", confidence_windows.shape)
plot_accuracy_over_time_and_epochs(w_times, scores_windows, class_balance)
plot_confidence_over_time_and_epochs(w_times, confidence_windows, threshold)