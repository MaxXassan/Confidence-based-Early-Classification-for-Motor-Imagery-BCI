import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit

from mne.decoding import CSP

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from data.make_dataset import make_data

def plot_over_time(w_times, scores_windows, class_balance, onset):
    plt.figure()
    plt.plot(w_times, np.mean(scores_windows, 0), label="Score")
    plt.axvline(onset, linestyle="--", color="k", label="Onset")
    plt.axhline(class_balance, linestyle="-", color="k", label="Chance")
    plt.xlabel("time (s)")
    plt.ylabel("classification accuracy")
    plt.title("Classification score over time")
    plt.legend(loc="lower right")
    plt.show()

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

# Classify the signal using a sliding window

w_length = int(sfreq * 0.5)  # Window length
w_step = int(sfreq * 0.1)  # window step size
w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step) # Set of starting positions in the signal(Note! the signal is 2s to 4s)

scores_windows = []  

# Running classification across the signal
for train_idx, test_idx in cv_split:
    y_train, y_test = labels[train_idx], labels[test_idx] # Get the current labels and data
    
    # Exatract spatial filters and transform the data as a whole
    X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
    X_test = csp.transform(epochs_data_train[test_idx])

    # Fit the classifier on the training data
    lda.fit(X_train, y_train)

    # Test the classifier on the windows. This is where we run over the signal
    score_this_window = []
    for n in w_start:
        X_test = csp.transform(epochs_data[test_idx][:, :, n : (n + w_length)])
        score_this_window.append(lda.score(X_test, y_test))
    scores_windows.append(score_this_window)
    
# Plot the scores over time
w_times = (w_start + w_length / 2.0) / sfreq + epochs.tmin
plot_over_time(w_times, scores_windows, class_balance, epochs.tmin)