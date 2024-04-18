
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

project_root = current_directory

sys.path.append(project_root)

print("ROOT:", project_root)
from Early_predict_UQ.data.make_dataset import make_data
from Early_predict_UQ.data.plots import plot_accuracy_over_time_and_epochs, plot_confidence_over_time_and_epochs #, plot_cost_over_time_and_epochs

def calculate_entropy(probabilities):
    entropy = -np.sum(probabilities * np.log(probabilities))
    return entropy
def norm_entropy(pk, base ):
    return 1
def find_mean_probabilities(probabilities):
   return np.mean(probabilities, axis=0)

def calculate_max_mean_probability_and_class(probabilities):
    max_mean_probability_index = np.argmax(np.mean(probabilities, axis=0))
    max_mean_probability = np.max(np.mean(probabilities, axis=0))
    corresponding_class = max_mean_probability_index + 1  # Classes are typically indexed starting from 1
    return max_mean_probability, corresponding_class

def most_frequent(List):
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
 
    return num

threshold = 0.7

subject_list = [1] # Choosing the subject or subjects
# Preprocessed epochs
epochs, labels = make_data(subject_list)
# Asserting the epochs and labels (last row of the events matrix) to be used for the classification
epochs_train = epochs.copy()
labels = epochs.events[:, -1] - 4

# Cross validation 
## (Might need to do cross session - session 1 as train, and session 2 as test. See dataset_structure.ipynb)
scores = []
epochs_data = epochs.get_data(copy=False)
print("shape of epochs data:\n",epochs_data.shape)
epochs_data_train = epochs_train.get_data(copy=False)
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
cv_split = cv.split(epochs_data_train)
print("cv split: \n", cv_split)

# LDA and CSP pipeline
lda = LinearDiscriminantAnalysis()
csp = CSP(n_components=5, reg=None, log=True, norm_trace=False) # why 4 components or 5

# Class balance between the 4 classes. 
class_balance = np.zeros(4)
for i in range(4):
    class_balance[i] = np.mean(labels == i)

class_balance = np.max(class_balance)

sfreq = 250 # Sampling frequency of 250 Hz as per the BCI competion dataset 2a

# Classify the signal using a growing window
# Define initial window parameters
initial_window_length = int(sfreq * 0.5)  # Initial window length
w_step = int(sfreq * 0.1)  # Window step size
w_start = np.arange(0, epochs_data.shape[2] - initial_window_length, w_step)  # Set of starting positions in the signal (Note! the signal is 2s to 4s)
w_length = int(sfreq * 0.5)  # Window length - Hyperparameter.
print("w start shape: ", w_start.shape)
#print("w start: \n", w_start)
scores_windows = [] 
#threshold = 0.5
entropy_windows = []
# Running classification across the signal
for train_idx, test_idx in cv_split:
    #print("train idx: ", train_idx)
    #print("test idx: ", test_idx)
    print("nr train_index:", len(train_idx))
    print("nr test_index:", len(test_idx))
    y_train, y_test = labels[train_idx], labels[test_idx] # Get the current labels and data
    # Exatract spatial filters and transform the data as a whole
    X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
    X_test = csp.transform(epochs_data_train[test_idx]) #  why define and transform it here, and then do it later as well!

    # Fit the classifier on the training data
    lda.fit(X_train, y_train)
    w_times = (w_start + w_length / 2.0) / sfreq + epochs.tmin



    # Test the classifier on the windows. This is where we run over the signal
    preds_this_window = []
    probs_this_window = []
    score_this_window = []
    entropy_this_window = []
    confidence_this_window = []
    chosen_epoch = 18 #Choose an epoch between 1 and 116. As the testset is 20% 116 epochs of the whole data 576 epochs
    for n, window_start in enumerate(w_start):
        window_length = initial_window_length + n * w_step
        X_test  = csp.transform(epochs_data[test_idx][:, :, window_start:(window_start + window_length)])
        print("X_test  shape:\n",X_test.shape)
        X_test_1_epoch = X_test [chosen_epoch] #Chooosing a specific epoch in the test set 
        print("X_test_1_epoch shape:\n", X_test_1_epoch.shape)

        #Accuracy
        score = lda.score(X_test_1_epoch.reshape(1, -1), [y_test[chosen_epoch]])
        score_this_window.append(score)
        
        probabilities = lda.predict_proba([X_test_1_epoch])
        
        if len(probs_this_window) == 0:
            probs_this_window = probabilities
        else:
            probs_this_window = np.vstack((probs_this_window, probabilities))

        prediction = lda.predict([X_test_1_epoch])
        preds_this_window.append(prediction)

        print("Prediction for this time window: ", prediction)
        print("prob shape: ", probabilities.shape)
        print("probabilities: \n", probabilities)

        #predictive entropy - H_pred(p) 
        entropy_score = entropy(pk = probabilities, axis = 1, base = len(probabilities[0]))
        print(entropy_score) #- see if entropy is better than probabilites
        entropy_this_window.append(entropy_score)

        '''
        Confidence - as seen in: 
        Uncertainty Quantification in Machine Learning for Biosignal Applications - A Review, page 13.
        1 - H_pred(p) can be used as a confidence measure. Normalizing seems useful - 1 / (1- entropy-score)
        '''
        #confidence
        confidence = 1 - entropy_score
        confidence_this_window.append(confidence)

    class_names = {
        1: "Left hand",
        2: "Right hand",
        3: "Both feet",
        4: "Tongue"
    }

    ##Probabiltiies for each of the classes for each window
    plt.plot(w_times, probs_this_window, label=[class_names[label] for label in [1, 2, 3, 4]])
    plt.xlabel("Time (s)")
    plt.ylabel("Probabilities")
    plt.axvline(2, linestyle="--", color="k", label="Onset")
    plt.axhline(0.5, linestyle="-", color="k", label="Threshold")
    plt.title("Classification probabilities over Time")
    plt.legend()
    plt.show()

    ##Predictions for each window
    preds_this_window_plottable = [x+4 for x in preds_this_window]
    plt.plot(w_times, preds_this_window_plottable)
    plt.xlabel("Time (s)")
    plt.ylabel("Class prediction")
    plt.axvline(2, linestyle="--", color="k", label="Onset")
    plt.title("Classification over Time")
    plt.yticks([1, 2, 3, 4], [class_names[label] for label in [1, 2, 3, 4]])
    plt.legend(loc="lower right")
    plt.show()
    plt.show()

    ##Accuracy for each window
    plt.plot(w_times, score_this_window, label="Score")
    plt.xlabel("Time (s)")
    plt.ylabel("Accuracy")
    plt.axvline(2, linestyle="--", color="k", label="Onset")
    plt.axhline(class_balance, linestyle="-", color="k", label="Chance")
    plt.title("Classification accuracy over Time")
    plt.legend()
    plt.show()

    ##Entropy for each window 
    print("Entropythisiwnodw:", entropy_this_window)
    plt.plot(w_times, entropy_this_window, label="Score")
    plt.xlabel("Time (s)")
    plt.ylabel("Entropy ")
    plt.ylim(0,1)
    plt.axvline(2, linestyle="--", color="k", label="Onset")
    plt.title("Model entropy over Time")
    plt.legend()
    plt.show()

    plt.plot(w_times, confidence_this_window, label="Score")
    plt.xlabel("Time (s)")
    plt.ylabel("Confidence")
    plt.ylim(0,1)
    plt.axvline(2, linestyle="--", color="k", label="Onset")
    #plt.axhline(threshold, linestyle="-", color="k", label="Threshold")
    plt.title("Model confidence over Time")
    plt.legend()
    plt.show()


    y_test = y_test+4
    print("right label:", class_names[y_test[chosen_epoch]])
    break
   