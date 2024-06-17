import matplotlib.pyplot as plt
import numpy as np

##Plot over time and epochs
def plot_cost_over_time_and_epochs(w_times, costs, predict_time):
    plt.plot(w_times, costs, label='Cost')
    plt.xlabel("Time (s)")
    plt.ylabel("Cost")
    plt.axvline(w_times[predict_time], linestyle="-", color="k", label="Stopping")
    plt.axvline(2, linestyle="--", color="k", label="Onset")
    plt.axhline(0.5, linestyle="-", color="k", label="Threshold")
    plt.title("Cost over Time")
    plt.legend()
    plt.show()

def plot_accuracy_over_time_and_epochs(w_times, scores_windows, predict_time, class_balance):
    plt.figure()

    plt.plot(w_times, np.mean(np.transpose(scores_windows), 1), label="Score")
    plt.axvline(2, linestyle="--", color="k", label="Onset")
    plt.axvline(w_times[predict_time], linestyle="-", color="k", label="Stopping")
    plt.axhline(class_balance, linestyle="-", color="k", label="Chance")
    plt.xlabel("time (s)")
    plt.ylabel("classification accuracy")
    plt.title("Classification score over time")
    plt.legend(loc="lower right")
    plt.show()

def plot_confidence_over_time_and_epochs(w_times, confidence_windows, predict_time, threshold):
    plt.figure()
    plt.plot(w_times, np.mean(confidence_windows, 0), label="Scores")
    plt.xlabel("Time (s)")
    plt.ylabel("Confidence")
    plt.ylim(0,1)
    plt.axvline(2, linestyle="--", color="k", label="Onset")
    plt.axvline(w_times[predict_time], linestyle="-", color="k", label="Stopping")
    #plt.axhline(threshold, linestyle="-", color="k", label="Threshold")
    plt.title("Model confidence over Time")
    plt.legend()
    plt.show()

def plots_over_time_and_epochs(w_times, scores_windows, entropy_windows, confidence_windows, threshold):
    plt.figure()
    scores = np.mean(np.transpose(scores_windows), 1)
    confidences = np.mean(confidence_windows, 0)
    entropys = np.mean(entropy_windows, 0)
    benefit = confidences + w_times
    plt.plot(w_times, scores, label="Scores")
    plt.plot(w_times, confidences, label="Confidence")
    plt.plot(w_times, benefit, label="Benefit")
    plt.xlabel("Time (s)")
    plt.ylabel("Confidence")
    plt.ylim(0,1)
    plt.axvline(2, linestyle="--", color="k", label="Onset")
    #plt.axhline(threshold, linestyle="-", color="k", label="Threshold")
    plt.title("Model confidence over Time")
    plt.legend()
    plt.show()

def plot_probabilities_over_time_and_epochs(w_times, probs_windows, threshold, class_names):
    plt.figure()
    plt.plot(w_times, np.mean(probs_windows, 0), label=[class_names[label] for label in [1, 2, 3, 4]])
    plt.xlabel("Time (s)")
    plt.ylabel("Probabilities")
    plt.axvline(2, linestyle="--", color="k", label="Onset")
    plt.axhline(0.5, linestyle="-", color="k", label="Chance")
    plt.title("Classification probabilities over Time")
    plt.legend()
    plt.show()

def plot_entropy_over_time_and_epochs(w_times,entropy_windows):
    plt.figure()
    plt.plot(w_times, np.mean(entropy_windows, 0), label="Score")
    plt.xlabel("Time (s)")
    plt.ylabel("Entropy ")
    plt.ylim(0,1)
    plt.axvline(2, linestyle="--", color="k", label="Onset")
    plt.title("Model entropy over Time")
    plt.legend()
    plt.show()

##Plot over time for 1 epoch
def plot_accuracy_over_time(w_times, scores_this_window, class_balance):
    plt.figure()
    plt.plot(w_times, scores_this_window, label="Score")
    plt.axvline(2, linestyle="--", color="k", label="Onset")
    plt.axhline(class_balance, linestyle="-", color="k", label="Chance")
    plt.xlabel("time (s)")
    plt.ylabel("classification accuracy")
    plt.title("Classification score over time")
    plt.legend(loc="lower right")
    plt.show()


def plot_confidence_over_time(w_times, confidence_this_window, threshold):
    plt.figure()
    plt.plot(w_times, confidence_this_window, label="Score")
    plt.xlabel("Time (s)")
    plt.ylabel("Confidence")
    plt.ylim(0,1)
    plt.axvline(2, linestyle="--", color="k", label="Onset")
    #plt.axhline(threshold, linestyle="-", color="k", label="Threshold")
    plt.title("Model condidence over Time")
    plt.legend()
    plt.show()

def plot_probabilities_over_time(w_times, probs_this_window, class_names, class_balance):
    plt.figure()
    plt.plot(w_times, probs_this_window, label=[class_names[label] for label in [1, 2, 3, 4]])
    plt.xlabel("Time (s)")
    plt.ylabel("Probabilities")
    plt.axvline(2, linestyle="--", color="k", label="Onset")
    plt.axhline(class_balance, linestyle="-", color="k", label="Chance")
    plt.title("Classification probabilities over Time")
    plt.legend()
    plt.show()

def plot_entropy_over_time(w_times,entropy_this_window):
    plt.figure()
    plt.plot(w_times, entropy_this_window, label="Score")
    plt.xlabel("Time (s)")
    plt.ylabel("Entropy ")
    plt.ylim(0,1)
    plt.axvline(2, linestyle="--", color="k", label="Onset")
    plt.title("Model entropy over Time")
    plt.legend()
    plt.show()

def plot_predictions_over_time(w_times, preds_this_window, class_names):
    preds_this_window_plottable = [x+4 for x in preds_this_window]
    plt.figure()
    plt.plot(w_times, preds_this_window_plottable)
    plt.xlabel("Time (s)")
    plt.ylabel("Class prediction")
    plt.axvline(2, linestyle="--", color="k", label="Onset")
    plt.title("Classification over Time")
    plt.yticks([1, 2, 3, 4], [class_names[label] for label in [1, 2, 3, 4]])
    plt.legend(loc="lower right")
    plt.show()