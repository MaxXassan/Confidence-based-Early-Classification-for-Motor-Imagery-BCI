from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery


'''
2 sessions, 
    - 6 runs in each session, 
    - 12*4(=48) trials in each run,
    - 288 trials in each session.
    - 25 channels - (first 22 are EEG, last 3 are EOG) - need to remove those
    - cue onset classes and event type values: in training data 1, 2, 3, 4 -> 769, 770, 771, 772
    - trials containing artifacts as scored by experts are marked as events
      with the type 1023
'''

## (Need to adjust for multiple people)

# Given a subject list, return the preprocessed and epoched data
def make_data(subject_list):
    dataset = BNCI2014_001()
    paradigm = MotorImagery(fmin=7, fmax=30) # Bandpass filter between to enhance mu and beta frequencies

    epochs, labels, meta = paradigm.get_data( 
            dataset=dataset, subjects=subject_list, return_epochs=True
        )
    
    return epochs, labels
