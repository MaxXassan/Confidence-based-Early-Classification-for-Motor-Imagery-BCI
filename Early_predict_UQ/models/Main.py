import os
import sys

current_directory = os.path.abspath('')

project_root = current_directory

sys.path.append(project_root)
print("ROOT:", project_root)


from Early_predict_UQ.models.LDA_models.DynVsStat_LDA_expanding import main_lda_expanding
from Early_predict_UQ.models.LDA_models.DynVsStat_LDA_sliding import main_lda_sliding
from Early_predict_UQ.models.SVM_models.DynVsStat_SVM_expanding import main_svm_expanding
from Early_predict_UQ.models.SVM_models.DynVsStat_SVM_sliding import main_svm_sliding

def main_dyn_vs_stat():
    results = []

    print("-----LDA-----")
    print("LDA - Expanding")
    itr_dyn_lda_expanding, itr_stat_lda_expanding = main_lda_expanding()
    results.append(("LDA - Expanding Dynamic", itr_dyn_lda_expanding))
    results.append(("LDA - Expanding Static", itr_stat_lda_expanding))

    print("LDA - Sliding")
    itr_dyn_lda_sliding, itr_stat_lda_sliding = main_lda_sliding()
    results.append(("LDA - Sliding Dynamic", itr_dyn_lda_sliding))
    results.append(("LDA - Sliding Static", itr_stat_lda_sliding))

    print("-----SVM-----")
    print("SVM - Expanding")
    itr_dyn_svm_expanding, itr_stat_svm_expanding = main_svm_expanding()
    results.append(("SVM - Expanding Dynamic", itr_dyn_svm_expanding))
    results.append(("SVM - Expanding Static", itr_stat_svm_expanding))

    print("SVM - Sliding")
    itr_dyn_svm_sliding, itr_stat_svm_sliding = main_svm_sliding()
    results.append(("SVM - Sliding Dynamic", itr_dyn_svm_sliding))
    results.append(("SVM - Sliding Static", itr_stat_svm_sliding))

    # Sort the results based on the values (you can choose to sort by dynamic or static values)
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    # Print the sorted order
    print("----- Model Order Based on Values -----")
    for model, value in sorted_results:
        print(f"{model}: {value}")


if __name__ == "__main__":
    main_dyn_vs_stat()