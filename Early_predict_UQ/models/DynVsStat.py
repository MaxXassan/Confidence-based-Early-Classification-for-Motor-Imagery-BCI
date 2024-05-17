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
    best_itr_lda_expanding = main_lda_expanding()
    best_itr_lda_sliding = main_lda_sliding()
    print("-----LDA-----")
    print(f"LDA Expanding Information Transfer Rate (ITR) - {best_itr_lda_expanding}, LDA Sliding Information Transfer Rate (ITR) -  {best_itr_lda_sliding}")
    if(best_itr_lda_expanding > best_itr_lda_sliding):
        print(f"Expaning lda model has a higher ITR value than Sliding lda model\n")
    elif(best_itr_lda_expanding == best_itr_lda_sliding):
        print(f"Expaning and sliding lda models have equal ITR value\n")
    else:
        print(f"Sliding lda model has a higher ITR value than Expanding lda model\n")
    print("-----SVM-----")
    best_itr_svm_expanding = main_svm_expanding()
    best_itr_svm_sliding = main_svm_sliding()
    print(f"SVM Expanding Information Transfer Rate (ITR) - {best_itr_svm_expanding}, SVM Sliding Information Transfer Rate (ITR) -  {best_itr_svm_sliding}")
    if(best_itr_svm_expanding > best_itr_svm_sliding):
        print(f"Expaning lda model has a higher ITR value than Sliding lda model\n")
    elif(best_itr_svm_expanding == best_itr_svm_sliding):
        print(f"Expaning and sliding lda models have equal ITR value\n")
    else:
        print(f"Sliding lda model has a higher ITR value than Expanding lda model\n")

if __name__ == "__main__":
    main_dyn_vs_stat()