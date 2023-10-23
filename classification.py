import streamlit as st
from keras import Sequential
from keras.src.preprocessing.text import Tokenizer

from keras.src.utils import pad_sequences
from matplotlib import pyplot as plt
from sklearn import model_selection, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC

from util import print_4_metrics, draw_confusion_matrix


def logistic_regression(X,Y):
    st.header("Logistic Regression : ")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=4)
    log_reg = LogisticRegression(penalty=None)
    log_reg.fit(X_train, y_train)
    log_predicted = log_reg.predict(X_test)
    log_score = log_reg.predict_proba(X_test)[:, 1]

    print_4_metrics(y_test, log_predicted)

    st.subheader("Confusion Matrix: ")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    draw_confusion_matrix(y_test, log_predicted, ['Not Sarcastic', 'Sarcastic'])

    st.subheader("Testing lambda values to optimize logistic regression model")

    lambda_list = np.linspace(start=0, stop=1, num=20)
    # In order to avoid over fitting, we regularize the parameters. A higher lambda means more penalty and less shrinkage and vice versa
    # L1 regularization adds the absolute value of the co-efficients to the cost function,while L2 regularization adds the square of the co-efficient to the cost function.
    # Both types of regularization have a parameter called lambda, which controls how much penalty is applied to the co-efficients.

    accuracy_list = []
    for lambd in lambda_list:
        kfold = model_selection.KFold(n_splits=10, random_state=42, shuffle=True)
        # Train the model
        log_model_kfold_train = LogisticRegression(C=1 / lambd)
        log_results_kfold_train = model_selection.cross_val_score(log_model_kfold_train, X_train, y_train, cv=kfold)
        # Calculate accuracy
        accuracy = log_results_kfold_train.mean()
        # Append lambda and accuracy to the lists
        accuracy_list.append(accuracy)
        str = "For lambda=", lambd, ", accuracy across folds is: %.2f%%" % (log_results_kfold_train.mean() * 100.0)
        st.text(str)

    # Plot the scatter plot
    plt.figure()
    plt.scatter(lambda_list, accuracy_list)
    plt.plot(lambda_list, accuracy_list)
    plt.xlabel('Lambda (Regularization Parameter)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Lambda')
    st.pyplot(plt.show())

    st.subheader("Optimal Lambda is : 0.2631578947368421")

    lambd = 0.2631578947368421  # this is the optimal lambda

    log_reg = LogisticRegression(penalty='l2', C=1 / lambd)
    log_reg.fit(X_train, y_train)
    log_predicted = log_reg.predict(X_test)
    log_score = log_reg.predict_proba(X_test)[:, 1]


    print_4_metrics(y_test, log_predicted)

    st.subheader("Confusion Matrix: ")
    draw_confusion_matrix(y_test, log_predicted, ['Not Sarcastic', 'Sarcastic'])

def svm(X,Y):
    st.header("Support Vector Machine : ")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2,
                                                        random_state=4)
    svm = SVC(probability=True, C=1e10)

    svm.fit(X_train, y_train)

    svm_predicted = svm.predict(X_test)
    svm_score = svm.predict_proba(X_test)[:, 1]

    print_4_metrics(y_test, svm_predicted)

    # Plot ROC curve and report area under ROC
    # use metrics.roc_curve(your y_test, predicted probabilities for y_test)
    # feel free to use the same code as 3.1.1.
    fpr_svm_reg, tpr_svm_reg, thresholds = metrics.roc_curve(y_test, svm_score)
    st.subheader("SVM Model Performance Results:\n")
    plt.figure(1)
    plt.plot(fpr_svm_reg, tpr_svm_reg, color='orange', lw=1)
    # The ROC curve is the plot of the true positive rate (TPR) against the false positive rate (FPR), at various threshold settings.
    plt.title("ROC curve with SVM Regression (rbf Kernel)")
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    st.pyplot(plt.show())
    # # report auc
    # # use metrics.auc(fpr, tpr)
    aucroc = metrics.auc(fpr_svm_reg, tpr_svm_reg)
    st.text("Area under curve : ")
    st.text(aucroc)
    #
    st.subheader("Confusion Matrix: \n")
    draw_confusion_matrix(y_test, svm_predicted, ['Not Sarcastic', 'Sarcastic'])
    #
    lambda_list = np.linspace(start=1, stop=2, num=20)
    accuracy_list = []
    for lambd in lambda_list:
        kfold = model_selection.KFold(n_splits=10, random_state=42, shuffle=True)
        # Train the model
        svc_model_kfold_train = SVC(probability=True, C=1 / lambd)
        svc_results_kfold_train = model_selection.cross_val_score(svc_model_kfold_train, X_train, y_train, cv=kfold)
        # Calculate accuracy
        accuracy = svc_results_kfold_train.mean()
        # Append lambda and accuracy to the lists
        accuracy_list.append(accuracy)
        str = "For lambda=", lambd, ", accuracy across folds is: %.2f%%" % (svc_results_kfold_train.mean() * 100.0)
        st.text(str)
    # # Plot the scatter plot
    plt.figure()
    plt.scatter(lambda_list, accuracy_list)
    plt.plot(lambda_list, accuracy_list)
    plt.xlabel('Lambda (Regularization Parameter)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Lambda')
    st.pyplot(plt.show())

    st.text("label\tP(Y=1)\tCorrect?")
    for i in range(0, 10):
        st_ = (y_test.iloc[i], " ", svm_score[i], " ",
              (y_test.iloc[i] == 1 and svm_score[i] >= 0.5) or (y_test.iloc[i] == 0 and svm_score[i] < 0.5))
        st.text(st_)

    lambd = 1.263157894736842
    svm = SVC(probability=True, C=1 / lambd)

    svm.fit(X_train, y_train)

    svm_predicted = svm.predict(X_test)
    svm_score = svm.predict_proba(X_test)[:, 1]

    print_4_metrics(y_test, svm_predicted)

    # Plot ROC curve and report area under ROC
    # use metrics.roc_curve(your y_test, predicted probabilities for y_test)
    # feel free to use the same code as 3.1.1.
    fpr_svm_reg, tpr_svm_reg, thresholds = metrics.roc_curve(y_test, svm_score)
    st.subheader("SVM Model Performance Results:")
    plt.figure(1)
    plt.plot(fpr_svm_reg, tpr_svm_reg, color='orange', lw=1)
    plt.title("ROC curve with SVM Regression (rbf Kernel)")
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    st.pyplot(plt.show())
    # report auc
    # use metrics.auc(fpr, tpr)
    aucroc = metrics.auc(fpr_svm_reg, tpr_svm_reg)
    st.text("Area under curve : ")
    st.text(aucroc)