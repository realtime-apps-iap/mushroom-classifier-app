
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from functions import load_data, split_data, plot_metrics


def main():
    st.title("Machine Learning Binary Classification Web App")
    st.markdown("Classifying the mushrooms 🍄 to be Edible or Poisonous")
    st.markdown("(Configure classifier options on the left side menu)")
    st.sidebar.title("Machine Learning Binary Classification Web App")
    st.sidebar.markdown(
        "Classifying the mushrooms 🍄 to be Edible or Poisonous")

    df = load_data()

    if st.checkbox("Show raw data", True):
        st.subheader("Mushroom Data Set (Classification")
        st.write(df)

    x_train, x_test, y_train, y_test = split_data(df)
    class_names = ['edible', 'poisonous']
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox(
        "Classifier", ("Support Vector Machine (SVM)", "Logistic Regression",  "Random Forest"))

    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input(
            "C (Regularisation parameter)", 0.01, 10.0, step=0.01, key="C")
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel")
        gamma = st.sidebar.radio(
            "Gamma (Kernal Coefficient)", ("scale", "auto"), key="gamma")
        metrics = st.sidebar.multiselect("What metrics to plot?", (
            'Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'), key='metrics')

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results: ")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(
                y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(
                y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics, model, x_test, y_test, class_names)

    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input(
            "C (Regularisation parameter)", 0.01, 10.0, step=0.01, key="C_LR")
        max_iter = st.sidebar.slider(
            "Maximum number of iterations", 100, 500, key='max_iter')
        metrics = st.sidebar.multiselect("What metrics to plot?", (
            'Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'), key='metrics')

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results: ")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(
                y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(
                y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics, model, x_test, y_test, class_names)

    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input(
            "The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input(
            "The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio(
            "Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')

        metrics = st.sidebar.multiselect("What metrics to plot?", (
            'Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'), key='metrics')

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results: ")
            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(
                y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(
                y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics, model, x_test, y_test, class_names)


if __name__ == '__main__':
    print("Running app.py")
    main()
