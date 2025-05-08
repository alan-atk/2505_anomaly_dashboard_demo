import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn.preprocessing import RobustScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from st_table_select_cell import st_table_select_cell
# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.metrics import classification_report


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    rob_scaler = RobustScaler()
    df["scaled_amount"] = rob_scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
    df["scaled_time"] = rob_scaler.fit_transform(df["Time"].values.reshape(-1, 1))

    df.drop(["Time", "Amount"], axis=1, inplace=True)

    scaled_amount = df["scaled_amount"]
    scaled_time = df["scaled_time"]

    df.drop(["scaled_amount", "scaled_time"], axis=1, inplace=True)
    df.insert(0, "scaled_amount", scaled_amount)
    df.insert(1, "scaled_time", scaled_time)

    return df


def sample_data(df: pd.DataFrame, random_seed) -> pd.DataFrame:
    df = df.sample(frac=1)
    fraud_df = df.loc[df["Class"] == 1]
    non_fraud_df = df.loc[df["Class"] == 0][:len(fraud_df)]
    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

    return normal_distributed_df.sample(frac=1, random_state=random_seed)


@st.cache_data
def run_model(data_list: list, random_seed: int, test_size: float) -> pd.DataFrame:
    dict_result = collections.defaultdict(list)
    for data in data_list:
        dict_result["OrderID"].append("XXX")
        dict_result["OrderEntity"].append("XXX")
        dict_result["Quantity"].append("XXX")
        dict_result["Amount"].append("XXX")

        X = data.data
        y = data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

        classifiers = {
            "IsolationForest": IsolationForest(random_state=random_seed),
            "OneClassSVM": OneClassSVM(),
            "LogisticRegression": LogisticRegression(random_state=random_seed),
            "RandomForest": RandomForestClassifier(random_state=random_seed),
            "GradientBoosting": GradientBoostingClassifier(random_state=random_seed),
            "KNeighbors": KNeighborsClassifier(),
            "DecisionTree": DecisionTreeClassifier(random_state=random_seed),
        }

        for key, classifier in classifiers.items():
            try:
                classifier.fit(X_train)
            except TypeError:
                classifier.fit(X_train, y_train)
            y_pred_test = classifier.predict(X_test)
            y_pred_test = np.where(y_pred_test == 1, 1, 0)
            report = classification_report(y_test, y_pred_test, output_dict=True)
            dict_result[key].append(f"{round(report['accuracy']*100, 1)}%")

    df_result = pd.DataFrame(dict_result)
    return df_result


def main():
    st.set_page_config(layout="wide")
    st.title("2505_anomaly_detection_demo_v1")

    st.sidebar.title("Sidebar")
    random_seed = int(st.sidebar.text_input("Random seed:", "42"))
    test_size = float(st.sidebar.text_input("Random seed:", "0.2"))

    # dummy datasets
    data_list = [datasets.load_breast_cancer(), datasets.load_iris(), datasets.load_wine()]
    df_result = run_model(data_list, random_seed, test_size)

    classifiers = {
        "IsolationForest": IsolationForest(random_state=random_seed),
        "OneClassSVM": OneClassSVM(),
        "LogisticRegression": LogisticRegression(random_state=random_seed),
        "RandomForest": RandomForestClassifier(random_state=random_seed),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_seed),
        "KNeighbors": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier(random_state=random_seed),
    }

    st.dataframe(df_result)
    selectedCell = st_table_select_cell(df_result)

    if selectedCell and int(selectedCell["colIndex"]) > 3:
        classifier_key = list(classifiers.keys())[int(selectedCell["colIndex"]) - 4]
        classifier = classifiers[classifier_key]
        selected_dataset_id = int(selectedCell["rowId"])
        selected_dataset = data_list[selected_dataset_id]

        X = selected_dataset.data
        y = selected_dataset.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

        try:
            classifier.fit(X_train)
        except TypeError:
            classifier.fit(X_train, y_train)
        y_pred_test = classifier.predict(X_test)
        y_pred_test = np.where(y_pred_test == 1, 1, 0)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_test, y_pred_test, alpha=0.5)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Predicted vs Actual for {classifier_key}")
        st.pyplot(fig)


@st.cache_data
def get_raw_data(filename: str) -> pd.DataFrame:
    df_raw = pd.read_csv(filename)
    return df_raw


if __name__ == "__main__":
    main()
