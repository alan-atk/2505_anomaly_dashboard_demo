import streamlit as st
from st_table_select_cell import st_table_select_cell
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


@st.cache_data
def generate_data(random_seed=42) -> np.ndarray:
    np.random.seed(random_seed)

    # === 1. Generate Normal Data (bulk of the dataset)
    quantity_normal = np.random.normal(loc=100, scale=10, size=300)
    price_normal = np.random.normal(loc=1000, scale=100, size=300)
    # === 2. Inject Anomalies of Different Types
    # a. Extreme price, normal quantity
    quantity_a = np.random.normal(loc=100, scale=5, size=5)
    price_a = np.random.normal(loc=3000, scale=100, size=5)
    # b. Extreme quantity, normal price
    quantity_b = np.random.normal(loc=300, scale=10, size=5)
    price_b = np.random.normal(loc=1000, scale=100, size=5)
    # c. Both off but within bounds
    quantity_c = np.random.normal(loc=180, scale=5, size=5)
    price_c = np.random.normal(loc=1700, scale=50, size=5)
    # === 3. Combine Data
    X_normal = np.column_stack([quantity_normal, price_normal])
    X_anom = np.vstack([
        np.column_stack([quantity_a, price_a]),
        np.column_stack([quantity_b, price_b]),
        np.column_stack([quantity_c, price_c])
    ])
    return np.vstack([X_normal, X_anom])


def standardize_data(X_all: np.ndarray) -> np.ndarray:
    # Standardize features
    scaler = StandardScaler()
    return scaler.fit_transform(X_all)


def convert_to_percentage(score):
    return MinMaxScaler((0, 100)).fit_transform(score.reshape(-1, 1)).flatten()


@st.cache_data
def run_model(dataset, df_output, random_seed) -> pd.DataFrame:
    model_isolation_forest = IsolationForest(random_state=random_seed)
    model_isolation_forest.fit(dataset)
    score_isolation_forest = model_isolation_forest.decision_function(dataset)
    score_isolation_forest = 100 - convert_to_percentage(score_isolation_forest)
    df_output["IsolationForest"] = score_isolation_forest

    model_local_outlier_factor = LocalOutlierFactor()
    model_local_outlier_factor.fit_predict(dataset)
    score_local_outlier_factor = -model_local_outlier_factor.negative_outlier_factor_
    score_local_outlier_factor = convert_to_percentage(score_local_outlier_factor)
    df_output["LocalOutlierFactor"] = score_local_outlier_factor

    model_elliptic_envelope = EllipticEnvelope(random_state=random_seed)
    model_elliptic_envelope.fit(dataset)
    score_elliptic_envelope = -model_elliptic_envelope.decision_function(dataset)
    score_elliptic_envelope = convert_to_percentage(score_elliptic_envelope)
    df_output["EllipticEnvelope"] = score_elliptic_envelope

    return df_output


def main():
    st.set_page_config(layout="wide")
    st.title("2505_anomaly_detection_demo_v1")

    st.sidebar.title("Sidebar")
    random_seed = int(st.sidebar.text_input("Random seed:", "42"))

    data = generate_data(random_seed=random_seed)
    data_standardized = standardize_data(data)
    df_result = pd.DataFrame(data, columns=["数量", "単価"])
    df_result = run_model(data_standardized, df_result, random_seed=random_seed)

    df_result["総合評価"] = df_result[["IsolationForest", "LocalOutlierFactor", "EllipticEnvelope"]].mean(axis=1).round(1)

    np.random.seed(random_seed)
    df_result.insert(0, "注文番号", [f"ORD_{str(i).zfill(4)}" for i in range(len(df_result))])
    df_result.insert(1, "注文主体", np.random.choice(["部門A", "部門B", "部門C"], size=len(df_result)))
    df_result.set_index("注文番号", inplace=True)
    df_result["IsolationForest"] = df_result["IsolationForest"].round(1)
    df_result["LocalOutlierFactor"] = df_result["LocalOutlierFactor"].round(1)
    df_result["EllipticEnvelope"] = df_result["EllipticEnvelope"].round(1)
    df_result["EllipticEnvelope"] = df_result["EllipticEnvelope"].round(1)
    st.dataframe(df_result)


if __name__ == "__main__":
    main()
