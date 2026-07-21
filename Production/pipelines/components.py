from kfp.dsl import component, Output, Artifact, Dataset, Model, Input

"""This file contains your modular, reusable pipeline steps. 
Each function uses the @component decorator to define its base execution environment, required packages, and input/output artifacts."""


@component(base_image="python:3.11-slim", packages_to_install=["pandas", "pyarrow"])
def data_ingestion_and_split(
    dataset_path: str, fraud_dataset: Output[Dataset], sales_dataset: Output[Dataset]
):
    """Ingests the main parquet file and partitions it for dual-model target streams."""
    import pandas as pd
    import os

    print(f"📥 Ingesting dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    df = pd.read_parquet(dataset_path)

    # Split 1: Data for fraud detection
    fraud_df = df.drop(columns=["order_id"], errors="ignore")
    fraud_df.to_parquet(fraud_dataset.path, index=False)

    # Split 2: Data for sales prediction
    sales_df = df.drop(columns=["fraud", "order_id"], errors="ignore")
    sales_df.to_parquet(sales_dataset.path, index=False)

    print(f"📊 Dataset partitioned successfully: {len(df)} records processed.")


# ===========================================
# Fraud Detection Model Training Components
# ===========================================


@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "pyarrow", "scikit-learn", "xgboost"],
)
def train_fraud_xgb_model(dataset: Input[Dataset], model_output: Output[Model]):
    """Trains an XGBoost classifier to detect anomalies or fraudulent transactions."""
    import pandas as pd
    import pickle
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_parquet(dataset.path)

    # Target and features
    X = df.drop(columns=["fraud"])
    y = df["fraud"]

    # Encode categorical features if any
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🚀 Training XGBoost model for fraud detection...")
    model = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
    model.fit(X_train, y_train)

    # Save as .pkl format for the inference gateway
    with open(model_output.path + ".pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model trained and saved to: {model_output.path}.pkl")


@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "pyarrow", "scikit-learn"],
)
def train_fraud_rf_model(dataset: Input[Dataset], model_output: Output[Model]):
    """Trains a Random Forest classifier to detect anomalies or fraudulent transactions."""
    import pandas as pd
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_parquet(dataset.path)

    # Target and features
    X = df.drop(columns=["fraud"])
    y = df["fraud"]

    # Encode categorical features if any
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🚀 Training Random Forest model for fraud detection...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Save as .pkl format for the inference gateway
    with open(model_output.path + ".pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model trained and saved to: {model_output.path}.pkl")


@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "pyarrow", "scikit-learn"],
)
def train_fraud_gbc_model(dataset: Input[Dataset], model_output: Output[Model]):
    """Trains an Gradient Boosting classifier to detect anomalies or fraudulent transactions."""
    import pandas as pd
    import pickle
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_parquet(dataset.path)

    # Target and features
    X = df.drop(columns=["fraud"])
    y = df["fraud"]

    # Encode categorical features if any
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🚀 Training Gradient Boosting model for fraud detection...")
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
    model.fit(X_train, y_train)

    # Save as .pkl format for the inference gateway
    with open(model_output.path + ".pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model trained and saved to: {model_output.path}.pkl")


# ===========================================
# Sales Prediction Model Training Components
# ===========================================
@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "pyarrow", "scikit-learn", "xgboost"],
)
def train_sales_xgb_model(dataset: Input[Dataset], model_output: Output[Model]):
    """Trains an XGBoost regressor to predict sales."""
    import pandas as pd
    import joblib
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_parquet(dataset.path)

    # Target and features
    X = df.drop(columns=["sales"])
    y = df["sales"]

    # Encode categorical features if any
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🚀 Training XGBoost model for sales prediction...")
    model = XGBRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
    model.fit(X_train, y_train)

    # Save as .joblib format for the inference gateway
    joblib.dump(model, model_output.path + "_model.joblib")
    print(f"✅ Model trained and saved to: {model_output.path}_model.joblib")


@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "pyarrow", "scikit-learn"],
)
def train_sales_rf_model(dataset: Input[Dataset], model_output: Output[Model]):
    """Trains a Random Forest regressor to predict sales."""
    import pandas as pd
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_parquet(dataset.path)

    # Target and features
    X = df.drop(columns=["sales"])
    y = df["sales"]

    # Encode categorical features if any
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🚀 Training Random Forest model for sales prediction...")
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Save as .joblib format for the inference gateway
    joblib.dump(model, model_output.path + "_model.joblib")
    print(f"✅ Model trained and saved to: {model_output.path}_model.joblib")


@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "pyarrow", "scikit-learn"],
)
def train_sales_cbr_model(dataset: Input[Dataset], model_output: Output[Model]):
    """Trains a CatBoost regressor to predict sales."""
    import pandas as pd
    import joblib
    from catboost import CatBoostRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_parquet(dataset.path)

    # Target and features
    X = df.drop(columns=["sales"])
    y = df["sales"]

    # Encode categorical features if any
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🚀 Training CatBoost model for sales prediction...")
    model = CatBoostRegressor(
        iterations=1000, depth=5, learning_rate=0.1, loss_function="RMSE", verbose=0
    )
    model.fit(X_train, y_train)

    # Save as .joblib format for the inference gateway
    joblib.dump(model, model_output.path + "_model.joblib")
    print(f"✅ Model trained and saved to: {model_output.path}_model.joblib")


@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "pyarrow", "scikit-learn"],
)
def train_sales_dt_model(dataset: Input[Dataset], model_output: Output[Model]):
    """Trains a Decision Tree regressor to predict sales."""
    import pandas as pd
    import joblib
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_parquet(dataset.path)

    # Target and features
    X = df.drop(columns=["sales"])
    y = df["sales"]

    # Encode categorical features if any
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🚀 Training Decision Tree model for sales prediction...")
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Save as .joblib format for the inference gateway
    joblib.dump(model, model_output.path + "_model.joblib")
    print(f"✅ Model trained and saved to: {model_output.path}_model.joblib")
