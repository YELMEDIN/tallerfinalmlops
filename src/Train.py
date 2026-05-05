import os
import yaml
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from mlflow.models.signature import infer_signature


# =========================
# 1. Cargar configuración
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(BASE_DIR, "config.yaml")

with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


# =========================
# 2. Configuración MLflow
# =========================
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])


# =========================
# 3. Cargar dataset
# =========================
data_path = os.path.join(BASE_DIR, config["data"]["path"])
target = config["data"]["target"]

df = pd.read_csv(data_path)

X = df.drop(columns=[target])
y = df[target]


# =========================
# 4. División train / test
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config["training"]["test_size"],
    random_state=config["model"]["random_state"]
)


# =========================
# 5. Identificación de columnas
# =========================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()


# =========================
# 6. Preprocesamiento
# =========================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# =========================
# 7. Modelo
# =========================
model = RandomForestRegressor(
    n_estimators=config["model"]["n_estimators"],
    max_depth=config["model"]["max_depth"],
    random_state=config["model"]["random_state"]
)


# =========================
# 8. Pipeline completo
# =========================
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])


# =========================
# 9. Entrenamiento, evaluación y tracking
# =========================
with mlflow.start_run(run_name="random_forest_ev_price"):

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Métricas del modelo:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.4f}")

    # Parámetros desde config.yaml
    mlflow.log_param("model_type", config["model"]["type"])
    mlflow.log_param("n_estimators", config["model"]["n_estimators"])
    mlflow.log_param("max_depth", config["model"]["max_depth"])
    mlflow.log_param("random_state", config["model"]["random_state"])
    mlflow.log_param("test_size", config["training"]["test_size"])
    mlflow.log_param("target", target)
    mlflow.log_param("data_path", config["data"]["path"])

    # Métricas
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Firma y ejemplo de entrada
    input_example = X_train.head(5)
    signature = infer_signature(X_train, pipeline.predict(X_train))

    # Guardar modelo como artefacto en MLflow
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        registered_model_name=config["mlflow"]["registered_model_name"]
    )

    # Guardar modelo local
    model_path = os.path.join(BASE_DIR, config["artifacts"]["model_path"])
    joblib.dump(pipeline, model_path)

    print("Modelo entrenado, evaluado y registrado correctamente.")
    print(f"Modelo guardado como {model_path}")