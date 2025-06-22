import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from project.data_loader import load_data
from project.model_utils import hyperparameter_tuning
from project.config import TRACKING_URI, EXPERIMENT_NAME, MODEL_NAME

# ----------------------------------------
# Set MLflow tracking and experiment
# ----------------------------------------
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ----------------------------------------
# Load dataset and perform train/test split
# ----------------------------------------
df = load_data()
X = df.drop("Price", axis=1)
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ----------------------------------------
# Define hyperparameter grid
# ----------------------------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# ----------------------------------------
# Start MLflow run and train model
# ----------------------------------------
with mlflow.start_run():
    # Train and tune using grid search
    grid_search = hyperparameter_tuning(X_train, y_train, param_grid)
    best_model = grid_search.best_estimator_

    # Evaluate on test data
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Log best hyperparameters and metric
    for k, v in grid_search.best_params_.items():
        mlflow.log_param(k, v)
    mlflow.log_metric("mse", mse)

    # Log the model as an artifact (but not register here)
    signature = infer_signature(X_train, y_train)
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        signature=signature
    )

    # Register the model explicitly to avoid warnings
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"

    mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    
    print(f"✅ Model registered with run ID: {mlflow.active_run().info.run_id}")
    print(f"🔢 MSE: {mse:.4f}")
