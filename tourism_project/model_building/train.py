import os
import json
import inspect
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score, roc_auc_score
import mlflow
import joblib
import urllib.request

# --- Configure Tracking Server URI ---
# Point MLflow to track runs on the local server at port 5001
tracking_url = "http://127.0.0.1:5001"
try:
    # Perform a network socket ping to see if local port 5001 is listening
    urllib.request.urlopen(tracking_url, timeout=2)
    mlflow.set_tracking_uri(tracking_url)
    print(f"Connected to local MLflow tracking server at {tracking_url}")
except Exception as e:
    # Fall back to default file-based logger if port is unreachable
    print(f"Local tracking server on port 5001 not reachable ({e}). Falling back to default file-based logging.")

# Configure the master tracking experiment workspace name
mlflow.set_experiment("Tourism_Package_Prediction")

def train_production_model():
    # --- Define Input Directories and Paths ---
    splits_dir = "tourism_project/data_splits"
    X_train_path = os.path.join(splits_dir, "Xtrain.csv")
    X_test_path = os.path.join(splits_dir, "Xtest.csv")
    y_train_path = os.path.join(splits_dir, "ytrain.csv")
    y_test_path = os.path.join(splits_dir, "ytest.csv")

    # --- Load Preprocessed Dataset Splits ---
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    y_test = pd.read_csv(y_test_path).values.ravel()

    # --- Calculate Class Imbalance Weight Dynamically ---
    # Programmatically determine negative-to-positive class distribution
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    # --- Configure Production Run Experiment Context ---
    mlflow.set_experiment("Tourism_Wellness_Pack_Prod")

    # --- Load Validated Optimal Hyperparameters from Dev search ---
    params_path = "tourism_project/model_building/best_params.json"
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            prod_params = json.load(f)
        print(f"Successfully loaded optimized parameters from Dev: {prod_params}")
    else: 
        # Define default baseline parameters if parameters map file is missing
        prod_params = {
            'max_depth': 7,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'subsample': 0.8
        }
        print(f"Optimal parameters config not found. Falling back to defaults: {prod_params}")

    print("Starting Streamlined Production Training...")
    # --- Start Tracking the Training Run ---
    with mlflow.start_run(run_name="XGB_Production_Run") as run:
        # Initialize XGBoost classifier with our hyperparameters and balance weight
        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            **prod_params
        )

        # Train the model on the preprocessed training dataset
        model.fit(X_train, y_train)

        # --- Evaluation on Training Data ---
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]

        # Calculate performance metrics on training data
        train_acc = accuracy_score(y_train, y_train_pred)
        train_prec = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        train_auc = roc_auc_score(y_train, y_train_proba)

        # --- Evaluation on Test Data ---
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate performance metrics on testing data
        test_acc = accuracy_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)
        test_auc = roc_auc_score(y_test, y_proba)

        # --- Log Model Hyperparameters ---
        mlflow.log_params(prod_params)

        # --- Log Calculated Training Metrics ---
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("train_precision", train_prec)
        mlflow.log_metric("train_recall", train_recall)
        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("train_roc_auc", train_auc)

        # --- Log Calculated Testing Metrics ---
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_precision", test_prec)
        mlflow.log_metric("test_recall", test_recall)

        # --- Dynamic Log Model Parameter Check ---
        # Check if the environment's mlflow.sklearn.log_model accepts 'skops_trusted_types'
        sig = inspect.signature(mlflow.sklearn.log_model)
        if 'skops_trusted_types' in sig.parameters:
            mlflow.sklearn.log_model(
                model,
                "model",
                skops_trusted_types=['xgboost.core.Booster', 'xgboost.sklearn.XGBClassifier']
            )
        else:
            # Fallback for older MLflow versions that do not support or require skops constraints
            mlflow.sklearn.log_model(model, "model")

        # --- Save Serialized Model Locally for App Deployments ---
        deployment_dir = "tourism_project/deployment"
        os.makedirs(deployment_dir, exist_ok=True)
        model_path = os.path.join(deployment_dir, "best_model.joblib")
        joblib.dump(model, model_path)

        # Register local model file as a workflow tracking artifact
        mlflow.log_artifact(model_path, artifact_path="model")
        print(f"Model saved to {model_path}")

        # Fetch run parameters to build final telemetry links
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id

        # Dynamically verify if a public Ngrok tunnel mapping is active for port 5001
        try:
            from pyngrok import ngrok
            tunnels = ngrok.get_tunnels()
            base_link = next((t.public_url for t in tunnels if ":5001" in t.config.get("addr", "")), tracking_url)
        except Exception:
            base_link = tracking_url

        print("\nProduction Model Successfully Trained & Serialized!\n")
        print("===================== PERFORMANCE REPORT =====================")
        print(f"[Train] Accuracy : {train_acc:.4f} | Precision : {train_prec:.4f} | Recall : {train_recall:.4f} | F1 : {train_f1:.4f} | ROC-AUC : {train_auc:.4f}")
        print(f"[Test]  Accuracy : {test_acc:.4f}  | Precision : {test_prec:.4f}  | Recall : {test_recall:.4f}  | F1 : {test_f1:.4f}  | ROC-AUC : {test_auc:.4f}")
        print("==============================================================")
        print(f"\n🏃 View run XGB_Production_Run at: {base_link}/#/experiments/{experiment_id}/runs/{run_id}")
        print(f"🧲 View experiment at: {base_link}/#/experiments/{experiment_id}")

if __name__ == "__main__":
    train_production_model()
