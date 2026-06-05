import sys
import io
# Set stdout encoding to UTF-8 to handle MLflow emoji printing on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import mlflow

def main():
    # Set local tracking URI explicitly to ensure runs are logged in the MLProject/mlruns directory
    # Only do this if not running within an MLflow Project run to avoid tracking conflicts
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if "MLFLOW_RUN_ID" not in os.environ:
        mlruns_path = os.path.join(current_dir, "mlruns")
        mlflow.set_tracking_uri(f"file:///{mlruns_path}".replace("\\", "/"))
        mlflow.set_experiment("Heart_Disease_Status_Basic")
    
    # Enable autolog
    mlflow.autolog()
    
    # Load dataset
    dataset_name = "heart_disease_preprocessing.csv"
    dataset_path = os.path.join(current_dir, dataset_name)
    
    print(f"Loading preprocessed dataset from: {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    # Separate features and target
    target_col = "Heart Disease Status"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Train-test split (stratify to preserve class imbalance ratios)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print("Training RandomForestClassifier (without hyperparameter tuning)...")
    with mlflow.start_run() as run:
        # Standard Random Forest model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Explicitly log the model to resolve autologging version warning/skipping issues
        print("Logging model explicitly to artifacts/model...")
        mlflow.sklearn.log_model(model, "model")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print("\nModel Evaluation:")
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(report)
        
        print(f"Basic run successfully tracked. Run ID: {run.info.run_id}")
        
        # Save model URI for CI/CD workflow use
        try:
            model_uri = f"runs:/{run.info.run_id}/model"
            print(f"Model URI: {model_uri}")
            
            # Resolve target directory: use GITHUB_WORKSPACE in CI environment, otherwise use parent directory
            if "GITHUB_WORKSPACE" in os.environ:
                target_dir = os.environ["GITHUB_WORKSPACE"]
            else:
                target_dir = os.path.dirname(current_dir)
                
            with open(os.path.join(target_dir, "latest_model_uri.txt"), "w") as f:
                f.write(model_uri)
            print(f"Successfully saved model URI to: {os.path.join(target_dir, 'latest_model_uri.txt')}")
        except Exception as e:
            print(f"Warning: Could not save model URI: {e}")

if __name__ == "__main__":
    main()

