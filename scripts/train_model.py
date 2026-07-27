import pandas as pd
import mlflow
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def train():
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv').values.ravel()
    
    # Prepare Cat features (simple encoding)
    X_train = pd.get_dummies(X_train)
    
    with mlflow.start_run(run_name="Wellness_Package_Tuning"):
        rf = RandomForestClassifier()
        param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
        
        grid = GridSearchCV(rf, param_grid, cv=3)
        grid.fit(X_train, y_train)
        
        # Log Best Hyperparameters
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("best_cv_score", grid.best_score_)
        
        # Save Model
        if not os.path.exists('models'): os.makedirs('models')
        with open('models/best_model.pkl', 'wb') as f:
            pickle.dump(grid.best_estimator_, f)

if __name__ == "__main__":
    train()
