import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

data = pd.DataFrame({
    "hours_studied": [1,2,3,4,5,6,7,8,2,5,6,3,7,4,8],
    "attendance": [50,60,65,70,75,80,90,95,55,68,85,62,88,72,91],
    "passed":      [0,0,0,1,1,1,1,1,0,0,1,0,1,1,1]
})

X = data[["hours_studied", "attendance"]]
y = data["passed"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

experiments = [
    {"run_name": "Run_1", "n_estimators": 10, "max_depth": 2},
    {"run_name": "Run_2", "n_estimators": 20, "max_depth": 3},
    {"run_name": "Run_3", "n_estimators": 30, "max_depth": 4},
    {"run_name": "Run_4", "n_estimators": 40, "max_depth": 5},
    {"run_name": "Run_5", "n_estimators": 50, "max_depth": 6},
    {"run_name": "Run_6", "n_estimators": 60, "max_depth": 7},
    {"run_name": "Run_7", "n_estimators": 70, "max_depth": 8},
    {"run_name": "Run_8", "n_estimators": 80, "max_depth": 9},
    {"run_name": "Run_9", "n_estimators": 90, "max_depth": 10},
    {"run_name": "Run_10", "n_estimators": 100, "max_depth": 11}
]

for exp in experiments:
    with mlflow.start_run(run_name=exp["run_name"]):
        model = RandomForestClassifier(
            n_estimators=exp["n_estimators"],
            max_depth=exp["max_depth"],
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_param("n_estimators", exp["n_estimators"])
        mlflow.log_param("max_depth", exp["max_depth"])
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(model, "student_pass_model")

        print("Run Name:", exp["run_name"])
        print("Accuracy:", accuracy)
        print("-" * 30)