import matplotlib.pyplot as plt
import pandas as pd
import skops.io as sio
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression

class ChurnModelPipeline:
    def __init__(self, data_path, target_column, index_col=None, random_state=125):
        self.data_path = data_path
        self.target_column = target_column
        self.index_col = index_col
        self.random_state = random_state
        self.model_pipeline = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.preprocessor = None

    def load_and_prepare_data(self, drop_columns=None, nrows=None):
        print('Loading data....')
        df = pd.read_csv(self.data_path, index_col=self.index_col, nrows=nrows)
        if drop_columns:
            df = df.drop(drop_columns, axis=1)
        df = df.sample(frac=1)
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        print('Data loaded and split successfully.')
    def build_preprocessor(self, cat_cols, num_cols):
        print('building preprocessing pipeline...')
        numerical_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="mean")),
                   ("scaler", StandardScaler())]
        )
        categorical_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                   ("encoder", OrdinalEncoder())]
        )
        self.preprocessor = ColumnTransformer(
            transformers = [
                ("num", numerical_transformer, num_cols),
                ("cat", categorical_transformer, cat_cols),
            ],
            remainder="passthrough"
        )
        print("Preprocessing pipeline built.")

    def build_model_pipeline(self, k_best=5):
        print('Building model pipeline...')
        feature_selector = SelectFromModel(LogisticRegression(max_iter=1000))
        model = GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)

        train_pipeline = Pipeline(steps=[("feature_selection", feature_selector), ("GBmodel", model)])

        self.model_pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("train", train_pipeline)
            ]
        )
        print('Model pipeline built.')
    def train_model(self):
        if self.model_pipeline is None:
            raise ValueError("Model pipeline is not initialized. Build the model pipeline first")

        print('Training the model....')
        self.model_pipeline.fit(self.X_train, self.y_train)
        print("Model training completed.")

    def evaluate_model(self):
        print('Evaluating the model...')
        predictions = self.model_pipeline.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions, average="macro")
        print(f'Accuracy: {round(accuracy*100, 2)}%, F1 Score: {round(f1, 2)}')
        return accuracy, f1

    def plot_confusion_matrix(self):
        print("Plotting confusion matrix...")
        prediction = self.model_pipeline.predict(self.X_test)
        cm = confusion_matrix(self.y_test, prediction, labels=self.model_pipeline.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model_pipeline.classes_)
        disp.plot()
        plt.savefig("model_result.png", dpi=120)
        print("Confusion matrix saved as model_results.png.")

    def save_metrics(self, accuracy, f1):
        print("Saving metrics to file...")
        with open("metrics.txt", "w") as outfile:
            outfile.write(f"Accuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}\n")
        print("Metrics saved in 'metrics.txt'.")

    def save_pipeline(self):
        print("Saving pipeline to file...")
        sio.dump(self.model_pipeline, "Churn_pipeline.skops")
        print("Pipeline saved as 'churn_prediction.skops'.")
if __name__ == "__main__":
    # Configuration
    data_file = "Churn_modelling.csv"
    target_col = "Exited"
    drop_cols = ["RowNumber", "CustomerId", "Surname"]

    cat_columns = [1, 2] # indices after dropping columns
    num_columns = [0, 3, 4, 5, 6, 7, 8, 9]

    # Initialize and build pipeline
    churn_pipeline = ChurnModelPipeline(data_file, target_col)
    churn_pipeline.load_and_prepare_data(drop_columns=drop_cols, nrows=1000)
    churn_pipeline.build_preprocessor(cat_cols=cat_columns, num_cols=num_columns)
    churn_pipeline.build_model_pipeline()

    # Train and evaluate model
    churn_pipeline.train_model()
    accuracy, f1 = churn_pipeline.evaluate_model()

    # plot CM and save the metrics
    churn_pipeline.plot_confusion_matrix()
    churn_pipeline.save_metrics(accuracy, f1)

    # Save the pipeline
    churn_pipeline.save_pipeline()