import logging
from pathlib import Path
from src.data.read_dataset import get_data
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

project_dir = Path(__file__).resolve().parents[2]


class BaselinePredict:
    """
    Provides a classic baseline for comparison
    """

    def __init__(self, model_name):
        self._model = self.train(model_name)

    def predict(self, row):
        """Predict the binary class of a sentence using a Logistic Regression
        Args:
            sentence (str): DataFrame
        Returns:
            binary class (str): Apartment (class 0) | House (class 1)
        """
        # predict
        return self._model.predict_proba(row)

    def train(self, model) -> str:
        """Train a logistic regression method"""
        try:

            # Get data
            df_train, _, df_test = get_data()

            # Removing label
            train = df_train.drop(["label"], axis=1)

            # Numerical transformations
            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())])
            # Categorical transformations
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))])

            # Numerical features
            numeric_features = train.select_dtypes(include=['int64', 'float64']).columns

            # Categorical features
            categorical_features = train.select_dtypes(include=['object']).columns

            # Convert in a column
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)])

            # Pipeline
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('lr', LogisticRegression())])

            # fit
            pipeline.fit(train, df_train['label'])

            return pipeline

        except Exception:
            logging.error(f'directory or model is invalid or does not exist: {self._model}')
