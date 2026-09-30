import pandas as pd
from sklearn.linear_model import LogisticRegression

def train_model():
    # Load data
    df = pd.read_csv("data.csv")

    # Split features and target
    X = df.drop("Converted", axis=1)
    y = df["Converted"]

    # Convert categorical → numerical
    X = pd.get_dummies(X)

    # Train model
    model = LogisticRegression()
    model.fit(X, y)

    return model, X.columns