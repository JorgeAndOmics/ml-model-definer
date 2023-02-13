import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_data(file_path):
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)

def perform_feature_engineering(data):
    """Performs feature engineering on the input data."""
    data['feature_1_squared'] = data['feature_1'] ** 2
    data['feature_2_times_3'] = data['feature_2'] * data['feature_3']
    return data

def split_data(data, target_column, test_size, random_state):
    """
    Splits the input data into training and testing sets.
    :param data: pandas DataFrame, input data
    :param target_column: str, the name of the target column in the dataset
    :param test_size: float, the proportion of the dataset to use as the test set
    :param random_state: int, the random seed for reproducibility
    :return: tuple of four pandas DataFrames, X_train, X_test, y_train, y_test
    """
    feature_columns = [col for col in data.columns if col != target_column]
    X_train, X_test, y_train, y_test = train_test_split(data[feature_columns], data[target_column], test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def preprocess_data(X_train, X_test):
    """Preprocesses the input data by scaling it using a StandardScaler."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def evaluate_models(X_train, X_test, y_train, y_test, models):
    """
    Evaluates a list of models using hyperparameter tuning with a grid search.
    :param X_train: pandas DataFrame, input data for training
    :param X_test: pandas DataFrame, input data for testing
    :param y_train: pandas DataFrame, target data for training
    :param y_test: pandas DataFrame, target data for testing
    :param models: dict, a dictionary of models to be evaluated, where keys are model names and values are tuples of (model, param_grid)
    :return: None
    """
    for name, (model, param_grid) in models.items():
        grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)
        grid_search.fit(X_train, y_train)
        y_pred = grid_search.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'{name} MSE: {mse:.2f}')
        print(f'Best parameters: {grid_search.best_params_}')
        print(f'Best score: {grid_search.best_score_:.2f}')

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Run machine learning on a dataset.')
    parser.add_argument('file_path', type=str, help='the file path to the CSV file containing the dataset')
    parser.add_argument('target_column', type=str, help='the name of the target column in the dataset')
    parser.add_argument('--test_size', type=float, default=0.2, help='the proportion of the dataset to use as the test set')
    parser.add_argument('--random_state', type=int, default=42, help='the random seed for reproducibility')
    return parser.parse_args()

def main():
    args = get_args()

    data = load_data(args.file_path)
    data = perform_feature_engineering(data)
    X_train, X_test, y_train, y_test = split_data(data, args.target_column, args.test_size, args.random_state)
    X_train, X_test = preprocess_data(X_train, X_test)
    
    models = {
        'Linear Regression': (LinearRegression(), {}),
        'Decision Tree': (DecisionTreeRegressor(random_state=args.random_state), {'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10]}),
        'Random Forest': (RandomForestRegressor(random_state=args.random_state), {'n_estimators': [10, 50, 100], 'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10]})
    }

    evaluate_models(X_train, X_test, y_train, y_test, models)

if __name__ == '__main__':
    main()
