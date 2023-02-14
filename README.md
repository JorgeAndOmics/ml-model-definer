# Machine Learning Model Definer

This is a Python script that defines a machine learning pipeline for a regression task. It loads a CSV file, performs feature engineering, splits the data into training and testing sets, preprocesses the data by scaling it, and evaluates several regression models using hyperparameter tuning with a grid search.

## Prerequisites

The script requires Python 3 and the following Python packages:

-   argparse
-   pandas
-   scikit-learn

## Usage

To use the script, run the following command in a terminal:

    python ml-model-definer.py data.csv target_column [--test_size TEST_SIZE] [--random_state RANDOM_STATE] 

Here, `data.csv` is the path to the CSV file containing the dataset, and `target_column` is the name of the target column in the dataset. The optional arguments `--test_size` and `--random_state` specify the proportion of the dataset to use as the test set and the random seed for reproducibility, respectively.

## Workflow

The script performs the following steps:

1.  Load the data from a CSV file using the `load_data` function.
2.  Perform feature engineering on the input data using the `perform_feature_engineering` function.
3.  Split the input data into training and testing sets using the `split_data` function.
4.  Preprocess the input data by scaling it using a StandardScaler using the `preprocess_data` function.
5.  Evaluate several regression models using hyperparameter tuning with a grid search using the `evaluate_models` function.
6.  Identify the best-performing model and print its parameters.

The models to be evaluated are defined in a dictionary in the `main` function, where the keys are model names and the values are tuples of (model, param_grid), where `model` is an instance of a scikit-learn estimator class, and `param_grid` is a dictionary of hyperparameters to be tuned using a grid search.

## Contributions

If you require a feature, identify a bug or just want some info, dont hesitate to contact. All contributions are welcome!

## License

Under MIT License
