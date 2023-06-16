# SKGA Repository

Welcome to the SKGA repository! This repository contains the `skga` package, which is a Python library for performing machine learning hyperparameters optimization. 

## Installation

To install the `skga` package, follow these steps:

1. Make sure you have Python 3.9 installed on your system.
2. Open a terminal or command prompt.
3. Run the following command to install the package using pip:

   ```bash
   pip install skga
   ```

   This will automatically download and install the `skga` package along with its dependencies.

## Basic Usage

Once you have installed the `skga` package, you can start using it in your Python projects. Here's a basic example to get you started:

```python
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from skga import HyperBRKGASearchCV

# Load the Iris dataset
irisX, irisY = datasets.load_iris(return_X_y=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(irisX, irisY, test_size=0.5, random_state=0)

# Define the parameters for the SVM classifier
parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}

# Create an SVM classifier
svc = svm.SVC()

# Create a HyperBRKGASearchCV instance
clf = HyperBRKGASearchCV(svc, parameters=parameters, data=X_train, target=y_train)

print("# Optimizing hyperparameters for accuracy\n")

# Fit the classifier to the training data
clf.fit(X_train, y_train)

print("Best combination of parameters found in the training set:\n")
print(clf.cv_results_['best_param_decoded'])
print()
print("Scores of HyperBRKGA in the training set:\n")
print(clf.cv_results_)
```

This example demonstrates how to use the `HyperBRKGASearchCV` class from the `skga` package to perform hyperparameter optimization for an SVM classifier.

For more detailed examples and use cases, please refer to the [examples](https://github.com/MLRG-CEFET-RJ/skga/tree/main/examples) directory in the SKGA repository.

## Contributing

If you find any issues or have suggestions for improvement, please feel free to contribute to this repository. You can submit bug reports, feature requests, or pull requests through the [GitHub repository](https://github.com/MLRG-CEFET-RJ/skga). We appreciate your contributions!

## License

This project is licensed under the [BSD 3-Clause License](https://github.com/MLRG-CEFET-RJ/skga/blob/main/LICENSE).