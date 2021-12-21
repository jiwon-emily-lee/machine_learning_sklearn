# machine_learning_sklearn

This project aims to find the best Machine Learning model to fit the training data points using Scikit Learn. Olivetti faces dataset, which contains face images, is used as a dataset.

#### Build Status: ![Build Status](https://shields.io/badge/build-passing-brightgreen)

#### Code Style: Python, followed [PEP8](https://www.python.org/dev/peps/pep-0008/)

## Visuals
![screen shot](https://github.com/jiwon-emily-lee/notion-todo/blob/main/Screen%20Shot%202021-12-21%20at%2011.06.54%20AM.png?raw=true)

## Tech/Framework used & Features

This project used **Logistic Regression** which fits the training data points best.

Hyperparameter Changes:
+ C=100
+ solver='liblinear'
+ class_weight='balanced'
+ random_state=0

Here's a description of logistic regression and hyperparameters: [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

## Installation
Installing Scikit Learn...
```pip
pip install -U scikit-learn
```
Installing Numpy...
```pip
pip install numpy
```

You should import...
```python
import sklearn.datasets
import sklearn.linear_model
import sklearn.tree
import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt 
```

## Usage with Code Example 
* ####  Load Packages

    Import sklearn, numpy and matplotlib
   
* ####  Load Data Points

    Olivetti faces in sklearn.datasets
 
* ####  Classification with Scikit Learn Library
  1. Create a classification object in scikit learn package (Find the best model and hyperparameter for face recognition)
  2. Fit the object to training dataset (X_train, y_train)
  3. Predict the label of test data point (X_test)

```python
log_reg = sklearn.linear_model.LogisticRegression(C=100, solver='liblinear', class_weight='balanced', random_state=0)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
```

## Tests
![accuracy rate screenshot](https://github.com/jiwon-emily-lee/notion-todo/blob/main/acc.png?raw=true)

---

## Contributing

To contribute, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

## Cotact

If you want to contact me you can reach me at emillie0416@gmail.com

## License
This project uses the following license: [MIT](https://choosealicense.com/licenses/mit/)
