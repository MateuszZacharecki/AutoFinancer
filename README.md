![AutoFinancer](https://github.com/user-attachments/assets/bfa92da0-1cb7-440d-8391-a0dcdc4f169d)

## Introduction

This tutorial demonstrates how to use the `AutoFinancer` package for automated machine learning (AutoML).
The package is designed to handle binary classification, multi-class classification and regression tasks.
It offers an end-to-end solution for data preprocessing, model selection, hyperparameter optimization and result evaluation.

This package is intended for:
- **Bank employees**
- **Financial analysts**
- **Employees of financial institutions**
- **Risk management specialists**
- **Data scientists and machine learning engineers** working in the financial sector.

The AutoML package supports three main types of machine learning problems:
1. **Binary Classification**
2. **Multiclass Classification**
3. **Regression**

This specialization makes it suitable for a wide range of practical applications.

### Comparison with similar packages:

Popular tools such as **AutoGluon** and **mljar** provide similar functionalities. However, the `AutoFinancer` package offers several key differentiators:

1. **Customization and Flexibility**:
    - While **AutoGluon** and **mljar** offer high automation levels, `AutoFinancer` provides greater customization options for preprocessing, feature selection, and model optimization.
    - Users can control specific feature selection methods (`random_forest`, `select_k_best`, or both) and choose between `random_search` and `grid_search` for hyperparameter optimization.

2. **Integrated Report Generation**:
    - `AutoFinancer` includes a comprehensive reporting system that generates detailed reports with metrics, confusion matrices, ROC curves, feature importance plots, and model interpretability (e.g., Break Down and Ceteris Paribus plots for regression tasks).
    - In contrast, **AutoGluon** and **mljar** focus more on prediction performance without native support for detailed interpretability reports.

3. **Target Audience and Usability**:
    - While **AutoGluon** is geared toward general-purpose machine learning tasks, `AutoFinancer` specifically caters to financial and banking applications.
    - **mljar**, with its automatic web-based interface, is user-friendly for non-developers but lacks the in-depth control over the training pipeline provided by `AutoFinancer`.

4. **Performance Tuning**:
    - `AutoFinancer` allows for fine-tuning parameters such as cross-validation folds, number of iterations, and correlation thresholds, which gives advanced users the flexibility to balance performance and computational efficiency.
    - **AutoGluon** focuses on stacking and ensembling models for maximum accuracy, which can sometimes result in longer training times.

5. **Problem Type Specialization**:
    - The `AutoFinancer` package supports binary classification, multi-class classification and regression tasks with tailored approaches for each type, ensuring robust handling of a wide range of datasets.

## Parameters

**problem_type** : `str`, default=`'binary_classification'`  
    Type of machine learning task to be performed.  
    Accepted values:  
    - `'binary_classification'`: Two-class classification task.  
    - `'multiclass_classification'`: Multi-class classification task.  
    - `'regression'`: Regression task for continuous target values.


**metric** : `str`, default=None, if none, `'accuracy'` is chosen for classification and `'r2'` for regression  
    Evaluation metric to be used during model selection.  
    Accepted values for classification:  
    - `'roc_auc'`, `'accuracy'`, `'precision'`, `'recall'`, `'f1'`, `'balanced_accuracy'`  
    Accepted values for regression:  
    - `'r2'`, `'neg_mean_squared_error'`, `'neg_mean_absolute_error'`


**random_state** : `int`, default=`2024`  
    Random seed used for reproducibility of results.  
    Accepted values: Any integer.


**k** : `int`, default=`10`  
    Number of features to select using the `SelectKBest` method.  
    Accepted values: Positive integer.


**correlation_threshold** : `float`, default=`0.9`  
    Threshold for removing highly correlated features.  
    Features with correlation higher than this value are excluded.  
    Accepted values: A float in the range `[0.0, 1.0]`.


**selection_method** : `str`, default=`'both'`  
    Method to use for feature selection.  
    Accepted values:  
    - `'random_forest'`: Feature selection based on Random Forest importance.  
    - `'select_k_best'`: Statistical feature selection using `SelectKBest`.  
    - `'both'`: Combines both methods.


**method** : `str`, default=`'random_search'`  
    Hyperparameter optimization strategy.  
    Accepted values:  
    - `'random_search'`: Randomized search over hyperparameter space.  
    - `'grid_search'`: Exhaustive search over specified parameter grid.


**cv_folds** : `int`, default=`5`  
    Number of cross-validation folds.  
    Accepted values: Integer greater than 1.


**n_iter** : `int`, default=`10`  
    Number of iterations for randomized search.  
    Accepted values: Positive integer.

## Components

The package is an automated machine learning system with the following main components:

### 1. Automated Model Training and Optimization (`train` method)
- **Purpose**: Automates the training process, including data preprocessing, model selection, and optimization.
- **Parameters**:
 - `X` (array/DataFrame): Input data;
 - `y` (array/Series): Target variable.
- **Key Steps**:
  1. **Data Preprocessing**: Prepares the input data (`X`) and target variable (`y`) for model training. This includes handling missing values, feature scaling, and encoding categorical variables.
  2. **Model Selection and Optimization**: Selects the best model from a pool of candidates and tunes its hyperparameters, using methods like `RandomizedSearchCV` and `GridSearchCV` (optional).
  3. **Report Generation**: Produces detailed performance reports summarizing the training and optimization process.
- **Attributes**:
 - `summary`: table of basic descriptive statistics for numerical variables,
 - `unique_values`: list of labels in target variable,
 - `class_counts`: only for `problem_type = 'binary_classification'` and `problem_type = 'multiclass_classification'`, list of labels in target variable,
 - `final_features`: list of features selected during feature selection process,
 - `results`: list of results received during training process concluding name of model, chosen parameters, evaluation score, metric used for evaluation, time of fitting,
 - `scores[model_name]`: table of scores achieved during training process for a chosen metric,
 - `optimizers[model_name]`: visualization of a chosen model with its parameters,
 - `all_results`: table of all best models, their chosen parameters, scores, evaluation metrics and training time.

---

### 2. Prediction (`predict` method)
- **Purpose**: Provides predictions for new data using the selected or optimized models.
- **Parameters**:
 - `X` (DataFrame): Input data;
 - `model_name` (str): Name of model, if None, best model on training data is used.
- **Returns**:
 - `y_pred` (Series): Vector of predicted classes.
- **Key Features**:
  - Accepts an optional `model_name` parameter to specify which model to use. If none is provided, the best model from training is used.
  - Preprocesses test data to ensure compatibility with the trained model.
  - Outputs predictions as a series.

---

### 3. Probability Prediction (`predict_proba` method)
- **Purpose**: Computes and returns class probabilities for classification problems.
- **Parameters**:
 - `X` (array/DataFrame): Input data;
 - `model_name` (str): Name of model, if None, best model on training data is used.
- **Returns**:
 - `y_pred` (Series): Vector of predicted probabilities.
- **Key Features**:
  - Applicable only to classification tasks (`binary_classification` or `multiclass_classification`).
  - Preprocesses input data and allows specifying a model via the `model_name` parameter. If none is provided, the best model from training is used.
  - Outputs probabilities for each class as a series.

---

### 4. Prediction and Report Generation (`predict_and_report` method)
- **Purpose**: Combines prediction, probability prediction (for classification), and report generation for a test dataset.
- **Parameters**:
 - `X` (array/DataFrame): Test data;
 - `y` (array/Series): Target for test data;
 - `model_name` (str): Name of model, if None, best model on training data is used;
 - `prediction_threshold` (float): Threshold used for confusion matrices (only for binary classification).
- **Key Features**:
  1. Preprocesses both test input data (`X`) and target labels (`y`).
  2. Computes predictions and probabilities (if applicable).
  3. Generates detailed reports:
     - **Binary Classification**: Includes performance metrics such as accuracy, precision, recall, f1, balanced accuracy, AUC, visualization of ROC curve and confusion matrix with an optional `prediction_threshold`.
     - **Multiclass Classification**: Summarizes class-level performance metrics such as accuracy, precision, recall, f1, balanced accuracy, AUC, visualization of ROC curve and confusion matrix.
     - **Regression**: Reports regression metrics such as R², MSE, MAE.

### Preprocessing

The package includes comprehensive preprocessing functions to handle missing data, normalize features, and encode categorical variables.
This section describes the key methods responsible for preparing the data before model training.

#### Key Methods:
1. **Converting target variable `y`** to Series type.

2. **Converting input data `X`** to DataFrame type.

3. **Handling missing values in `y`**: rows with missing values in `y` are removed before proceeding with further preprocessing

4. **Processing target variable (`y`)** based on the problem type:
   - For **binary classification**, it maps the target to 0 and 1.
   - For **multiclass classification**, it uses `LabelEncoder` to encode the target into numerical labels.
   - For **regression**, it ensures that the target variable is converted to a numeric type.

5. Identifying **binary**, **numerical**, **categorical**, and **datetime** columns.

6. **Handling missing data** is a critical part of preprocessing, especially when dealing with real-world datasets.
The package uses different strategies depending on the type of the variable:

  - **Symmetric numerical columns**: Missing values are imputed using the mean.
  - **Skewed numerical columns**: Missing values are imputed using the median.
  - **Categorical columns**: Missing values are imputed using the most frequent category (mode).
  - **Binary columns**: Missing values are imputed using iterative imputation with a `RandomForestRegressor` to maintain correlations with other variables.

7. **Standardizing numerical features**: Numerical features are standardized using `StandardScaler`, which ensures that the mean is 0 and the variance is 1.

8. **Encoding categorical variables** using `LabelEncoder`. This ensures that the model can interpret these variables correctly by converting them into numerical values.

9. **Feature selection**: Feature selection is a critical step in machine learning workflows. It helps improve model performance by reducing overfitting, speeding up training, and improving model interpretability. The `AutoFinancer` package incorporates multiple feature selection techniques to identify the most relevant features. After feature selection, the package stores the selected features in the `self.final_features` attribute, and these features are used in subsequent model training and evaluation steps.

 - **Correlation-based Feature Removal**:
   Highly correlated features can introduce multicollinearity, negatively impacting the model’s performance. The package identifies pairs of highly correlated features and removes one of them if their correlation exceeds a specified threshold (default: 0.9).

 - **Random Forest-based Feature Selection**:
   This method uses the feature importances provided by a `RandomForestClassifier` to rank and select the top-k features. Random Forest is effective at capturing complex feature interactions.

 - **Statistical Feature Selection with SelectKBest**:
   This method uses statistical tests, specifically ANOVA F-tests (`f_classif`), to rank and select the top-k features based on their relationship with the target variable. It’s particularly useful for identifying features that have a significant linear correlation with the target.

 - **Combined Method**:
   The combined method merges the results from both the Random Forest-based and statistical selection methods. Only the features selected by both approaches are retained for the final model training, ensuring robust feature selection.

### Model Selection and Optimization

The `AutoFinancer` package supports automatic model selection and hyperparameter tuning using two primary methods:
- **Random search**
- **Grid search** (optional)

These methods allow efficient exploration of hyperparameter spaces, leading to better model performance.

#### **Random Search vs. Grid Search**

**Random Search**:  
- Randomly samples hyperparameters from specified distributions.  
- **Default Parameter**: `n_iter=10` (number of iterations).  
- Suitable for large hyperparameter spaces or when computational resources are limited.

**Grid Search**:  
- Exhaustively searches through a manually specified subset of hyperparameters.  
- A predefined grid of values for each model.  
- Best used for smaller hyperparameter spaces or when computational time is not a constraint.


#### **Cross-Validation Strategy**

The package uses **k-fold cross-validation** for model evaluation during hyperparameter optimization.
- **Default Parameter**: `cv_folds=5` (number of folds in cross-validation).  
This approach ensures that the model is tested on different subsets of data, providing a robust estimate of its performance.


#### **Models and Their Hyperparameters**

The following models are supported by the package, along with their hyperparameters that are tuned during optimization process:

| Model                  | Tuned Hyperparameters                                         |
|------------------------|-----------------------------------------------------------------|
| **RandomForest**       | `n_estimators`, `max_depth`, `min_samples_split`     |
| **DecisionTree**       | `max_depth`, `min_samples_split`                         |
| **XGBoost**            | `n_estimators`, `max_depth`, `learning_rate`          |
| **GradientBoosting**   | `n_estimators`, `max_depth`, `learning_rate`          |
| **LogisticRegression** | `penalty=None`, `solver='saga'`, `max_iter=500`                |
| **LassoRegression** | `penalty=l1`, `solver='saga'`, `max_iter=500`, `C`                |
| **RidgeRegression** | `penalty=l1`, `solver='saga'`, `max_iter=500`, `C`                |
| **LDA** | `store_covariance=True`, `solver`                |
| **QDA** | `store_covariance=True`, `reg_param`                |
| **LinearRegression**   | Standard linear regression without hyperparameters              |



#### **Evaluation Metrics**

The package automatically selects appropriate metrics based on the problem type:

**Classification Metrics**:
- **Default Metric**: `accuracy`  
- Other Supported Metrics:  
  - `roc_auc`  
  - `precision`  
  - `recall`  
  - `f1`  
  - `balanced_accuracy`

**Regression Metrics**:
- **Default Metric**: `r2`  
- Other Supported Metrics:  
  - `neg_mean_squared_error`  
  - `neg_mean_absolute_error`

These metrics ensure comprehensive evaluation of model performance for both classification and regression tasks.

### Postprocessing Functions and Report Generation

After model training, the package generates a comprehensive report that includes:

1. Summary with basic results for the best model.

2. A tabular summary showing the model names, best parameters, scores, evaluation metrics, and training times.

3. Overall descriptive statistics of the processed dataset for numerical variables.

4. Histogram for target variable `y` with its distribution (only for regression task).

5. Classes breakdown for target variable `y` (only for classification task).

6. **Permutation Importance** – Shows the relative importance of each feature by measuring the drop in model performance when the feature's values are permuted.

7. **Partial Dependence Profiles (PDP)** – Displays how a feature affects the predicted outcome, averaged over all instances.

8. **Accumulated Local Effects (ALE)** – Similar to PDP but accounts for feature interactions more robustly.

9. **Break Down plots** – Explains individual predictions by showing how each feature contributes to the final prediction (for observations with the largest and smallest prediction difference from actual values for the best model for regression task).

10. **Ceteris Paribus plots** – Displays how changing one feature affects the prediction while keeping other features constant (for observations with the largest and smallest prediction difference from actual values for the best model for regression task).

11. Reports based on specifics of each model:
 - **DecisionTree** - Scores for decision tree, Textual representation of the tree, Decision tree visualization, Feature importance plots.
 - **RandomForest and GradientBoosting** - Scores for random forest and gradient boosting, Visualization of the first decision tree, Feature importance plots.
 - **XGBoost** - Scores for XGBoost, Feature importance plots.
 - **LogisticRegression, LassoRegression, RidgeRegression** - Scores for logistic, lasso and ridge regression, Model coefficients Visualization of model coefficients.
 - **LDA** - Scores for linear discriminant analysis, Model coefficients, Visualization of model coefficients, Means and covariance matrix.
 - **QDA** - Scores for quadratic discriminant analysis, Means and covariance matrix.
 - **LinearRegression** - Scores for linear regression, Model coefficients, Visualization of model coefficients.

This approach ensures that users not only receive high-level insights into the model's performance but also have access to in-depth, model-specific analyses, making it easier to interpret and trust the results.

## Innovative Approach

This package incorporates several advanced libraries and methods to enhance its functionality:

- **dalex**: Provides advanced model interpretability, generating insights such as Break Down plots and Ceteris Paribus profiles.
- **Iterative Imputer**: Imputes missing binary values using a machine learning model (`RandomForestRegressor`) to maintain correlations between features.

### Adaptability to Problem Type
The package dynamically adapts its behavior based on the user-specified `problem_type`.
Supported types include:
- **Binary Classification**
- **Multiclass Classification**
- **Regression**

For each problem type, the package automatically:
- Selects suitable models (e.g., `RandomForestClassifier` for classification tasks or `LinearRegression` for regression).
- Applies appropriate scoring metrics (e.g., `accuracy` for classification or `R²` for regression).
- Customizes preprocessing methods (e.g., encoding categorical variables or converting targets based on the problem type).

### Customizable Hyperparameter Optimization
Users have full control over hyperparameter optimization by selecting:
- **Random Search**: Efficient for large parameter spaces with limited computational resources.
- **Grid Search**: Exhaustive for smaller parameter spaces to find the best hyperparameters.

Additionally, users can configure:
- **Number of cross-validation folds (`cv_folds`)**
- **Number of iterations in random search (`n_iter`)**

### Feature Selection Customization
Feature selection is an essential step in building effective models. This package offers:
- **Multiple Feature Selection Methods**:
  - `random_forest`: Selects important features using a Random Forest model.
  - `select_k_best`: Selects the top `k` features based on statistical tests.
  - `both`: Combines the results of both methods.
- **Adjustable Correlation Threshold**:
  - Highly correlated features (above a specified threshold) are automatically removed to reduce multicollinearity.
  - The default correlation threshold is **0.9**, but it can be customized by the user (`correlation_threshold` parameter).

### Advanced Reporting
The package includes comprehensive reporting capabilities, such as:
- **Summary Reports**: Display the performance of all trained models and highlight the best-performing model.
- **Individual Model Reports**:
  - Provide detailed insights into a specific model, including:
    - Model details: Name, best parameters, and performance score.
    - Visualizations:
      - Decision tree plots for tree-based models.
      - Feature coefficients for linear models.
      - Break Down plots and Ceteris Paribus plots for interpretability.
- **Reporting on test dataset**, including:
 - Metric values on a test set,
 - ROC Curve (for classification),
 - Confusion Matrix (for classification).

## Tutorial

The tutorial will guide you through:

1. Loading and exploring data.
2. Creating and configuring the `AutoFinancer` object.
3. Training models and generating a report.
4. Making predictions on new data.
5. Generating report on test dataset.
