# Exploratory Data Analysis and Prediction Models for Employee Churn at an Auto Manufacturing Company
Using a dataset of past employee data (10 features), I performed exploratory data analysis to identify key features and data distrubitions. I used this information to build a prediction model to predict future employee churn using similar data. The Jupyter notebook in this repository contains all analyses, data visualizations, and model-building details. Below is a summary of the model performance, as well as a discussion of next steps to consider for future model tuning.

## Model Performance
I trained a tree-based gradient-boosted machine (XGBoost) to predict employee churn. Below is a summary of the final model performance on the held-out validation set:
| Metric | Score |
| ------ | ----- |
| Precision | 0.973 |
| Recall | 0.914 |
| F1 score | 0.943 |
| Accuracy | 0.982 |

### Confusion Matrix for the held-out validation set:
![confusion matrix](https://raw.githubusercontent.com/seancascarina/EmployeeChurn_PredictionModels/master/images/XGBoost_Model_ConfusionMatrix.png)

### Feature Importances for XGBoost model:
![feature importances](https://raw.githubusercontent.com/seancascarina/EmployeeChurn_PredictionModels/master/images/XGBoost_Model_FeatureImportances.png)

## Additional Model-building Details
The dataset was split into a training set (60%), validation set (20%), and final test set (20%) using sklearn's train_test_split. I trained an XGBClassifier using a small hyperparameter grid search and five-fold cross validation on the training set. Below is the hyperparameter grid, which was chosen to balance hyperparameter searching while limiting training time/resources:
| Hyperparameter | Score |
| ------ | ----- |
| n_estimators | [20, 50, 100, 150] |
| max_depth | [3, 9, 15, 20, None] |
| min_child_weight | [1, 3, 5] |
| learning_rate | [0.1, 0.2, 0.3] |

Model performance was then estimated using the validation set. The model with the highest F1 score was considered the best model. A systematic search for alternative probability thresholds (i.e., other than the default 0.5) was explored. Recall for the model could be improved with an alternative threshold, but there was a disproportionate cost to precision. In other words, each additional false negative that was prevented by an alternative threshold resulted in multiple additional false positives.

## Possibilities for Improvement
High model performance was achieved with limited feature engineering. Further improvements in the model could likely be achieved by including engineered features and performing more extensive model tuning. Candidate features for engineering could include separate categories for high-performance employees, over-worked employees, or low-performance/underworked employees, as both of these groups appear to be associated with an increased risk of leaving the company. Additional model tuning could be done by a more extensive grid search of hyperparameters or a randomized search of hyperparameters over a larger hyperparameter space. Finally, comparison with additional types of ML models could identify one with higher performance. These could be trained on the same training and validation sets. Once a final "champion" model is chosen, this model would be tested on the final held-out test set to estimate model performance on unseen data.
