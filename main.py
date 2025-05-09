# general imports
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pingouin as pg
# modules from sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve

# STEP 0: Loading in the data!!
shoppers = pd.read_csv("./online_shoppers_intention.csv", sep = ",")
# Our dataset contains info on online shopping sessions and whether or not they resulted in some revenue for the business


# EDA: Looking at the data:
# display the first few rows of the dataframe
print('EDA: Looking at the data:')
print('First Five Rows:')
print(shoppers.head())
print('Column data types:')
print(shoppers.dtypes)
# check for missing values
missing_values = shoppers.isnull().sum()
print('Missing values in each column:')
print(missing_values[missing_values > 0])
if missing_values.sum() == 0:
    print("No missing values found! Yahooo!")
    # Yeah we got no missing values, this will make it easier for us
else:
    print("Missing values found... uh oh")
# check for strange values like na, NA, n/a, etc.
strange_values_list = ['None', 'N/A', 'na', 'NA', 'n/a', 'null', 'NULL', ' ']
strange_counts = {}
for col in shoppers.columns:
    if shoppers[col].dtype == 'object':
        for strange_val in strange_values_list:
            count = shoppers[col].astype(str).str.strip().eq(strange_val).sum()
            if count > 0:
                if col not in strange_counts:
                    strange_counts[col] = {}
                strange_counts[col][strange_val] = count
if not strange_counts:
    print("No weird placeholder values found yippeee") 
    # We in fact found no weird placeholder values, this makes it easier for us
else:
    print(f"Found potentially strange values: {strange_counts}")


# EDA: Statistical measures:
# summary statistics for numerical, then categorical features
print('EDA: Statistical measures:')
print(shoppers.describe())
print('Summary statistics for categorical features:')
print(shoppers.describe(include=['object', 'bool']))
# check target variable 'revenue' distribution
print('Distribution of our target variable (revenue):')
revenue_counts = shoppers['Revenue'].value_counts()
revenue_percentage = shoppers['Revenue'].value_counts(normalize=True) * 100
print(revenue_counts)
print(revenue_percentage)
# Our target variable is pretty imbalanced, only 15.47% of sessions resulted in some revenue
# We have to be careful about model evaluation metrics and tradeoffs between precision and recall

# EDA: Visualizations:
# histogram for a numerical feature
print('EDA: Visualizations:')
sns.histplot(data=shoppers, x='PageValues', kde=True)
plt.title('distribution of pagevalues')
plt.show()
# PageValues shows heavy right skew, most sessions have low values

# count plot for a categorical feature
sns.countplot(data=shoppers, x='VisitorType')
plt.title('distribution of visitortype')
plt.show()
# Most visitors are returning visitors, customer retention is important?

# count plot for target variable
sns.countplot(data=shoppers, x='Revenue')
plt.title('distribution of revenue (target)')
plt.show()
# Class imbalance in our target variable

# scatter plot for two numerical features
sns.scatterplot(data=shoppers, x='BounceRates', y='ExitRates', hue='Revenue')
plt.title('bouncerates vs. exitrates by revenue')
plt.show()
# BounceRates has a moderate/high positive correlation with ExitRates

# box plot for a numerical feature grouped by a categorical feature
sns.boxplot(data=shoppers, x='VisitorType', y='PageValues', hue='Revenue')
plt.title('pagevalues by visitortype and revenue')
plt.show()

# correlation heatmap for numerical features
# selecting only numeric types for correlation calculation
numerical_cols_for_corr = shoppers.select_dtypes(include=np.number).columns
correlation_matrix = shoppers[numerical_cols_for_corr].corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('correlation matrix of numerical features')
plt.show()

# EDA: Statistical tests:
# check if target is binary and groups have data
print('EDA: Statistical tests:')
if shoppers['Revenue'].nunique() == 2 and True in shoppers['Revenue'].values and False in shoppers['Revenue'].values:
    revenue_true_bounce = shoppers[shoppers['Revenue'] == True]['BounceRates']
    revenue_false_bounce = shoppers[shoppers['Revenue'] == False]['BounceRates']
    if len(revenue_true_bounce) > 1 and len(revenue_false_bounce) > 1 and revenue_true_bounce.var() > 0 and revenue_false_bounce.var() > 0:
        ttest_result_bounce = pg.ttest(revenue_true_bounce, revenue_false_bounce, correction=True)
        print('t-test result for bounce rates between revenue groups:')
        print(ttest_result_bounce)
# Significant difference in bounce rates between revenue groups (p < 0.001)

# ANOVA for page values across visitor types
anova_data_pv = shoppers[['PageValues', 'VisitorType']].dropna()
if anova_data_pv['VisitorType'].nunique() > 1 and len(anova_data_pv) > anova_data_pv['VisitorType'].nunique():
    anova_result_pagevalues = pg.anova(dv='PageValues', between='VisitorType', data=anova_data_pv)
    print('anova result for pagevalues across visitor types:')
    print(anova_result_pagevalues)
# Significant difference in PageValues across visitor types (p < 1e-39)

# ANOVA for administrative duration across visitor types
anova_data_ad = shoppers[['Administrative_Duration', 'VisitorType']].dropna()
if anova_data_ad['VisitorType'].nunique() > 1 and len(anova_data_ad) > anova_data_ad['VisitorType'].nunique():
    anova_result_admin_dur = pg.anova(dv='Administrative_Duration', between='VisitorType', data=anova_data_ad)
    print('anova result for administrative duration across visitor types:')
    print(anova_result_admin_dur)
# Significant relationship between visitor type and administrative duration (p = 0.014)

# DATA CLEANING & WRANGLING:
print('DATA CLEANING & WRANGLING:')
# identify categorical and numerical features
# bool columns often treated as categorical for one-hot encoding
categorical_features = shoppers.select_dtypes(include=['object', 'bool']).columns.tolist()
numerical_features = shoppers.select_dtypes(include=np.number).columns.tolist()

# define target variable
target = 'Revenue'
# remove target variable 'revenue' from feature lists if present
if target in categorical_features:
    categorical_features.remove(target)
if target in numerical_features:
    numerical_features.remove(target)
if 'Weekend' in shoppers.columns and shoppers['Weekend'].dtype == 'bool' and 'Weekend' not in categorical_features:
    if 'Weekend' in numerical_features:
        numerical_features.remove('Weekend')
    categorical_features.append('Weekend')
print(f"Categorical features for encoding: {categorical_features}")
print(f"Numerical features for scaling: {numerical_features}")
# We have a lot of observations, which is super great

# define feature set x and target y
features = numerical_features + categorical_features
x = shoppers[features]
y = shoppers[target]
# convert boolean target to integer
if y.dtype == 'bool':
    y = y.astype(int)
    print(f"converted target '{target}' from boolean to integer.")
print(f"Shape of features (x): {x.shape}")
print(f"Shape of target (y): {y.shape}")


# FEATURE ENGINEERING/PREPROCESSING
# pipeline for numerical features:
print('FEATURE ENGINEERING/PREPROCESSING:')
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
# pipeline for categorical features:
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# create a preprocessor object using columntransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')


# SPLITTING THE DATA (70/30)
print('SPLITTING THE DATA (70%/30%)')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=777, stratify=y)
print(f"Training set shape: x={x_train.shape}, y={y_train.shape}")
print(f"Testing set shape: x={x_test.shape}, y={y_test.shape}")
# A 70/30 split gives a good amount of data for both training and testing

# MODEL SELECTION
print('MODEL SELECTION:')
# define models
# logistic regression
model_lr = LogisticRegression(solver='liblinear', random_state=777, max_iter=1000)
# random forest classifier
model_rf = RandomForestClassifier(random_state=777)
# baseline
model_null = DummyClassifier(strategy='most_frequent', random_state=42)

# null model training and evaluation
print("Evaluating null model...")
model_null.fit(x_train, y_train)
y_pred_null = model_null.predict(x_test)
y_prob_null = model_null.predict_proba(x_test)[:, 1]
print("Null model evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_null)}")
print(f"Precision: {precision_score(y_test, y_pred_null, zero_division=0)}")
print(f"Recall: {recall_score(y_test, y_pred_null, zero_division=0)}")
print(f"F1-score: {f1_score(y_test, y_pred_null, zero_division=0)}")
if len(np.unique(y_test)) > 1:
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_null)}")
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred_null))
print("Classification report:")
print(classification_report(y_test, y_pred_null, zero_division=0, target_names=['No Revenue', 'Revenue']))
# Baseline model achieves 84.5% accuracy by simply predicting majority class, but it captures 0% of sessions that gave us revenue, so this is useless to us and our objective

# logistic regression training and evaluation with hyperparameter tuning
print("Evaluating logistic regression...")
pipeline_lr = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model_lr)])
param_grid_lr = {
    'classifier__C': [0.01, 0.1, 1, 10, 100], 
    'classifier__penalty': ['l1', 'l2']
}
cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)

# gridsearchcv for logistic regression
grid_search_lr = GridSearchCV(pipeline_lr, param_grid=param_grid_lr, cv=cv_stratified, scoring='roc_auc', n_jobs=-1)
grid_search_lr.fit(x_train, y_train)
print("Best parameters for logistic regression:", grid_search_lr.best_params_)
print(f"Best cross-validation ROC-AUC score: {grid_search_lr.best_score_}")

# evaluate the best logistic regression model on the test set
best_lr_model = grid_search_lr.best_estimator_
y_pred_lr = best_lr_model.predict(x_test)
y_prob_lr = best_lr_model.predict_proba(x_test)[:, 1]
print("Logistic regression test set evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr)}")
print(f"Precision: {precision_score(y_test, y_pred_lr, zero_division=0)}")
print(f"Recall: {recall_score(y_test, y_pred_lr, zero_division=0)}")
print(f"F1-score: {f1_score(y_test, y_pred_lr, zero_division=0)}")
if len(np.unique(y_test)) > 1:
    print(f"roc auc: {roc_auc_score(y_test, y_prob_lr)}")
print("confusion matrix:") # source: [310, 391]
print(confusion_matrix(y_test, y_pred_lr))
print("classification report:")
print(classification_report(y_test, y_pred_lr, zero_division=0, target_names=['No Revenue', 'Revenue']))
# Logistic regression has high precision (75%) but low recall (37%)
# it could be missing potential revenue generating customers for us

# random forest training and evaluation with hyperparameter tuning
print("Evaluating random forest...")
pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model_rf)])

# define hyperparameter grid for random forest
param_grid_rf = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 5, 10]
}

# using gridsearchcv, could also use randomizedsearchcv for larger grids
grid_search_rf = GridSearchCV(pipeline_rf, param_grid=param_grid_rf, cv=cv_stratified, scoring='roc_auc', n_jobs=-1, verbose=0)
grid_search_rf.fit(x_train, y_train)
print("Best parameters for random forest:", grid_search_rf.best_params_)
print(f"Best cross-validation ROC-AUC score: {grid_search_rf.best_score_}")

# evaluate the best random forest model on the test set
best_rf_model = grid_search_rf.best_estimator_
y_pred_rf = best_rf_model.predict(x_test)
y_prob_rf = best_rf_model.predict_proba(x_test)[:, 1]
print("Random forest test set evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(f"Precision: {precision_score(y_test, y_pred_rf, zero_division=0)}")
print(f"Recall: {recall_score(y_test, y_pred_rf, zero_division=0)}")
print(f"F1-score: {f1_score(y_test, y_pred_rf, zero_division=0)}")

# calculate roc auc only if both classes are present in y_test
if len(np.unique(y_test)) > 1:
    print(f"roc auc: {roc_auc_score(y_test, y_prob_rf)}")
print("Confusion matrix:") # source: [310, 391]
print(confusion_matrix(y_test, y_pred_rf))
print("Classification report:")
print(classification_report(y_test, y_pred_rf, zero_division=0, target_names=['No Revenue', 'Revenue']))
# has great accuracy at 90.67%, and significant improved recall at 55.76%. F1-score also increases from 0.50 to 0.65, so it's overall better for predicting class


# MODEL EVALUATION & COMPARISON
print("MODEL EVALUATION & COMPARISON:")
if len(np.unique(y_test)) > 1:
    print(f"Null model ROC-AUC: {roc_auc_score(y_test, y_prob_null)}")
    print(f"Logistic regression ROC-AUC: {roc_auc_score(y_test, y_prob_lr)}")
    print(f"Random forest ROC-AUC: {roc_auc_score(y_test, y_prob_rf)}")
# Our Random Forest Classifier achieved the highest ROC-AUC of 0.94, significantly outperforming other models

    # plot roc curves
    fpr_null, tpr_null, _ = roc_curve(y_test, y_prob_null)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

    plt.plot(fpr_null, tpr_null, label=f'null model (auc = {roc_auc_score(y_test, y_prob_null)})', linestyle=':')
    plt.plot(fpr_lr, tpr_lr, label=f'logistic regression (auc = {roc_auc_score(y_test, y_prob_lr)})')
    plt.plot(fpr_rf, tpr_rf, label=f'random forest (auc = {roc_auc_score(y_test, y_prob_rf)})')

    plt.plot([0, 1], [0, 1], 'k--', label='random chance') # diagonal line
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('roc curves')
    plt.legend()
    plt.show()

# generate model insight
print("Generate model insight:")
# feature importance from the best random forest model
preprocessor_fitted = best_rf_model.named_steps['preprocessor']
onehot_encoder_fitted = preprocessor_fitted.named_transformers_['cat'].named_steps['onehot']
# get feature names after one-hot encoding
ohe_feature_names = onehot_encoder_fitted.get_feature_names_out(categorical_features)

# combine numerical and encoded categorical feature names
# ensure correct order as used by the columntransformer
all_feature_names = list(numerical_features) + list(ohe_feature_names)

importances = best_rf_model.named_steps['classifier'].feature_importances_
if len(all_feature_names) == len(importances):
    feature_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    print("top 10 important features from random forest:")
    print(feature_importance_df.head(10))
    # Page values is, quite possibly the most important feature at 53%, while exit rates, product related browsing, and administrative duration also contribute significantly

    # plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15))
    plt.title('top 15 feature importances (random forest)')
    plt.tight_layout()
    plt.show()
    # Concluding insights: Focus on optimizing these high value pages and developing strategies to guide customners to these pages in order to maximize profit