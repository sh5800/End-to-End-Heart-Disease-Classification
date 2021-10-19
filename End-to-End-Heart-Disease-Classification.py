# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Predicting heart disease using machine learning
# 
# This notebook looks into using various Python-based machine learning and data science libraries in an attempt to build a machine-learning model capable of predicting whether or not someone has heart disease based on their medical attributes.
# 
# We're going to take the following approach:
# 1. Problem definition
# 2. Data
# 3. Evaluation 
# 4. Features
# 5. Modelling
# 6. Experimentation
# 
# ## 1. Problem Definition
# 
# In a statement,
# > Given clinical parameters about a patient, can we predict whether or not they have heart disease?
# 
# ## 2. Data
# 
# The original data came from the Cleavland data from UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/heart+disease
# 
# There is also a version of it available on Kaggle. https://www.kaggle.com/ronitf/heart-disease-uci
# 
# ## 3. Evaluation
# 
# > If we can 95% accuracy at predicting whether or not a patient has heart disease during the proof of concept, we'll pursue the project.
# 
# ## 4. Features
# 
# This is where you'll get different information about each of the features in your data.
# 
# **Data dictionary**
# 
# * age. The age of the patient.
# * sex. The gender of the patient. (1 = male, 0 = female).
# * cp. Type of chest pain. (1 = typical angina, 2 = atypical angina, 3 = non — anginal pain, 4 = asymptotic).
# * trestbps. Resting blood pressure in mmHg.
# * chol. Serum Cholestero in mg/dl.
# * fbs. Fasting Blood Sugar. (1 = fasting blood sugar is more than 120mg/dl, 0 = otherwise).
# * restecg. Resting ElectroCardioGraphic results (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hyperthrophy).
# * thalach. Max heart rate achieved.
# * exang. Exercise induced angina (1 = yes, 0 = no).
# * oldpeak. ST depression induced by exercise relative to rest.
# * slope. Peak exercise ST segment (1 = upsloping, 2 = flat, 3 = downsloping).
# * ca. Number of major vessels (0–3) colored by flourosopy.
# * thal. Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect).
# * num. Diagnosis of heart disease (0 = absence, 1, 2, 3, 4 = present).
# %% [markdown]
# ## Preparing the tools
# 
# We're going to use Pandas, Matplotlib and Numpy for data analysis and manipulation.

# %%
# Import all the tools we need

# Regular EDA(exploratory data analysis) and plotting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# We want our plots to appear inside the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Models from Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Model evaluation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import plot_roc_curve, roc_curve, auc

# %% [markdown]
# ## Load data

# %%
df = pd.read_csv("zero-to-mastery-ml-master/data/heart-disease.csv")
df.shape # (rows, columns)

# %% [markdown]
# ## Data Exploration (exploratory data analysis or EDA)
# 
# The goal here is to  find out more about the data and become a subject matter expert on the dataset you're working with.
# 
# 1. What question(s) are you trying to solve?
# 2. What kind of data do we have and how do we treat different types?
# 3. What's missining from the data and how do you deal with it?
# 4. Where are the outliers and why should you care about them?
# 5. How can you add, change or remove features to get more out of your data?

# %%
df.head()


# %%
df.tail()


# %%
# Let's find out how many of each class there
df["target"].value_counts()


# %%
# Plot using Matplotlib
df["target"].value_counts().plot(kind="bar",color=["salmon", "lightblue"]);


# %%
# Plot using Plotly
fig = px.bar(df["target"].value_counts(),color=["1","0"])
fig.update_xaxes(title_text=" ")
fig.update_yaxes(title_text=" ")
fig.show()


# %%
df.info()


# %%
# Are there any missing values?
df.isna().sum()


# %%
df.describe()

# %% [markdown]
# ### Heart Disease Frequeny according to Sex

# %%
df.sex.value_counts()


# %%
# Compare target column with sex column
pd.crosstab(df.target,df.sex)


# %%
# Create a plot of crosstab using Matplotlib
pd.crosstab(df.target,df.sex).plot(kind="bar",
                                   figsize=(10,6),
                                   color=["salmon","lightblue"]);
plt.title("Heart Disease Frequency for Sex");
plt.xlabel("0 = No Disease, 1= Disease");
plt.ylabel("Amount");
plt.legend(["Female", "Male"]);
plt.xticks(rotation=0);
plt.show();


# %%
# Create a plot of crosstab using Plotly
fig = px.bar(pd.crosstab(df.target,df.sex),barmode="group")
fig.update_xaxes(title_text="0 = No Disease, 1= Disease")
fig.update_yaxes(title_text="Amount")
fig.update_layout(title="Heart Disease Frequency for Sex")
fig.show()

# %% [markdown]
# ### Age vs. Max Heart Rate for Heart Disease

# %%
# Create another fig using Matplotlib
plt.figure(figsize=(10,6))

# Scatter with positive examples
plt.scatter(df.age[df.target==1],
            df.thalach[df.target==1],
            c="salmon");

# Scatter with negative examples
plt.scatter(df.age[df.target==0],
            df.thalach[df.target==0],
            c="lightblue");

# Add some helpful info
plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease", "No Disease"]);


# %%
# Create another fig using Plotly
fig = px.scatter(x=df.age,y=df.thalach,color=df.target)
fig.update_xaxes(title_text="Age")
fig.update_yaxes(title_text="Max Heart Rate")
fig.update_layout(title="Heart Disease in function of Age and Max Heart Rate")
fig.show()


# %%
# Check the distribution of the age column with a histogram using Matplotlib
df.age.plot.hist();


# %%
# Check the distribution of the age column with a histogram using Matplotlib
fig = px.histogram(df.age,nbins=10)
fig.update_xaxes(title_text=" ")
fig.update_yaxes(title_text="Frequency")
fig.show()

# %% [markdown]
# ### Heart Disease Frequency per Chest pain Type
# 
# **cp. Type of chest pain.** 
# * 0 = Typical angina 
# * 1 = Atypical angina 
# * 2 = Non-anginal pain 
# * 3 = Asymptomatic.

# %%
pd.crosstab(df.cp,df.target)


# %%
# Make the crosstab more visual using Matplotlib
pd.crosstab(df.cp,df.target).plot(kind="bar",
                                  figsize=(10,6),
                                  color=["salmon","lightblue"]);

# Add some communication
plt.title("Heart Disease Frequency Per Chest Pain Type");
plt.xlabel("Chest Pain Type");
plt.ylabel("Amount");
plt.legend(["No Disease","Disease"]);
plt.xticks(rotation=0);


# %%
# Make the crosstab more visual using Plotly
fig = px.bar(pd.crosstab(df.cp,df.target),barmode="group")
fig.update_xaxes(title_text="Chest Pain Type")
fig.update_yaxes(title_text="Amount")
fig.update_layout(title="Heart Disease Frequency Per Chest Pain Type")
fig.show()


# %%
# Make a correlation matrix
df.corr()


# %%
# Let's make our correlation matrix a little prettier using Matplotlib
corr_matrix = df.corr()
fig,ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt=".2f",
                 cmap="YlGnBu");


# %%
fig = px.imshow(corr_matrix,color_continuous_scale="YlGnBu")
fig.show()

# %% [markdown]
# ## 5. Modelling

# %%
df.head()


# %%
# split data into X and y
X = df.drop("target",axis=1)
y = df["target"]


# %%
X


# %%
y


# %%
# Split data into train and test sets
np.random.seed(42)

# Split into train & test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)


# %%
X_train


# %%
y_train , len(y_train)

# %% [markdown]
# Now we've got our data split into training and test sets, it's time to build a machine learning model.
# 
# We'll train it (find the patterns) on the training set.
# 
# And we'll test it (use the patterns) on the test set.
# 
# We're going to try 3 different machine learning models:
# 1. Logistic Regression
# 2. K-Nearest Neighbours Classifier
# 3. Random Forest Classifier

# %%
# Put models in a dictionary
models = {"Logistic Regression":LogisticRegression(),
          "KNN":KNeighborsClassifier(),
          "Random Forest":RandomForestClassifier()}

# Create a function to fit and score models
def fit_and_score(models,X_train,X_test,y_train,y_test):
    """
    Fits and evaluates given machine learning models.
    models : a dict of different Scikit-Learn machine learning models
    X_train : training data (no labels)
    X_test : testing data (no labels)
    y_train : training labels
    y_test : test labels
    """
    # Set random seed
    np.random.seed(42)
    # Make a dictionary to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train,y_train)
        # Evaluate the model and append it's score to model_scores
        model_scores[name] = model.score(X_test,y_test)
    return model_scores    


# %%
model_scores = fit_and_score(models=models,
                             X_train=X_train,
                             X_test=X_test,
                             y_train=y_train,
                             y_test=y_test)
model_scores

# %% [markdown]
# ## Model Comparison

# %%
# Using Matplotlib
model_compare = pd.DataFrame(model_scores,index=["Accuracy"])
model_compare.T.plot.bar();
plt.xticks(rotation=0);


# %%
# Using Plotly
fig = px.bar(model_compare.T,barmode="group")
fig.update_xaxes(title_text = " ")
fig.update_yaxes(title_text = " ")
fig.show()

# %% [markdown]
# Now we've got a baseline model... and we know a model's first predictions aren't always what we should based our next steps off. What should we do?
# 
# Let's look at the following:
# * Hyperparameter tuning
# * Feature importance
# * Confusion matrix
# * Cross-validation
# * Precision
# * Recall
# * F1 score
# * Classification report
# * ROC curve
# * Area under the curve (AUC)
# 
# ### Hyperparameter tuning (by hand)

# %%
# Let's tune KNN

train_scores = []
test_scores = []

# Create a list of different values for n_neighbors
neighbors = range(1,21)

# Setup KNN instance
knn = KNeighborsClassifier()

# Loop through different n_neighbors
for i in neighbors:
    knn.set_params(n_neighbors=i)
    
    # Fit the algorithm
    knn.fit(X_train,y_train)
    
    # Update the training scores list
    train_scores.append(knn.score(X_train,y_train))
    
    # Update the test scores list
    test_scores.append(knn.score(X_test,y_test))


# %%
train_scores


# %%
test_scores


# %%
# Using Matplotlib
plt.plot(neighbors,train_scores,label="Train score")
plt.plot(neighbors,test_scores,label="Test score")
plt.xticks(np.arange(1,21,1))
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.legend()

print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")


# %%
# Using Plotly
fig = px.line(x=neighbors,y=[train_scores,test_scores])
fig.update_yaxes(title_text="Model score")
fig.update_xaxes(title_text="Number of neighbors")
fig.show()
print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")

# %% [markdown]
# ## Hyperparameter tuning with RandomizedSearchCV
# 
# We're going to tune:
# * LogisticRegression()
# * RandomForestClassifier()
# 
# ... using RandomizedSearchCV

# %%
# Create a hyperparameter grid for LogisticRegression
log_reg_grid = {"C":np.logspace(-4,4,20),
                "solver":["liblinear"]}

# Create a hyperparameter grid for RandomForestClassifier
rf_grid = {"n_estimators":np.arange(10,1000,50),
           "max_depth":[None,3,5,10],
           "min_samples_split":np.arange(2,20,2),
           "min_samples_leaf":np.arange(1,20,2)}

# %% [markdown]
# Now we've got hyperparameter grids setup for each of our models,let's tune them using RandomizedSearchCV.

# %%
# Tune LogisticRegression

np.random.seed(42)

# Setup rnadom hyperparameter search for LogisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)

# Fit random hyperparameter search model for LogisticRegression
rs_log_reg.fit(X_train,y_train)


# %%
rs_log_reg.best_params_


# %%
rs_log_reg.score(X_test,y_test)

# %% [markdown]
# Now we've tuned LogisticRegression(), let's do the same for RandomForestClassifier()...

# %%
# Setup random seed
np.random.seed(42)

# Setup random hyperparameter search for RandomForestClassifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions=rf_grid,
                           cv=5,
                           n_iter=20,
                           verbose=True)

# Fit random hyperparameter search model for RandomForestClassifier()
rs_rf.fit(X_train,y_train)


# %%
# Find the best hyperparameters
rs_rf.best_params_


# %%
# Evaluate the randomized search RandomForestClassifier model
rs_rf.score(X_test,y_test)


# %%
model_scores

# %% [markdown]
# ## Hyperparameter tuning with GridSearchCV
# 
# Since our LogisticRegression model provides the best scores so far, we'll try to improve them again using GridSearchCV...

# %%
# Different hyperparameters for our LogisticRegression model
log_reg_grid = {"C":np.logspace(-4,4,30),
                "solver":["liblinear"]}

# Setup grid hyperparameter search for LogisticRegression
gs_log_reg = GridSearchCV(LogisticRegression(),
                          param_grid=log_reg_grid,
                          cv=5,
                          verbose=True)

# Fit grid hyperparameter search model
gs_log_reg.fit(X_train,y_train)


# %%
# Check the best hyperparameters
gs_log_reg.best_params_


# %%
# Evaluate the grid search LogisticRegression model
gs_log_reg.score(X_test,y_test)

# %% [markdown]
# ## Evaluating our tuned machine learning classifier, beyond accuracy
# 
# * ROC curve and AUC score
# * Confusion matrix
# * Classification report
# * Precision
# * Recall
# * F1-score
# 
# ...and it would be great if cross-validation was used where possible.
# 
# To make comparisons and evaluate our trained model, first we need to make predictions.

# %%
# Make predictions with tuned model
y_preds = gs_log_reg.predict(X_test)


# %%
y_preds


# %%
y_test


# %%
# Plot ROC curve and calculate AUC metric using matplotlib
plot_roc_curve(gs_log_reg,X_test,y_test);


# %%
# Plot ROC curve and calculate AUC metric using Plotly
y_probs = gs_log_reg.predict_proba(X_test)
y_probs_positive = y_probs[:,1]
fpr,tpr,thresholds = roc_curve(y_test,y_probs_positive)
fig = px.area(x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500)

fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.show()


# %%
# Confusion matrix
print(confusion_matrix(y_test,y_preds))


# %%
# Using Matplotlib
sns.set(font_scale=1.5)

def plot_conf_mat(y_test,y_preds):
    """
    Plots a nice looking confusion matrix using Seaborn's heatmap()
    """
    fig, ax = plt.subplots(figsize=(3,3))
    ax = sns.heatmap(confusion_matrix(y_test,y_preds),
                     annot=True,
                     cbar=False)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    
plot_conf_mat(y_test,y_preds)    


# %%
# Using Plotly
fig = px.imshow(confusion_matrix(y_test,y_preds),color_continuous_scale="inferno")
fig.update_xaxes(title_text="Predicted label")
fig.update_yaxes(title_text="True label")
fig.show()

# %% [markdown]
# Now we've got a ROC curve, an AUC metric and a confusion matrix, let's get a classification report as well as cross-validated precision, recall and f1-score.

# %%
print(classification_report(y_test,y_preds))

# %% [markdown]
# ### Calculate evaluation metrics using cross-validation
# 
# We're going to calculate accuracy, precision, recall and f1-score of our model using cross-validation and to do so we'll be using `cross_val_score()`

# %%
# Chceck best hyperparameters
gs_log_reg.best_params_


# %%
# Create a new classifier with best parameters
clf = LogisticRegression(C=0.20433597178569418,
                         solver="liblinear")


# %%
# Cross-validated accuracy
cv_acc = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="accuracy")
cv_acc


# %%
cv_acc = np.mean(cv_acc)
cv_acc


# %%
# Cross validated precision
cv_precision = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="precision")
cv_precision = np.mean(cv_precision)
cv_precision


# %%
# Cross validated recall
cv_recall = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="recall")
cv_recall = np.mean(cv_recall)
cv_recall


# %%
# Cross validated f1-score
cv_f1 = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="f1")
cv_f1 = np.mean(cv_f1)
cv_f1


# %%
# Visualise cross-validated metrics using Matplotlib
cv_metrics = pd.DataFrame({"Accuracy":cv_acc,
                          "Precision":cv_precision,
                          "Recall":cv_recall,
                          "F1":cv_f1},
                          index=[0])

cv_metrics.T.plot.bar(title="Cross-validated classification metrics",
                      legend=False);
plt.xticks(rotation=0);


# %%
# Visualise cross-validated metrics using Plotly
fig = px.bar(cv_metrics.T)
fig.update_xaxes(title_text=" ")
fig.update_yaxes(title_text=" ")
fig.update_layout(title="Cross-validated classification metrics")
fig.show()

# %% [markdown]
# ### Feature Importance
# 
# Feature importance is same as asking, "Which features contributed most to the outcomes of the model and how did they contribute?"
# 
# Finding feature importance is different for each machine learning model. One way to find feature importance is to search for "(MODEL NAME) feature importance".
# 
# Let's find the feature importance for our LogisticRegression model...

# %%
# Fit an instance of LogisticRegression
clf = LogisticRegression(C=0.20433597178569418,
                         solver="liblinear")

clf.fit(X_train,y_train)


# %%
# Check coef_
clf.coef_


# %%
df.head()


# %%
# Match coef's of features to columns
feature_dict = dict(zip(df.columns,list(clf.coef_[0])))
feature_dict


# %%
# Visualize feature importance using Matplotlib
feature_df = pd.DataFrame(feature_dict,index=[0])
feature_df.T.plot.bar(title="Feature Importance",legend=False);


# %%
# Visualize feature importance using Plotly
fig = px.bar(feature_df.T)
fig.update_xaxes(title_text=" ")
fig.update_yaxes(title_text=" ")
fig.update_layout(title="Feature Importance")
fig.show()


# %%
pd.crosstab(df["sex"],df["target"])


# %%
pd.crosstab(df["slope"],df["target"])

# %% [markdown]
#  slope. Peak exercise ST segment 
#  * 0 = upsloping 
#  * 1 = flat
#  * 2 = downsloping.
# %% [markdown]
# ### Function to return Dictionary of Calculated Metrics.

# %%
def evaluate_preds(y_true,y_preds):
    """
    Performs evaluation comparison on y_true labels vs. y_pred labels 
    on a classification model.
    """
    accuracy = accuracy_score(y_true,y_preds)
    precision = precision_score(y_true,y_preds)
    recall = recall_score(y_true,y_preds)
    f1 = f1_score(y_true,y_preds)
    metric_dict = {"accuracy": round(accuracy,2),
                   "precision": round(precision,2),
                   "recall": round(recall,2),
                   "f1": round(f1,2)}
    print(f"Acc:{accuracy * 100:.2f}%")
    print(f"Precision:{precision:.2f}")
    print(f"Recall:{recall:.2f}")
    print(f"F1 score:{f1:.2f}")
    
    return metric_dict

# %% [markdown]
# ### Exporting the model
# 
# **Using `Joblib`**

# %%
from joblib import dump, load

# Save model to file
dump(clf, filename="Heart-Disease-Project.joblib")


# %%
# Import a saved joblib model
loaded_job_model = load(filename = "Heart-Disease-Project.joblib")


# %%
# Make and evaluate joblib predictions
joblib_y_preds = loaded_job_model.predict(X_test)
evaluate_preds(y_test,joblib_y_preds)

# %% [markdown]
# ## 6. Experimentation
# 
# If you haven't hit your evaluation metric yet... ask yourself...
# 
# * Could you collect more data?
# * Could you try a better model? Like CatBoost or XGBoost?
# * Could you improve the cuurent models? (beyond what we've done so far)
# * If your model is good enough (you have hit your evaluation metric) how would you export it and share it with others?

