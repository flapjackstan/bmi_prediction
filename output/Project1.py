#!/usr/bin/env python
# coding: utf-8

# # Project 1

# Use the data *diabetes2.csv* for this project. More information about the dataset can be found here: https://www.kaggle.com/kandij/diabetes-dataset

# ## Linear Regression
# 
# Can you predict BMI based on other features in the dataset?
# 
# 1. Explore the Data
# 2. Build your Model
#     - Build a Linear Regression Model using train_test_split() for your cross-validation
#     - Standardize your continuous predictors
# 3. Evaluate your model
#     - How did your model do? What metrics do you use to support this?
# 4. Interpret the coefficients to your model
#     - In the context of this problem, what do the coefficients represent?
#   
# 

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from plotnine import *
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression # Linear Regression Model
from sklearn.preprocessing import StandardScaler, LabelBinarizer 
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve,mean_squared_error, r2_score, roc_auc_score #model eval
from sklearn.model_selection import train_test_split # simple TT split cv
from sklearn.model_selection import KFold # k-fold cv
#from sklearn.model_selection import LeaveOneOut #LOO cv
from sklearn.model_selection import cross_val_score # cross validation metrics
from sklearn.model_selection import cross_val_predict # cross validation metrics

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


diabetes = pd.read_csv('data/diabetes2.csv')
print(diabetes.head())
print(diabetes.describe())
print(diabetes.info())
print(diabetes.isnull().sum())
#diabetes.loc[diabetes['SkinThickness'] == 0]
#diabetes.loc[diabetes['Weight'].isnull()]


# Removing 0 values to improve model metrics

# In[3]:


diabetes = diabetes[diabetes.BloodPressure != 0] #removed 0 values to avoid throwing off the model
diabetes = diabetes[diabetes.SkinThickness != 0]
diabetes = diabetes[diabetes.Insulin != 0]
diabetes = diabetes[diabetes.BMI != 0]
diabetes = diabetes[diabetes.Glucose != 0]

diabetes = diabetes.reset_index(drop = True)
print(diabetes.shape)
print(diabetes.describe())


# In[ ]:


#g = sns.pairplot(diabetes)


# In[4]:


diabetes.loc[diabetes['BMI'] < 18.5, 'Weight'] = 'Underweight'
diabetes.loc[diabetes['BMI'].between(18.5,24.999), 'Weight'] = 'Normal'
diabetes.loc[diabetes['BMI'].between(25,29.999), 'Weight'] = 'Overweight'
diabetes.loc[diabetes['BMI'] >= 30, 'Weight'] = 'Obese'

diabetes["Weight"] = diabetes["Weight"].astype('category')

label_binary = LabelBinarizer()
lb_results = label_binary.fit_transform(diabetes["Weight"])
column_names = label_binary.classes_
# lb_results.shape
# column_names

lb_df = pd.DataFrame(lb_results, columns = column_names)
#print(lb_df)

diabetes = diabetes.join(lb_df, lsuffix='index', rsuffix='index')

diabetes.head(20)


# diabetes["Weight_cat"] = diabetes["Weight"].cat.codes

# print(diabetes.describe(include = "all"))
# diabetes.head(20)


# In[5]:


g = sns.scatterplot(data = diabetes,x = 'Age', y = 'BMI',
                   sizes=(20, 200), hue_norm=(0, 7), legend = False)


# Seems like age doesnt play too much of a factor on BMI. I figured there would be lower BMI's in older women

# In[6]:


g = sns.scatterplot(data = diabetes,x = 'BloodPressure', y = 'BMI',
                   sizes=(20, 200), hue_norm=(0, 7), legend = False)


# It seems like there are many women of with both above average (31) BMI and above average blood pressure (69) 

# In[7]:


corr = diabetes.corr()
sns.heatmap(corr,xticklabels=corr.columns, yticklabels=corr.columns)


# Correlation matrix shows strong relationships between a few variables. The darker the color the more negative a relationship and the lighter means the more positive relationship. Examples of negative relationships are highlighed in the Normal weight range. Notice how nearly all squares are dark indicating that a normal weight range has a negative relationship with nearly all other variables. The opposite can be seen in the obese weight range where colors are lighter and signify a positive relationship between obesity and variables such as glucose and insulin.

# In[8]:


predictors = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction', 'Age', 'Outcome']


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(diabetes[predictors], diabetes["BMI"], test_size=0.2)


# In[10]:


zscore = StandardScaler()
zscore.fit(X_train)
Xz_train = zscore.transform(X_train)
Xz_test = zscore.transform(X_test)


# In[11]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[12]:


train_pred = model.predict(X_train)
test_pred = model.predict(X_test)


# In[13]:


print('training r2 is:', model.score(X_train, y_train)) #training R2
print('testing r2 is:', model.score(X_test, y_test)) #testing R2

print('\ntrain mse is: ', mean_squared_error(y_train,train_pred))
print('test mse is: ', mean_squared_error(y_test,test_pred))


# In[14]:


coefficients = pd.DataFrame({"Coef":model.coef_,
              "Name": predictors})

coefficients = coefficients.append({"Coef": model.intercept_,
               "Name": "intercept"}, ignore_index = True)

coefficients


# How did your model do? What metrics do you use to support this?
# <br/>
# <br/>
# The model did well given decent r-squared scores in both training and testing sets. MSE in both training and test set are within one point of each other signifying a good fit. A way to improve this model might be to include a better cross validation method as well as getting more data.
# <br/>

# In[ ]:





# In the context of this problem, what do the coefficients represent?
# <br/>
# <br/>
# The coefficients represent the value that BMI increases or decreases for every one standard deviation for its corresponding variable. For example, for every one standard deviation in pregnancy a woman is, her BMI is predicted to go down by her number of pregnancies multiplied by -.17. The strongest predictor in our model is whether the person is diabetic or not, if a woman were to be diabetic, our model adds 1.93 to her predicted BMI. Second highest variable is the diabetes pedigree funtion meaning that family history plays a large role in a womans BMI.
# <br/>

# ## Logistic Regression
# 
# Can you predict Diabetes (Outcome) based on other features in the dataset?
# 
# 1. Explore the Data (if using different variables from Linear Regression)
# 2. Build your Model
#     - Build a Logistic Regression Model using cross-validation
#        - What cross-val method did you choose, why?
#     - Standardize your continuous predictors
# 3. Evaluate your model
#     - How did your model do? What metrics do you use to support this?
#   

# In[15]:


yes = diabetes[diabetes['Outcome'] == 1]
no = diabetes[diabetes['Outcome'] == 0]


# In[16]:


(ggplot()
+geom_point(yes, aes(x = 'BloodPressure' ,y = 'Insulin'), color = 'red')
+geom_point(no, aes(x = 'BloodPressure' ,y = 'Insulin'), color = 'green'))


# In[17]:


(ggplot()
+geom_bar(yes, aes(x = 'DiabetesPedigreeFunction'), color = 'red')
+geom_bar(no, aes(x = 'DiabetesPedigreeFunction'), color = 'green'))


# In[18]:


predictors = ['Normal','Pregnancies', 'BMI', 'BloodPressure', 'SkinThickness', 'Glucose', 'Insulin', 'DiabetesPedigreeFunction', 'Age']
X = diabetes[predictors]
y = diabetes["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[39]:


# create k-fold object
kf = KFold(n_splits = 5)
kf.split(X)


lr = LogisticRegression() 

acc = [] #create empty list to store accuracy for each fold
predictedVals = []


# In[ ]:


#print(diabetes.isnull().sum())
#print(diabetes.isna().sum())
#diabetes.replace([np.inf, -np.inf], np.nan).dropna(axis=1)


# In[40]:


for train_indices, test_indices in kf.split(X):
    # Get your train/test for this fold
    y=y.reset_index(drop=True)
    X_train = X.iloc[train_indices] #iloc used to find col index
    X_test  = X.iloc[test_indices]
    y_train = y[train_indices]
    y_test  = y[test_indices]
    
    #standardize
    zscore = StandardScaler()
    zscore.fit(X_train)
    
    Xs_train = zscore.transform(X_train)
    Xs_test = zscore.transform(X_test)
    
    # model
    model = lr.fit(Xs_train, y_train)
    # record accuracy and predictions
    
    acc.append(accuracy_score(y_test, model.predict(Xs_test)))
    
print(acc)
np.mean(acc)


# What cross-val method did you choose, why?
# <br/>
# <br/>
# I chose a k-fold validation method with 5 folds because k-fold reduces bias when constructing a model and limits the variance our model is exposed to in training.
# <br/>

# In[41]:


predictedVals = model.predict(Xs_test) #predic
accuracy_score(y_test,predictedVals)


# In[42]:


cnf_matrix = confusion_matrix(y_test, predictedVals)
cnf_matrix


# In[43]:


class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=False, cmap="Blues")
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[44]:


print("Accuracy - TP+TN/TP+FP+FN+TN:",accuracy_score(y_test, predictedVals))
print("Precision - TP/TP+FP:",precision_score(y_test, predictedVals)) #relates to low false positivity
print("Recall/Sensitivity/TPR - TP/TP+FN:",recall_score(y_test, predictedVals))


# In[45]:


y_pred_proba = model.predict_proba(Xs_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="Outcome, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# How did your model do? What metrics do you use to support this?
# <br/>
# <br/>
# The model did well accurately predicting 83% of our test data and having an auc score of 87%. However our models true positive rate is 68% meaning that 32% percent of women would be given a false negative report when they actually do have diabetes. 
# <br/>

# In[27]:


coeff = list(model.coef_[0])
labels = list(X_test.columns)
features = pd.DataFrame()
features['Features'] = labels
features['importance'] = coeff
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0
features.set_index('Features', inplace=True)
features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')


# ## Data Viz
# 
# Based on your new understanding of the data create 2 graphs using ggplot/plotnine. These should **not** be graphs you made in the Explore phase of either the Logistic or Linear Regression portion.
# 
# Make sure you include at **least** 3 out of these 5 elements in your at least one of your graphs:
# 
# 1. Custom x-axis labels, y-axis labels and titles
# 2. Fill and/or Color by a variable
# 3. Use facet_wrap()
# 4. Layer multiple geoms
# 5. Change the theme of your graph (see: https://plotnine.readthedocs.io/en/stable/generated/plotnine.themes.theme.html)
# 

# In[46]:


diabetes['count'] = 1

(ggplot(diabetes,aes(x = 'Outcome', y = 'count', fill = 'Outcome')) #fill
+geom_bar(stat = 'identity')
+facet_wrap('~Weight') #facet wrap
+theme_minimal()) #theme


# In[47]:


(ggplot(diabetes,aes(x = 'Outcome', y = 'Glucose', fill = 'Outcome')) #fill
+geom_bar(stat = 'identity')
+facet_wrap('~Weight') #facet wrap
+theme_minimal()) #theme


# In[ ]:




