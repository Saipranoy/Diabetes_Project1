
#Import the required packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.utils import resample, shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,recall_score,f1_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
# Get the file

data = pd.read_csv("C:/Users/Sai Praneeth S/Desktop/Machine_learning_Project/diabetes.csv")
data.head(5)
##############################################################################

# Dealing With Missing Values 

##############################################################################
# Look for Missing Values

data.isna().sum() # No missing values

 # looking at the data there are many values that are assigned as zero/null

data[data.columns[1:8]] = data[data.columns[1:8]].replace(0,np.nan)
data.isna().sum()
data.shape
# To fill the missing values we can use imputation method
data.skew() # Insulin and Diabates Pedigree Function are skewed

for var in ["Glucose","BloodPressure","SkinThickness","BMI"]:
    data[var].fillna(value = data[var].mean(),inplace = True)


data["Insulin"].fillna(value = data["Insulin"].median(), inplace = True)

# Though there are outliers in the data its better not to remove them as they are important in modelling the data

##############################################################################

# Exploratory Data Analysis

##############################################################################

# Corelation matrix

corr = data.corr()

# Glucose has high correlation with target variable that means people with higher glucose levels tend to get diabatics

# To understand in detail how each column varies with the target variable (outcome) 
# Lets plot each column variable to the target variable.

p = sns.catplot(x = "Pregnancies", hue = "Outcome", kind = "count",data = data, legend = False).set(title = "Distribution of Pregnancies")
plt.legend(title = "Diabatics", loc = "right", labels = ["No","Yes"])
plt.tight_layout()
plt.show(p)  # Higher the pregancies higher you get diabetics

p = sns.histplot(data = data, x = "BMI", hue = "Outcome", legend = False,binwidth = 2)
p.set_title("Distribution of data by BMI Index", fontdict = {'fontsize': 18, 'fontweight':'bold'})
sns.set(font_scale = 1)
plt.legend(title = "Diabatics", loc = "right", labels = ["Yes","No"])
plt.tight_layout()
plt.show(p) # There is no patternbut having BMI INdex between (24-30) have higher chance of getting diabetes 

p = sns.histplot(data = data, x = "Age", hue = "Outcome", legend = False,binwidth = 3)
p.set_title("Distribution of data by Age", fontdict = {'fontsize': 18, 'fontweight':'bold'})
sns.set(font_scale = 1)
plt.legend(title = "Diabatics", loc = "right", labels = ["Yes","No"])
plt.tight_layout()
plt.show(p) # People of Age between (22-30) tend to get diabates 

p = sns.histplot(data = data, x = "Glucose", hue = "Outcome", legend = False,binwidth = 4)
p.set_title("Distribution of data by Glucose Levels",fontdict = {'fontsize': 18, 'fontweight':'bold'})
sns.set(font_scale = 1)
plt.legend(title = "Diabatics", loc = "right", labels = ["Yes","No"])
plt.tight_layout()
plt.show(p) # People with Glucose levels between (80 - 128) tend to get diabates 

p = sns.histplot(data = data, x = "SkinThickness", hue = "Outcome", legend = False,binwidth = 4)
p.set_title("Distribution of data by SkinThickness",fontdict = {'fontsize': 18, 'fontweight':'bold'})
sns.set(font_scale = 1)
plt.legend(title = "Diabatics", loc = "right", labels = ["Yes","No"])
plt.tight_layout()
plt.show(p) # having skin thickness between (27 - 32) tends to have diabates mostly

p = sns.histplot(data = data, x = "Insulin", hue = "Outcome", legend = False,binwidth = 46)
p.set_title("Distribution of data by Insulin", fontdict = {'fontsize': 18, 'fontweight':'bold'})
sns.set(font_scale = 1)
plt.legend(title = "Diabatics", loc = "right", labels = ["Yes","No"])
plt.tight_layout()
plt.show(p) # Having Insulin levels between ( 108-154) generally get diabates

p = sns.histplot(data = data, x = "BloodPressure", hue = "Outcome", legend = False,binwidth = 4)
p.set_title("Distribution of data by Blood Pressure levels", fontdict = {'fontsize': 18, 'fontweight':'bold'})
sns.set(font_scale = 1)
plt.legend(title = "Diabatics", loc = "right", labels = ["Yes","No"])
plt.tight_layout()
plt.show(p) # high BP levels leads to having diabatics

p = sns.histplot(data = data, x = "DiabetesPedigreeFunction", hue = "Outcome", legend = False,binwidth = 0.41)
p.set_title("Distribution of data by DiabetesPedigreeFunction", fontdict = {'fontsize': 18, 'fontweight':'bold'})
sns.set(font_scale = 1)
plt.legend(title = "Diabatics", loc = "right", labels = ["Yes","No"])
plt.tight_layout()
plt.show(p) #  having below 0.91 degree function value leads to Diabetes

p = sns.catplot(x = "Outcome", kind = "count", data = data).set(title= "Distribution of Target Variable (Outcome)")
sns.set(font_scale = 1)
p.ax.patches
for i, bar in enumerate(p.ax.patches):
    h = bar.get_height()
    percent = (h/768)*100
    p.ax.text(
        i, # bar index (x coordinate of text)
        h+10, # y coordinate of text
        '{}%'.format(round(percent,2)),  # y label
        ha='center', 
        va='center', 
        fontweight='bold', 
        size=14)
plt.tight_layout()
plt.show(p) # This shows that the target value is not balanced which can lead to un-bias


#############################################################################

# Modeling the data

#############################################################################

# Feature Scaling - Is done so that all the feature values have same weightage when training.It is essential for
# ML Algorithms that use distance paramenter. 

# Dealing with Imbalanced Target Variable Column
#####################################################################

# Oversampling the outcome = 1 data to match the majority variable

# Seperate input features and target columns
y = data["Outcome"]
X = data.drop("Outcome", axis = 1)

# setting up training and testing set for logistic regression
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state  = 42)

# Join training data together
X = pd.concat([X_train,y_train],axis =1)

# Seperating the target class
not_diabatic = data[data["Outcome"] == 0]
diabatic = data[data["Outcome"] == 1]

# Upsampling diabatic variable
diabatic_upsampled = resample(diabatic,
                              replace = True,
                              n_samples = len(not_diabatic),
                              random_state = 42)


# join both the target variables
new_data = pd.concat([diabatic_upsampled, not_diabatic])

# Verify the resampling
new_data.Outcome.value_counts()

# Splitting the Data
y_train = new_data.Outcome
X_train = new_data.drop("Outcome", axis  = 1)

# Standardization of the data

scale = StandardScaler()
features = X_train.columns
X_train = scale.fit_transform(X_train[features])

X_test = scale.transform(X_test)

# Check with Classifier
upsampled = LogisticRegression(solver= 'liblinear').fit(X_train, y_train)

upsampled_pred = upsampled.predict(X_test)

# Checking accuracy score
accuracy_score( y_test, upsampled_pred)

# Checking Recall score
recall_score(y_test, upsampled_pred)

# Checking F1- Score
f1_score(y_test,upsampled_pred)

###################################################################

# Finding Optimum Threshold to get max f1 score

###################################################################

y_pred_prob = upsampled.predict_proba(X_test)[:,1]

thresholds = np.arange(0,1,0.01)

precision_scores = []
recall_scores = []
f1_scores = []

for threshold in thresholds:
    
    pred_class = (y_pred_prob >= threshold) * 1
    
    precision = precision_score(y_test, pred_class, zero_division = 0)
    precision_scores.append(precision)

    recall = recall_score(y_test, pred_class)
    recall_scores.append(recall)
 
    f1 = f1_score(y_test, pred_class)
    f1_scores.append(f1)

max_f1 = max(f1_scores)
max_f1_idx = f1_scores.index(max_f1)


plt.style.use("seaborn-poster")
plt.plot(thresholds, precision_scores, label = "precision", linestyle = "--")
plt.plot(thresholds, recall_scores, label = "recall", linestyle = "--")
plt.plot(thresholds, f1_scores, label = "f1 score", linewidth = 5)
plt.title(f"Finding the Optimal Threshold for Classification Model \n Maximum f1 Score :{round(max_f1,2)} at threshold {round(thresholds[max_f1_idx],2)} ")
plt.xlabel("Threshold")
plt.ylabel("Assesment Score")
plt.legend(loc ="lower left")
plt.tight_layout()
plt.show()

# To apply the optimal threshold to the data

optimal_thresh = 0.39
y_pred_opt_thresh = (y_pred_prob >= optimal_thresh) * 1

accuracy_score(y_test,y_pred_opt_thresh)

# Of all the observations, how many were actually positive
precision_score(y_test, y_pred_opt_thresh)

#of All the postive observations, how many were predicted positive
recall_score(y_test, y_pred_opt_thresh)

# Hormonic mean of precall and recall scores
f1_score(y_test, y_pred_opt_thresh)

##################################################################
# Decision Tree Classification (dtc)
########################################################################

dtc = DecisionTreeClassifier(random_state=42, max_depth=9)
dtc.fit(X_train, y_train)

# Assesing the Model

y_pred_dtc = dtc.predict(X_test)

# Confusion Matrix

conf_matrix_dtc = confusion_matrix( y_test, y_pred_dtc)

plt.style.use("seaborn-poster")
plt.matshow(conf_matrix_dtc, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix (Decision Tree Classification)")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
for (i,j),corr_value in np.ndenumerate(conf_matrix_dtc):
    plt.text(j,i,corr_value,ha = "center",va = "center", fontsize  = 20)
plt.show()

accuracy_score(y_test, y_pred_dtc)
precision_score(y_test, y_pred_dtc)
recall_score(y_test, y_pred_dtc)
f1_score(y_test, y_pred_dtc)

# Optimise the Max Depth

max_depth = list(range(1,15))
accuracy_list = []

for depth in max_depth:
    clf = DecisionTreeClassifier(max_depth  = depth, random_state = 42 )
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = f1_score(y_pred, y_test)
    accuracy_list.append(accuracy)
    
max_accuracy = max(accuracy_list)
max_accuracy_index = accuracy_list.index(max_accuracy)
optimal_max_depth = max_depth[max_accuracy_index]


# Plot of max_depths

plt.plot(max_depth, accuracy_list)
plt.scatter(optimal_max_depth, max_accuracy, marker = "x", color = "red")
plt.title(f"Accuracy (F1 Score) by Max Depth \n Optimal Tree Depth:{optimal_max_depth} (Accuracy : {round(max_accuracy,4)})")
plt.xlabel("Max Depth of Decission Tree")
plt.ylabel("Accuracy (F1 Score)")
plt.tight_layout()
plt.show()

# Plot our Model

plt.figure(figsize=(25,15))
tree = plot_tree(clf,
                 feature_names = X.columns,
                 filled = True,
                 rounded = True,
                 fontsize = 16)

# With max_depth or stopping value for splitting the data as 12 we will have f1score 90% and then 
# we can test the data after training with new stopping value to split data.

######################################################################
# Random Forest Classification (rfc)
######################################################################

rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)

# Assesing the Model

y_pred_rfc = rfc.predict(X_test)

accuracy_score(y_test, y_pred_rfc)


















