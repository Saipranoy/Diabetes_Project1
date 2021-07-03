
#Import the required packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Get the file

data = pd.read_csv("C:/Users/Sai Praneeth S/Desktop/Machine_learning_Project/diabetes.csv")

##############################################################################

# Dealing With Missing Values 

##############################################################################
# Look for Missing Values

data.isna().sum() # No missing values

 # looking at the data there are many values that are assigned as zero/null

data[data.columns[1:8]] = data[data.columns[1:8]].replace(0,np.nan)
data.isna().sum()

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
plt.legend(title = "Diabatics", loc = "right", labels = ["No","Yes"])
plt.tight_layout()
plt.show(p) # There is no patternbut having BMI INdex between (24-30) have higher chance of getting diabetes 

p = sns.histplot(data = data, x = "Age", hue = "Outcome", legend = False,binwidth = 3)
p.set_title("Distribution of data by Age", fontdict = {'fontsize': 18, 'fontweight':'bold'})
sns.set(font_scale = 1)
plt.legend(title = "Diabatics", loc = "right", labels = ["No","Yes"])
plt.tight_layout()
plt.show(p) # People of Age between (22-30) tend to get diabates 

p = sns.histplot(data = data, x = "Glucose", hue = "Outcome", legend = False,binwidth = 4)
p.set_title("Distribution of data by Glucose Levels",fontdict = {'fontsize': 18, 'fontweight':'bold'})
sns.set(font_scale = 1)
plt.legend(title = "Diabatics", loc = "right", labels = ["No","Yes"])
plt.tight_layout()
plt.show(p) # People with Glucose levels between (80 - 128) tend to get diabates 

p = sns.histplot(data = data, x = "SkinThickness", hue = "Outcome", legend = False,binwidth = 4)
p.set_title("Distribution of data by SkinThickness",fontdict = {'fontsize': 18, 'fontweight':'bold'})
sns.set(font_scale = 1)
plt.legend(title = "Diabatics", loc = "right", labels = ["No","Yes"])
plt.tight_layout()
plt.show(p) # having skin thickness between (27 - 32) tends to have diabates mostly

p = sns.histplot(data = data, x = "Insulin", hue = "Outcome", legend = False,binwidth = 46)
p.set_title("Distribution of data by Insulin", fontdict = {'fontsize': 18, 'fontweight':'bold'})
sns.set(font_scale = 1)
plt.legend(title = "Diabatics", loc = "right", labels = ["No","Yes"])
plt.tight_layout()
plt.show(p) # Having Insulin levels between ( 108-154) generally get diabates

p = sns.histplot(data = data, x = "BloodPressure", hue = "Outcome", legend = False,binwidth = 4)
p.set_title("Distribution of data by Blood Pressure levels", fontdict = {'fontsize': 18, 'fontweight':'bold'})
sns.set(font_scale = 1)
plt.legend(title = "Diabatics", loc = "right", labels = ["No","Yes"])
plt.tight_layout()
plt.show(p) # high BP levels leads to having diabatics

p = sns.histplot(data = data, x = "DiabetesPedigreeFunction", hue = "Outcome", legend = False,binwidth = 0.41)
p.set_title("Distribution of data by DiabetesPedigreeFunction", fontdict = {'fontsize': 18, 'fontweight':'bold'})
sns.set(font_scale = 1)
plt.legend(title = "Diabatics", loc = "right", labels = ["No","Yes"])
plt.tight_layout()
plt.show(p) #  having below 0.91 degree function value leads to Diabetes












































