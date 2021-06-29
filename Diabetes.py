
#Import the required packages

import pandas as pd
import numpy as np


# Get the file

data = pd.read_csv("C:/Users/Sai Praneeth S/Desktop/Machine_learning_Project/diabetes.csv")

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



# Look for Outliers

data.plot(kind = "box", vert = True)

