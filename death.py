from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import numpy as np


death_data = pd.read_csv("death.csv")

new_data = death_data["Marital_Status"].unique()
print(new_data)

death_data["Sex"] = death_data["Sex"].map({
    "Male":0,
    "Female":1 
})

death_data["Development_Level"]=death_data["Development_Level"].map({
    "Low":0,"Medium":1,"High":2,"Very High":3
})

death_data["Urban_Rural"]=death_data["Urban_Rural"].map({
    "Urban":0,"Rural":1
})

death_data["Marital_Status"] = death_data["Marital_Status"].map({
    "Single":0,
    "Married":1,
    "Widowed":2,
    "Divorced":3
})

death_data["Employment_Status"]=death_data["Employment_Status"].map({
    "Employed":0,
    "Retired":1,
    "Student":2,
    "Unemployed":3,
    "Homemaker":4
})

death_data["Occupation_Risk"] = death_data["Occupation_Risk"].map({
    "Low":0,
    "Medium":1,
    "High":2
})

death_data["Smoking_Status"] = death_data["Smoking_Status"].map({
    "Former":0,
    "Never":1,
    "Current":2
})
death_data["Drug_Use"] = death_data["Drug_Use"].map({
    "Never":0,
    "Occasional":1,
    "Regular":2
})

death_data["Driving_Habits"] = death_data["Driving_Habits"].map({
    "Occasionally risky":0,
    "risky":1,
    "Safe":2,
    "Often risky":3
})

death_data["Diabetes_Status"]= death_data["Diabetes_Status"].map({
    "Prediabetes":0,
    "No":1,
    "Yes":2
})

death_data["Hypertension"] = death_data["Hypertension"].map({
    "No":0,
    "Yes":1
})

death_data["Heart_Disease"] = death_data["Heart_Disease"].map({
    "No":0,
    "Yes":1
})
death_data['Stroke_History'] = death_data["Stroke_History"].map({
    "No":0,
    "Yes":1
})

death_data['Cancer_History'] = death_data["Cancer_History"].map({
    "No":0,
    "Yes":1
})
death_data['Chronic_Lung_Disease'] = death_data["Chronic_Lung_Disease"].map({
    "No":0,
    "Yes":1
})
death_data['Kidney_Disease'] = death_data["Kidney_Disease"].map({
    "No":0,
    "Yes":1
})
death_data['Depression_Anxiety'] = death_data["Depression_Anxiety"].map({
    "No":0,
    "Yes":1
})
death_data['Annual_Checkup'] = death_data["Annual_Checkup"].map({
    "No":0,
    "Yes":1
})
death_data['Vaccinated'] = death_data["Vaccinated"].map({
    "No":0,
    "Yes":1
})

death_data['Parents_Lived_To_80'] = death_data["Parents_Lived_To_80"].map({
    "Neither":0,
    "Both":1,
    "One":2
})

death_data["Family_Heart_Disease"] = death_data['Family_Heart_Disease'].map({
    "No":0,
    "Yes":1
})
death_data["Family_Cancer"] = death_data['Family_Cancer'].map({
    "No":0,
    "Yes":1
})
death_data["Owns_Pet"] = death_data['Owns_Pet'].map({
    "No":0,
    "Yes":1
})

death_data.to_csv("new_data.csv",index=False)

dataset = pd.read_csv("new_data.csv")

f = dataset.drop(columns=['Years_Remaining','Life_Expectancy_Age'])

t = dataset['Years_Remaining']

model = RandomForestClassifier