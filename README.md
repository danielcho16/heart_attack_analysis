# Heart Attack Analysis

## Exploratory Data Analysis and Prediction Models

![heart_diagram](https://github.com/danielcho16/heart_attack_analysis/assets/82684796/0aba3b0b-0e83-4c71-bac0-48c66d330867)

<br>
<br>

### Introduction

This project uses the provided dataset and creates prediction models to analyze the likelihood of heart attacks in individuals based on various variables.

Full analysis can be found [here](heart_attack.ipynb)

----

### Dataset

Dataset can be found on [Kaggle](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset/code?datasetId=1226038&sortBy=voteCount)

----

### Variable definitions in the dataset

- **age**: age of the patient (in years)
- **sex**: sex of the patient
    - 0: female
    - 1: male
- **cp**: chest pain type
    - 1: typical angina
    - 2: atypical angina
    - 3: non-anginal pain
    - 4: asymptomatic
- **trtbps**: resting blood pressure (in mm Hg)
- **chol**: serum cholestoral in mg/dl
- **fbs**: fasting blood sugar > 120 mg/dl
    - 0: False
    - 1: True
- **restecg**: resting electrocardiographic results
    - 0: hypertrophy
    - 1: normal
    - 2: having ST-T wave abnormality
- **thalachh**: maximum heart rate achieved
- **exng**: exercise induced angina
    - 0: no
    - 1: yes
- **oldpeak**: previous peak
- **slp**: the slope of the peak exercise ST segment
    - 0: downsloping
    - 1: flat
    - 2: upsloping
- **caa**: number of major vessels (0-3)
- **thall**: thallium stress test result (0-3)
    - 0: null
    - 1: fixed defect
    - 2: normal
    - 3: reversable defect
- **output**: the predicted attribute
    - 0: less chance of heart attack
    - 1: more chance of heart attack

----

### Data Visualizations

Correlation Matrix

![corr](https://github.com/danielcho16/heart_attack_analysis/assets/82684796/c36c591e-2246-4721-82cf-6e16bfb84e71)

<br>
<br>

Countplot of the Target

![countplot_target](https://github.com/danielcho16/heart_attack_analysis/assets/82684796/632190c3-e0d0-490e-9838-359cc97e4b95)

<br>
<br>

Individual Countplot Graphs of Categorical Variables

![countplot_categorical](https://github.com/danielcho16/heart_attack_analysis/assets/82684796/5665fd5f-79db-4292-a0c7-451f1c946402)

<br>
<br>

Individual Countplot Graphs of Categorical Variables in Relation to the Target

![countplot_categorical_output](https://github.com/danielcho16/heart_attack_analysis/assets/82684796/9bdf1a50-8c3a-4940-a220-b6de97368cfd)

<br>
<br>

Individual Histplot Graphs of Numerical Variables

![histplot_numerical](https://github.com/danielcho16/heart_attack_analysis/assets/82684796/db2cb99b-f262-43dc-ab73-203058400740)

<br>
<br>

Individual KDEplot Graphs of Numerical Variables in Relation to the Target

![kdeplot_numerical](https://github.com/danielcho16/heart_attack_analysis/assets/82684796/840468f7-6cfd-4a27-be26-b469808956fa)

----

### Prediction Model

Data Preprocessing
```
heart = pd.get_dummies(heart_df,columns=cat_cols,drop_first=True)

X = heart.drop('output',axis=1)
y = heart['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

<br>

Random Forest model
```
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

classification_report(y_test,y_pred)

confusion_matrix(y_test,y_pred)

accuracy_score(y_test,y_pred)
```

<br>

<img width="400" alt="random_forest" src="https://github.com/danielcho16/desktop-tutorial/assets/82684796/19a5d847-58fa-4e06-ba4e-59279ea545e1">

----
