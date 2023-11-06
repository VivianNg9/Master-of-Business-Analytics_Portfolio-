# Portfolio 4
## Heart Failure Prediction 
- __Purpose__: Build and evaluate models for heart failure prediction so that we can determine which features play the most important role in predicting whether a patient has heart disease or not.
  
- __Source of the dataset__: Kaggle-https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

 ### Description of Fields
* __Age__: age of the patient [years]
* __Sex__: sex of the patient [M: Male, F: Female]
* __ChestPainType__: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
* __RestingBP__: resting blood pressure [mm Hg]
* __Cholesterol__: serum cholesterol [mm/dl]
* __FastingBS__: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
* __RestingECG__: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
* __MaxHR__: maximum heart rate achieved [Numeric value between 60 and 202]
* __ExerciseAngina__: exercise-induced angina [Y: Yes, N: No]
* __Oldpeak__: oldpeak = ST [Numeric value measured in depression]
* __ST_Slope__: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
* __HeartDisease__: output class [1: heart disease, 0: Normal]

### The task sections
- Explore the dataset
- Logistic Regression
- KNN Model 
- Evaluation 
  - Confusion Matrix
  - Classification Report
  - ROC Curve and AUC

### Outcome of the task 
- Model Performance
  - Both the Logistic Regression and KNN models showcased commendable performance with 85.87%.
  - Hyper-parameter K tuning on model performance with an accuracy of about 86.24%.

- Feature Importance
  - The 5 most important features that play a significant role in predicting heart disease are Sex, ChestPainType, FastingBS, ExerciseAngina, ST_Slope. 

- Recall & Specificity 
  - The model's high recall is of utmost importance in a medical context.

- AUC-ROC
  - The value of 0.9051 means that there's a 90.51% chance that the model will be able to distinguish between positive and negative classes.
  
