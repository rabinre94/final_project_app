import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.markdown("# Welcome to my Final Project Streamlit!")

s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    x = np.where(x==1,1,0)
    return x

ss = pd.DataFrame({
    "income":np.where(s["income"]>9,np.nan,s["income"]),
    "educ2":np.where(s["educ2"]>8,np.nan,s["educ2"]),
    "par":np.where(s["par"]==1,1,0),
    "marital":np.where(s["marital"]==1,1,0),
    "female":np.where(s["gender"]==2,1,0),
    "age":np.where(s["age"]>98,np.nan,s["age"]),
    "sm_li":clean_sm(s["web1h"])})

ss = ss.dropna()

y = ss["sm_li"]
X = ss[["income", "educ2", "par", "marital","female","age"]]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,
                                                    test_size=0.2,
                                                    random_state=987)

# Initialize algorithm 
lr = LogisticRegression(class_weight='balanced')

# Fit algorithm to training data
lr.fit(X_train, y_train)

# Make predictions using the model and the testing data
y_pred = lr.predict(X_test)

st.markdown("### In order to predict whether or not you are a LinkedIn user, please use the dropdowns below and ensure each accurately describes you. Once all inputs are updated, please note our final prediction and probability below.")

# income (household): 
#	1	Less than $10,000
#	2	10 to under $20,000
#	3	20 to under $30,000
#	4	30 to under $40,000
#	5	40 to under $50,000
#	6	50 to under $75,000
#	7	75 to under $100,000
#	8	100 to under $150,000, OR
#	9	$150,000 or more?
#	98	(VOL.) Don't know
#	99	(VOL.) Refused

income2 = st.selectbox("What is your income?",["Less than $10,000","10 to under $20,000","20 to under $30,000","30 to under $40,000","40 to under $50,000","50 to under $75,000",
                                 "75 to under $100,000","100 to under $150,000","$150,000 or more"])


if income2 == "Less than $10,000":
    income2 = 1
elif income2 == "10 to under $20,000":
    income2 = 2
elif income2 == "20 to under $30,000":
    income2 = 3
elif income2 == "30 to under $40,000":
    income2 = 4
elif income2 == "40 to under $50,000":
    income2 = 5
elif income2 == "50 to under $75,000":
    income2 = 6
elif income2 == "75 to under $100,000":
    income2 = 7
elif income2 == "100 to under $150,000":
    income2 = 8
else:
    income2 = 9

# st.write(f"{income3}")

#educ2 (highest level of school/degree completed):
#	1	Less than high school (Grades 1-8 or no formal schooling)
#	2	High school incomplete (Grades 9-11 or Grade 12 with NO diploma)
#	3	High school graduate (Grade 12 with diploma or GED certificate)
#	4	Some college, no degree (includes some community college)
#	5	Two-year associate degree from a college or university
#	6	Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)
#	7	Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)
#	8	Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)
#	98	Don’t know
#	99	Refused

educ22 = st.selectbox("What is your education level?",["Less than high school (Grades 1-8 or no formal schooling)","High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
                                   "High school graduate (Grade 12 with diploma or GED certificate)","Some college, no degree (includes some community college)",
                                   "Two-year associate degree from a college or university","Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)",
                                 "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
                                 "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)",])


if educ22 == "Less than high school (Grades 1-8 or no formal schooling)":
    educ22 = 1
elif educ22 == "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)":
    educ22 = 2
elif educ22 == "High school graduate (Grade 12 with diploma or GED certificate)":
    educ22 = 3
elif educ22 == "Some college, no degree (includes some community college)":
    educ22 = 4
elif educ22 == "Two-year associate degree from a college or university":
    educ22 = 5
elif educ22 == "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)":
    educ22 = 6
elif educ22 == "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)":
    educ22 = 7
else:
    educ22 = 8

#par (are you a parent of a child under 18 living in your home?):
#	1	Yes
#	2	No
#	8	(VOL.) Don't know
#	9	(VOL.) Refused

par2 = st.selectbox("Are you a parent?",["Yes","No"])

if par2 == "Yes":
    par2 = 1   
else:
    par2 = 0
    

# marital (current marital status)
#
#	1	Married
#	2	Living with a partner
#	3	Divorced
#	4	Separated
#	5	Widowed
#	6	Never been married
#	8	(VOL.) Don't know
#	9	(VOL.) Refused

marital2 = st.selectbox("Are you married?",["Yes","No"])

if marital2 == "Yes":
    marital2 = 1  
else:
    marital2 = 0

# gender
#	1 male
#	2 female
#	3 other
#	98 Don't know
#	99 Refused

gender2 = st.selectbox("What is your gender?",["Male","Female"])

if gender2 == "Female":
    gender2 = 1 
else:
    gender2 = 0

# age (numeric age):

#	number
#	97 - 97+
#	98 Don't know

age2 = st.slider("What is your age?",1,97)

# newdata = pd.DataFrame({
#    "income": [income2],
#    "educ2": [educ22],
#    "par": [par2],
#    "marital": [marital2],
#    "female": [gender2],
#    "age": [age2],
# })

# newdata["prediction_sm_li"] = lr.predict(newdata)

# New data for features: age, college, high_income, ideology
person = [income2,educ22,par2,marital2,gender2,age2]

# Predict class, given input features
predicted_class = lr.predict([person])

# Generate probability of positive class (=1)
probs = lr.predict_proba([person])

finalstatement = 0

if predicted_class == 0:
    finalstatement = "Based on the inputs above, we predict you ARE NOT a LinkedIn User!"
else:
    finalstatement = "Based on the inputs above, we predict you ARE a LinkedIn User!"

# Print final prediction and probability
st.markdown(f"## {finalstatement}") # 0=not pro-environment, 1=pro-envronment
st.markdown(f"### The probability that you are a LinkedIn user is {round(probs[0][1],2)}")

