import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit Title
st.title("Customer Churn Analysis")

# Load Data
def load_data():
    df = pd.read_csv('Customer Churn.csv')
    df["TotalCharges"] = df["TotalCharges"].replace(" ", "0").astype("float")
    df['SeniorCitizen'] = df['SeniorCitizen'].apply(lambda x: "Yes" if x == 1 else "No")
    return df

df = load_data()

# Show Data Preview
if st.checkbox("Show Raw Data"):
    st.write(df.head())

# Churn Count Plot
st.subheader("Churn Count Distribution")
fig, ax = plt.subplots()
sns.countplot(x='Churn', data=df, ax=ax)
ax.bar_label(ax.containers[0])
st.pyplot(fig)

# Churn Pie Chart
st.subheader("Percentage of Churned Customers")
fig, ax = plt.subplots(figsize=(3, 3))
gb = df.groupby("Churn").size()
ax.pie(gb, labels=gb.index, autopct='%1.2f%%')
st.pyplot(fig)

# Churn by Gender
st.subheader("Churn by Gender")
fig, ax = plt.subplots()
sns.countplot(x='gender', data=df, hue='Churn', ax=ax)
st.pyplot(fig)
