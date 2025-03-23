import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


st.set_page_config(page_title="Data Introduction", page_icon="🚗")

st.title('Automobile Prices')
st.subheader("In this web app, we will learn about the automobile prices in United States in the 80's.")

st.write("This is the data that we are working with:")
data = pd.read_csv("./data/auto_red.csv")
st.write(data.head(4))

st.write("Below we have the features and their datatypes:")
cols = data.columns
for i in range(0, len(cols), 2):
    col1, col2 = st.columns(2)  # Create two columns
    col1.write(f"**{cols[i]}**: {data[cols[i]].dtype}")  # First column
    if i + 1 < len(cols):  # Check if there's a second column
        col2.write(f"**{cols[i+1]}**: {data[cols[i+1]].dtype}")  # Second column

st.write("The numerical columns have the following values for basic statistics:")
st.write(data.describe())

st.write("We transform some of our columns and encode the non-numerical columns to obtain:")
endata = pd.read_csv("./data/auto_encoded.csv")
X = endata.drop(columns = ['engine_location', 'engine_type', 'fuel', 'aspiration', 'peak_rpm'])
st.write(X.head(4))

st.write("The correlation matrix for these features is as follows:")
# ndata = data.select_dtypes(include=['number'])
cm = X.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.write("Even though some of these correlation values are very small, as we will see, their corresponding SHAP impact values are reasonable enough to keep them for the model training.")

# sharing the X variable across pages
st.session_state['inputs'] = X