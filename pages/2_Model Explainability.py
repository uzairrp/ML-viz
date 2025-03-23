import streamlit as st
import shap
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  

st.set_page_config(page_title="Model Explainability", page_icon="💡")

st.title('Model Explainability')
st.subheader("In this page, we will see how the model arrives at the outputs it does and how these visualizations help tailor our model.")

# Load the pre-trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

X = st.session_state.get("inputs", None)
X = X.drop(columns=['price'])  # Drop the target column (price)

# Initialize SHAP explainer
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# SHAP Summary Plot
st.write("### SHAP Summary Plot")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X)
st.pyplot(fig)

st.write('In the plot above, we can see how the values of different features impact the output. All the features have a cluster around 0. Furthermore, we see that not all the features have the same spread.')

# Mean Impact Value Plot
st.write('### Mean Impact Value plot')
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X, show=False, plot_type="bar")
st.pyplot(fig)
st.write("Seeing the mean impact value of the different features, we decide to further look into the top 2 features to see the relation between the actual value and the impact value.")

# Identify the top 2 features based on mean impact
shap_summary = pd.DataFrame(
    {
        "Feature": X.columns,
        "Mean Impact": np.abs(shap_values.values).mean(axis=0)
    }
).sort_values(by="Mean Impact", ascending=False)

# Scatter plot for the two top features
top_features = shap_summary["Feature"].head(2).values
st.write("### Scatter Plot for Top 2 Features")
for feature in top_features:
    st.write(f"#### Scatter Plot: {feature} vs SHAP Impact")
    fig = shap.plots.scatter(shap_values[:, X.columns.get_loc(feature)], show=False)
    st.pyplot(fig)

st.write("In both cases we can see that the feature values have a range of impact values. This suggests that their impact is influenced by the values of the other features within the same record.")

# Waterfall Plot
st.write("### Waterfall Plot")
st.write("Now, we will visualize the detailed contribution of each feature for the first prediction.")
shap.plots.waterfall(shap_values[0])
st.pyplot(fig)

st.write('We can see how the features contribute to a prediction and if their values make the prediction go up or down. Cross checking this plot for the values of the features used in the first prediction, we can conclude that the model progresses logically.')

# For some reason the following lines kept returning error.
# Decision Plot
# st.write("### Decision Plot")
# st.write("We will now visualize how the model arrives at the prediction for the first instance.")
# shap.plots.decision(shap_values[0])  
# st.pyplot(fig)



