import pandas as pd
import pickle
import streamlit as st



# Set the page title and description
st.title("Mall Customer Subdivisions")


# # Optional password protection (remove if not needed)
# password_guess = st.text_input("Please enter your password?")
# # this password is stores in streamlit secrets
# if password_guess != st.secrets["password"]:
#     st.stop()


# Load the pre-trained model
with open("models/kmodel.pkl", "rb") as k_pickle:
    kmodel = pickle.load(k_pickle)
    k_pickle.close()

# Define the cluster centroids (replace with your actual centroids)
centroids = [
    [44.14285714, 25.14285714, 19.52380952],   # Cluster 0 centroid
    [32.69230769, 86.53846154, 82.12820513],   # Cluster 1 centroid
    [24.8,        41.46,       63.7       ],   # Cluster 2 centroid
    [40.32432432, 87.43243243, 18.18918919],   # Cluster 3 centroid
    [53.50943396, 54.73584906, 48.47169811],   # Cluster 4 centroid
]

# Prepare the form to collect user inputs
with st.form("user_inputs"):
    st.subheader("Customer Details")
    
    # Gender input
    Gender = st.selectbox("Gender", options=["Male", "Female"])
    
    # Age input
    Age = st.number_input("Age", min_value=0, step=1)
    
    # Annual Income input
    Annual_Income = st.number_input("Annual Income (in thousands of $)", min_value=10, max_value=150, step=10)
    
    # Spending Score input
    Spending_Score = st.number_input("Spending Score (1-100)", min_value=10, max_value=100, step=10)
    
    # Submit button
    submitted = st.form_submit_button("Predict Customer Clasification")


# Make prediction wwhen submitting form
if submitted:
     # Create a DataFrame with the user inputs
    user_data = pd.DataFrame({
        'Gender': [Gender],
        'Age': [Age],
        'Annual_Income': [Annual_Income],
        'Spending_Score': [Spending_Score]
    })

    # Feature engineering
    X_user = user_data[['Age', 'Annual_Income', 'Spending_Score']]

      # Make prediction
    label = kmodel.predict(X_user)[0]

    # Display result
    st.write(f"You are predicted to be in Cluster {label}.")

    # Add information based on clustering analysis
    st.write("Your cluster typically represents individuals who are:")
    if label == 0:
        st.write("Cluster 0: Low income and spending.")
    elif label == 1:
        st.write("Cluster 1: High income and high spending.")
    elif label == 2:
        st.write("Cluster 2: Younger, low to moderate income and moderate to high spending.")
    elif label == 3:
        st.write("Cluster 3: High income but low spending.")
    elif label == 4:
        st.write("Cluster 4: Older, moderate income and spending.")

st.write(
    """I used a machine learning (KMeans clustering) model to predict customer partitioning. The images show the clusters and the determined optimal values for number of clusters"""
)
st.image("scatter_plot_clusters.png") 
st.image("silhouette_plot.png")
st.image("elbow_plot.png")
