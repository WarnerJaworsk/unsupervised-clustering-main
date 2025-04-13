# Mall_Customer_Subdivisions_Application

This app has been built using Streamlit and deployed with Streamlit community cloud

[Visit the app here](https://unsupervised-clustering-main-yyscht3vb7ddngbicfwndx.streamlit.app/)

password - streamlit

This application performs customer classification for a mall using the KMeans algorithm. The model helps businesses understand different customer segments based on features like age, annual income, and spending score.

## Overview

The application uses the `mall_customers.csv` dataset to train a KMeans model and segment customers into distinct groups. It leverages machine learning techniques to identify patterns and provide insights for targeted marketing strategies.

## Features

- User-friendly interface powered by Streamlit.
- Input form to explore different customer segments.
- Real-time visualization of customer clusters.
- Accessible via Streamlit Community Cloud.

## Dataset

The application is trained on the **mall_customers.csv** dataset, which includes features like:

- Gender
- Age
- Annual Income (in thousands of $)
- Spending Score (1-100)

## Technologies Used

- **Streamlit**: For building the web application.
- **Scikit-learn**: For model training and evaluation.
- **Pandas** and **NumPy**: For data preprocessing and manipulation.
- **Seaborn** and **Matplotlib**: For exploratory data analysis and visualization.

## Model

The predictive model is trained using the KMeans algorithm. It applies preprocessing steps like creating dummy variables and using elbow and silhouette plots to find the best k value (number of clusters). The model aims to segment customers into clusters based on their age, annual income, and spending score.

## Future Enhancements

- Adding support for additional datasets.
- Incorporating more advanced clustering algorithms.

## Installation (for local deployment)

If you want to run the application locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/WarnerJaworsk/unsupervised-clustering-main.git
   cd unsupervised-clustering-main

   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\\Scripts\\activate`

   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt

   ```

4. Run the Streamlit application:
   ```bash
   streamlit run streamlit.py
   ```

#### Thank you for using the Clustering Application! Feel free to share your feedback.
