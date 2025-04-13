# from setuptools import find_packages, setup


# setup(
#     name='src',
#     packages=find_packages(),
#     version='0.1.0',
#     description='Credit Risk Model code structuring',
#     author='Warner Jawworski',
#     license='',
# )

# main.py
from src.data.load_data import load_data
from src.models.train_model import elbow_method, train_kmodel
from src.visualization.visualize import plot_elbow, plot_silhouette, plot_clusters


if __name__ == "__main__":
    
    # Load and preprocess the data
    data_path = "data/raw/mall_customers.csv"
    df = load_data(data_path)

    # Define features for visualization and clustering
    features = ['Age', 'Annual_Income', 'Spending_Score']
    
    # Find the best K using elbow method
    K, WCSS, wss=elbow_method(df[['Annual_Income', 'Spending_Score']])
    
    # Plot Elow method graph
    plot_elbow(K, WCSS)
    
    # Plot silhouette plot for different numbers of clusters
    silhouette_df = plot_silhouette(df[['Annual_Income', 'Spending_Score']])
    
    # Determine the best k
    K = 5
    
    # Train the KMeans model using all features
    kmodel, ypred = train_kmodel(df, K, features)
    
   # Plot scatter plot of clusters for Annual_Income vs Spending_Score
    plot_clusters(df, kmodel, features, 'Annual_Income', 'Spending_Score')
    
    # print cluster centers
    print("The best number of clusters:", K)
    print("Cluster centroids:", kmodel.cluster_centers_)
    
    
