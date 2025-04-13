# visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Plot clusters
def plot_clusters(df, kmodel, features, x, y):
    
    # Predict cluster labels
    df['Cluster'] = kmodel.predict(df[features])
    
    # Create scatter plots for each pair of features
    sns.scatterplot(x='Annual_Income', y='Spending_Score', data=df, hue='Cluster', palette='colorblind')
    plt.title('Clusters')
    plt.suptitle('Prediction is based on more than two features, which is why you see some overlap of data points in this scatter plot.', fontsize=7)
    plt.savefig('scatter_plot_clusters.png') 
    plt.close()
    
# Plot the Elbow Plot
def plot_elbow(K, WCSS):
    wss = pd.DataFrame({'cluster':K, 'WSS_Score': WCSS})
    wss.plot(x='cluster', y='WSS_Score')
    plt.xlabel('No. of Clusters')
    plt.ylabel('WCSS Score')
    plt.title('Elbow Plot')
    plt.savefig("elbow_plot.png")
    plt.close()

# Plot silhouette plot
def plot_silhouette(df):
    
    k = range(3,9)
    K = []
    ss = []
    for i in k:
        kmodel = KMeans(n_clusters=i,).fit(df[['Annual_Income','Spending_Score']], )
        ypred = kmodel.labels_
        sil_score = silhouette_score(df[['Annual_Income','Spending_Score']], ypred)
        K.append(i)
        ss.append(sil_score)
    
    # Store the number of clusters and their respective Silhouette scores in a dataframe
    wss = pd.DataFrame({'Cluster': K, 'Silhouette_Score': ss})

    # Plot Silhouette Plot
    wss.plot(x='Cluster', y='Silhouette_Score')
    plt.xlabel('No. of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Plot')
    plt.savefig('silhouette_plot.png')
    plt.close()

    return wss

    

      
    
    








