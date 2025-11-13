import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

data = {
    'PassengerId': [1, 2, 3, 4, 5, 6],
    'Sex': ['male', 'female', 'female', 'male', 'male', 'male'],
    'Age': [22, 38, 26, 35, 32, 54],
    'SibSp': [1, 1, 0, 0, 0, 0],
    'Parch': [0, 0, 0, 0, 0, 0],
    'Fare': [7.25, 71.28, 7.92, 8.05, 8.46, 51.86],
    'Embarked': ['S', 'C', 'S', 'S', 'S', 'S']
}

df = pd.DataFrame(data)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

print(" Titanic Dataset Created Successfully!")
print(df)


scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

print("\n Data Standardized!")


kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

df['Cluster'] = clusters
print("\n K-Means Clustering Completed!")
print(df[['PassengerId', 'Cluster']])

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

print("\n PCA Completed!")
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

plt.figure(figsize=(7, 5))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clusters Visualized with PCA")
plt.grid(True)

os.makedirs("plots", exist_ok=True)
plt.savefig("plots/titanic_clusters.png")

print("\n Cluster plot saved to /plots/titanic_clusters.png")


print("\n Project Completed Successfully!")
