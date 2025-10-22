# importing libraries 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# loading dataset
df = pd.read_csv('Mall_Customers.csv')
X= df.copy().drop(['CustomerID','Gender'], axis=1)
print(X.head())
# print first 5 rows of the dataset
print(df.head())
# checking basic information about the dataset
df.info()
# checking for missing values ,data types, and duplicates
print(df.columns)
print(df.dtypes)
print(df.isnull().sum())
print(df.duplicated().sum())

# summary of categorical  columns
if 'Gender' in df.columns:
    print("\n🔹 Gender Distribution:")
    print(df['Gender'].value_counts())

# Exploratory data analysis
print(df.describe()) #for outliers and distribution
def detect_outliers_iqr(df, column):
    Q1 =df[column].quantile(0.25)
    Q3 =df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers =df[(df[column]<lower_bound) | (df[column]>upper_bound)]
    return outliers

for col in X.select_dtypes(include=['int64', 'float64']).columns:
    outliers = detect_outliers_iqr(X, col)
    print(f"\n🔹 Outliers in {col}:")
    print(outliers)

# Distribution plots
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

for feature in features:
    plt.figure(figsize=(7, 4))
    sns.histplot(df[feature], bins=20, color='skyblue')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()


# # Boxplots to check for outliers
plt.figure(figsize=(15, 5))
for i, feature in enumerate(features):
    plt.subplot(1, 3, i+1)
    sns.boxplot(y=df[feature], color='lightcoral')
    plt.title(f'Boxplot of {feature}')
plt.tight_layout()
plt.show()

# # Pairplot to explore relationships
sns.pairplot(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']], diag_kind='kde')
plt.suptitle('Pairwise Relationships Between Features', y=1.02)
plt.show()

# pre processing for clustering
lb = LabelEncoder()
# df['Gender']= lb.fit_transform(df['Gender'])
lb.fit(df['Gender'])
df['Gender'] = lb.transform(df['Gender'])
 

scaler = StandardScaler()
X_scaled= scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns) 
X_scaled_df.head()

# KMeans clustering
sse = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)  # inertia_ is the sum of squared distances to closest cluster center

# Elbow method plot
plt.figure(figsize=(8, 5))
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.plot(k_range, sse, marker='o')
plt.show()

# apply k means with optimal k , here 6
km = KMeans(n_clusters=6, random_state=42)
y_km = km.fit_predict(X_scaled)
X_scaled_df['Cluster'] = y_km
X['Cluster'] = y_km


# check the clustering quality 
score = silhouette_score(X_scaled, y_km)
print(f"Silhouette Score: {score:.2f}")


# visualize by income and spending score
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled_df['Annual Income (k$)'], X_scaled_df['Spending Score (1-100)'], c=X_scaled_df['Cluster'], cmap='tab10', s=100)
centroids = km.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation using KMeans')
plt.legend()
plt.show()



# Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Scatter plot of PCA components with cluster color mapping
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=X_scaled_df['Cluster'], cmap='tab10', s=100)
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Centroids')

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Customer Segmentation using KMeans (k=6) with PCA')
plt.legend()
plt.show()

# Group by cluster and calculate the mean of each feature for every cluster
cluster_profiles =X.groupby('Cluster').mean() 
print(cluster_profiles)


cluster_profiles.plot(kind='bar', figsize=(10, 6))
plt.title('Cluster Profiles: Average Age, Income, and Spending Score')
plt.ylabel('Mean Value')
plt.xticks(rotation=0)
plt.show()

# mapping the clusters to meaningful names based on observed characteristics
cluster_map = {
    0: "Mature, Moderate Spenders",
    1: "Young, Affluent High Spenders",
    2: "Young, Low-Income, Moderate Spenders",
    3: "Young Adults, Budget-Conscious",
    4: "High-Income, Low Spenders",
    5: "Older, Low-Income, Low-Spenders"
}

