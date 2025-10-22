# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import streamlit as st

st.set_page_config(
    page_title="Customer Segmentation- In-Depth Analysis",
    layout="wide",)

df = pd.read_csv('Mall_Customers.csv')
st.title("Customer Segmentation Analysis")
X= df.copy().drop(['CustomerID','Gender'], axis=1)
st.markdown(
    "<h2 style='font-size:28px;'>Customer Data</h2>", 
    unsafe_allow_html=True
)
st.write(X.head())

# st.subheader("Insights & Marketing Strategy")

# Plot for data
st.subheader("Data Distribution Plots")
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
col1, col2,col3 = st.columns([0.5,2,0.5])
with col2:
    for feature in features:
        fig, ax = plt.subplots(figsize=(5, 2))
        sns.histplot(df[feature], bins=20, color='skyblue')
        ax.set_title(f'Distribution of {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        st.write(f'Distribution of {feature}')
        st.pyplot(fig)
        print('\n\n\n')  # for better readability in console


st.markdown(
    "<h3 style='font-size:28px;'>Outliers in the data</h3>", 
    unsafe_allow_html=True
)
# # Boxplots to check for outliers
fig1, axes = plt.subplots(1,3,figsize=(15, 5)) # Create 1 row, 3 columns
for i, feature in enumerate(features):
    sns.boxplot(y=df[feature], color='lightcoral', ax=axes[i])
    axes[i].set_title(f'Boxplot of {feature}')
    axes[i].set_ylabel(feature)
plt.tight_layout()
st.pyplot(fig1)

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
    if not outliers.empty:
        st.write(f"\n🔹 Outliers in {col}:")
        st.write(outliers)
    else:
        st.write(f"\n🔹 No outliers detected in {col}.")

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
# Elbow method plot
fig3, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_range, sse, marker='o')
ax.set_xlabel('Number of clusters (k)')
ax.set_ylabel('Sum of Squared Errors (SSE)')

st.subheader("Elbow Method for Optimal k")

km = KMeans(n_clusters=6, random_state=42)
y_km = km.fit_predict(X_scaled)
X_scaled_df['Cluster'] = y_km
X['Cluster'] = y_km


# check the clustering quality 
score = silhouette_score(X_scaled, y_km)
col1, col2 = st.columns([3,1])
with col1:
    st.pyplot(fig3)

with col2:
    st.markdown(
        "<p style='font-size: 20px; line-height:1.6;'>As it can be seen the optimal value for k is 6 where the elbow occurs.<br> To make it more robust we can also check the silhouette score for different values of k.</p>"
        "<br>"
        "<p style='font-size: 16px; line-height:1.6;'>The silhouette score measures how similar an object is to its own cluster compared to other clusters. A higher silhouette score indicates better-defined clusters. The score ranges from -1 to 1, where: <br><li> A value close to 1 suggests that the data points are well clustered</li> " 
        "<li>a value of 0 indicates overlapping clusters</li>" 
        "<li>Negative values imply that data points may have been assigned to the wrong cluster.</li></p>",
        unsafe_allow_html=True)
    st.markdown(f"<p style = 'font-size: 20px;'>Silhouette Score: {score:.2f}</b></p>"
            "<br>", unsafe_allow_html=True)
    

st.subheader("Cluster Visualizations") 


col1, col2, col3 = st.columns([2.25,0.5,2.25])
# visualize by income and spending score
fig5 , ax = plt.subplots(figsize=(10, 6))
ax.scatter(X_scaled_df['Annual Income (k$)'], X_scaled_df['Spending Score (1-100)'], c=X_scaled_df['Cluster'], cmap='tab10', s=100)
centroids = km.cluster_centers_
ax.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Centroids')
ax.set_xlabel('Annual Income (k$)')
ax.set_ylabel('Spending Score (1-100)')
ax.set_title('Customer Segmentation using KMeans')
with col1:
    plt.legend()
    st.pyplot(fig5)

# Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# Scatter plot of PCA components with cluster color mapping
with col3:
    fig4 , ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=X_scaled_df['Cluster'], cmap='tab10', s=100)
    ax.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Centroids')

    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('Customer Segmentation using KMeans (k=6) with PCA')
    plt.legend()
    st.pyplot(fig4)

st.subheader('Cluster Profiles')
cluster_profiles =X.groupby('Cluster').mean() 
st.write(cluster_profiles)
col1, col2, col3 = st.columns([0.25,2.5,0.25])
with col2:
    fig6 , ax = plt.subplots(figsize=(10, 6))
    cluster_profiles.plot(kind='bar', ax=ax)
    ax.set_title('Cluster Profiles: Average Age, Income, and Spending Score')   
    ax.set_ylabel('Mean Value')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    st.pyplot(fig6)

st.markdown(
    "<br><h3 style='font-size:28px;'>Cluster Interpretation & Marketing Strategies</h3>", 
    unsafe_allow_html=True
)
col1, col2,col3 = st.columns([2.25,0.5,2.25])
with col1:
    st.markdown(
        "<h4 style='font-size:22px;'>Cluster Interpretation</h4>", 
        unsafe_allow_html=True
    )
    st.markdown(
    """
    <div style='background-color:white; padding:10px; border-radius:10px; margin-bottom:10px;color:black;'>
        <p style='font-size:16px;'><b>Cluster 0 – Mature, Mid-Income Spenders</b></p>
        <ul style='font-size:15px;'>
            <li>Age: 56.33 (older age group)</li>
            <li>Income: 54.27 k$</li>
            <li>Spending Score: 49.07 (average spender)</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True)
    st.markdown(
    """
    <div style='background-color:white; padding:10px; border-radius:10px; margin-bottom:10px;color:black;'>
        <p style='font-size:16px;'><b>Cluster 1 – Young, Affluent High Spenders</b></p>
        <ul style='font-size:15px;'>
            <li>Age: 32.69 (younger demographic)</li>
            <li>Income: 86.54 k$</li>
            <li>Spending Score: 82.13 (high spender)</li>
        </ul>       
    </div>""" ,
    unsafe_allow_html=True)
    st.markdown(
    """ 
    <div style='background-color:white; padding:10px; border-radius:10px; margin-bottom:10px;color:black;'>
        <p style='font-size:16px;'><b>Cluster 2 – Young, Low-Income, Moderate Spenders</b></p>
        <ul style='font-size:15px;'>
            <li>Age: 26.44 (youngest group)</li>
            <li>Income: 26.12 k$</li>
            <li>Spending Score: 50.22 (moderate spender)</li>
        </ul>
    </div>""",
    unsafe_allow_html=True)
    st.markdown(
    """
    <div style='background-color:white; padding:10px; border-radius:10px; margin-bottom:10px;color:black;'>
        <p style='font-size:16px;'><b>Cluster 3 – Young Adults, Budget-Conscious</b></p>
        <ul style='font-size:15px;'>
            <li>Age: 30.20 (young adults)</li>
            <li>Income: 48.08 k$</li>
            <li>Spending Score: 20.96 (low spender)</li>
        </ul>
    </div>""",
    unsafe_allow_html=True)
    st.markdown(
    """
    <div style='background-color:white; padding:10px; border-radius:10px; margin-bottom:10px;color:black;'>
        <p style='font-size:16px;'><b>Cluster 4 – High-Income, Low Spenders</b></p>
        <ul style='font-size:15px;'>
            <li>Age: 40.91 (middle-aged)</li>
            <li>Income: 88.64 k$</li>
            <li>Spending Score: 17.54 (low spender)</li>
        </ul>
    </div>""",
    unsafe_allow_html=True)
    st.markdown(
    """
    <div style='background-color:white; padding:10px; border-radius:10px; margin-bottom:10px;color:black;'>
        <p style='font-size:16px;'><b>Cluster 5 – Older, Low-Income, Low Spenders</b></p>
        <ul style='font-size:15px;'>
            <li>Age: 50.46 (older demographic)</li>
            <li>Income: 26.95 k$</li>
            <li>Spending Score: 15.27 (low spender)</li>
        </ul>
    </div>""",
    unsafe_allow_html=True)

with col3:
    st.markdown(
        "<h4 style='font-size:22px;'>Marketing Strategies</h4>", 
        unsafe_allow_html=True
    )
    st.markdown(
    """
    <div style='background-color:white; padding:10px; border-radius:10px; margin-bottom:10px;color:black;'>
        <p style='font-size:16px;'><b>Cluster 0 – Mature, Mid-Income Spenders</b></p>
        <ul style='font-size:15px;'>
            <li>Promote loyalty programs and exclusive offers to retain this stable customer base.</li>
            <li>Focus on products that emphasize quality and value for money.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True)
    st.markdown(
    """
    <div style='background-color:white; padding:10px; border-radius:10px; margin-bottom:10px;color:black;'>
        <p style='font-size:16px;'><b>Cluster 1 – Young, Affluent High Spenders</b></p>
        <ul style='font-size:15px;'>
            <li>Target with premium products and personalized shopping experiences.</li>
            <li>Utilize social media and influencer marketing to engage this tech-savvy group.</li>
        </ul>       
    </div>""" ,
    unsafe_allow_html=True)
    st.markdown(
    """
    <div style='background-color:white; padding:10px; border-radius:10px; margin-bottom:10px;color:black;'>
        <p style='font-size:16px;'><b>Cluster 2 – Young, Low-Income, Moderate Spenders</b></p>
        <ul style='font-size:15px;'>
            <li>Offer budget-friendly options and discounts to attract this price-sensitive group.</li>
            <li>Focus on trendy yet affordable products to appeal to their preferences.</li>
        </ul>
    </div>""",
    unsafe_allow_html=True)
    st.markdown(
    """ 
    <div style='background-color:white; padding:10px; border-radius:10px; margin-bottom:10px;color:black;'>
        <p style='font-size:16px;'><b>Cluster 3 – Young Adults, Budget-Conscious</b></p>
        <ul style='font-size:15px;'>
            <li>Emphasize value deals and bundle offers to encourage higher spending.</li>
            <li>Leverage social media campaigns to reach and engage this demographic.</li>
        </ul>
    </div>""",
    unsafe_allow_html=True)
    st.markdown(
    """
    <div style='background-color:white; padding:10px; border-radius:10px; margin-bottom:10px;color:black;'>
        <p style='font-size:16px;'><b>Cluster 4 – High-Income, Low Spenders</b></p>
        <ul style='font-size:15px;'>
            <li>Introduce exclusive, high-end products to entice spending.</li>
            <li>Offer personalized services and experiences to enhance perceived value.</li>
        </ul>
    </div>""",
    unsafe_allow_html=True)
    st.markdown(
    """
    <div style='background-color:white; padding:10px; border-radius:10px; margin-bottom:10px;color:black;'>
        <p style='font-size:16px;'><b>Cluster 5 – Older, Low-Income, Low Spenders</b></p>
        <ul style='font-size:15px;'>
            <li>Focus on essential products and value-for-money offerings.</li>
            <li>Utilize traditional marketing channels like email and direct mail for outreach.</li>
        </ul>
    </div>""",
    unsafe_allow_html=True)




