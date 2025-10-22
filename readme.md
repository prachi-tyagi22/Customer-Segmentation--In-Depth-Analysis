🛍️ Customer Segmentation Analysis using K-Means Clustering
📌 Overview

This project aims to perform Customer Segmentation using the K-Means clustering algorithm on the Mall Customers dataset. The objective is to identify distinct groups of customers based on their age, annual income, and spending score, allowing businesses to tailor marketing strategies for each customer group.

The project is built with Python and deployed using Streamlit, providing an interactive dashboard for visualizing data distributions, outlier detection, optimal cluster selection (Elbow method), and segment profiles.

The project is deployed on streamlit: https://prachi-tyagi22-customer-segmentation--in-depth-analy-app-fpshfn.streamlit.app/

🚀 Features

📊 Interactive Streamlit Dashboard
Explore and visualize customer data with clean, modern UI components.

📈 Elbow Method & Silhouette Score
Identify the optimal number of clusters for the K-Means model.

🧮 Automatic Outlier Detection
Detects outliers in customer data using the IQR (Interquartile Range) method.

🎨 Cluster Visualization with PCA
Visualize customers in a 2D space after dimensionality reduction via PCA.

👥 Cluster Profiling
Automatically generates customer segments with descriptive insights.

🧰 Tech Stack
Component	Technology
Language	Python
Libraries	Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib
Visualization & App	Streamlit
Algorithm	K-Means Clustering
Dimensionality Reduction	PCA (Principal Component Analysis)

📊 Dataset Description

Source: Mall Customer Dataset (Kaggle)
Features Used:

Age

Annual Income (k$)

Spending Score (1–100)

🧠 Learnings

Understanding and applying unsupervised learning.
Using K-Means for real-world segmentation tasks.
Applying PCA for visualization in reduced dimensions.
Building interactive dashboards with Streamlit.
