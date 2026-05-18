
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Retail Customer Segmentation",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
    }

    .title {
        font-size: 42px;
        font-weight: bold;
        color: white;
        text-align: center;
        padding: 10px;
    }

    .subtitle {
        font-size: 18px;
        color: #cbd5e1;
        text-align: center;
        margin-bottom: 30px;
    }

    .metric-box {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- TITLE ----------------

st.markdown('<div class="title">📊 Retail Customer Segmentation Dashboard</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">K-Means Clustering using Quantity and UnitPrice</div>',
    unsafe_allow_html=True
)

# ---------------- FILE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Upload Online Retail Dataset (.xlsx or .csv)",
    type=["xlsx", "csv"]
)

if uploaded_file:

    # ---------------- LOAD DATA ----------------

    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded Successfully ✅")

    # ---------------- DATA PREVIEW ----------------

    with st.expander("🔍 View Dataset"):
        st.dataframe(df.head())

    # ---------------- CLEAN DATA ----------------

    df = df.dropna(subset=['Quantity', 'UnitPrice'])

    # Remove outliers using IQR
    Q1 = df[['Quantity', 'UnitPrice']].quantile(0.25)
    Q3 = df[['Quantity', 'UnitPrice']].quantile(0.75)

    IQR = Q3 - Q1

    df = df[
        ~((df[['Quantity', 'UnitPrice']] < (Q1 - 1.5 * IQR)) |
          (df[['Quantity', 'UnitPrice']] > (Q3 + 1.5 * IQR))).any(axis=1)
    ]

    # ---------------- FEATURE SELECTION ----------------

    X = df[['Quantity', 'UnitPrice']]

    # ---------------- SCALING ----------------

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------- SIDEBAR ----------------

    st.sidebar.title("⚙️ Clustering Settings")

    k = st.sidebar.slider(
        "Select Number of Clusters (K)",
        min_value=2,
        max_value=10,
        value=3
    )

    # ---------------- TRAIN MODEL ----------------

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X_scaled)

    df['Cluster'] = clusters

    # ---------------- METRICS ----------------

    inertia = kmeans.inertia_

    # Use sample for silhouette score
    sample_size = min(10000, len(X_scaled))

    indices = np.random.choice(len(X_scaled), sample_size, replace=False)

    sample_X = X_scaled[indices]
    sample_clusters = clusters[indices]

    silhouette = silhouette_score(sample_X, sample_clusters)

    # ---------------- KPI SECTION ----------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📦 Total Records", len(df))

    with col2:
        st.metric("🎯 Inertia", f"{inertia:,.2f}")

    with col3:
        st.metric("⭐ Silhouette Score", f"{silhouette:.3f}")

    st.markdown("---")

    # ---------------- CLUSTER VISUALIZATION ----------------

    st.subheader("📈 K-Means Cluster Visualization")

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=clusters,
        cmap='viridis',
        alpha=0.7
    )

    centers = kmeans.cluster_centers_

    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        s=300,
        c='red',
        marker='X',
        label='Centroids'
    )

    ax.set_title('K-Means Clustering')
    ax.set_xlabel('Scaled Quantity')
    ax.set_ylabel('Scaled UnitPrice')
    ax.legend()

    st.pyplot(fig)

    # ---------------- ELBOW METHOD ----------------

    st.subheader("📉 Elbow Method")

    wcss = []

    for i in range(1, 11):
        km = KMeans(n_clusters=i, random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)

    fig2, ax2 = plt.subplots(figsize=(8, 5))

    ax2.plot(range(1, 11), wcss, marker='o')

    ax2.set_title('Elbow Method')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('WCSS / Inertia')

    st.pyplot(fig2)

    # ---------------- CLUSTER DISTRIBUTION ----------------

    st.subheader("📊 Cluster Distribution")

    cluster_counts = df['Cluster'].value_counts().sort_index()

    fig3, ax3 = plt.subplots(figsize=(8, 5))

    ax3.bar(cluster_counts.index.astype(str), cluster_counts.values)

    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Count')
    ax3.set_title('Number of Records per Cluster')

    st.pyplot(fig3)

    # ---------------- CLUSTER SUMMARY ----------------

    st.subheader("📋 Cluster Summary")

    summary = df.groupby('Cluster')[['Quantity', 'UnitPrice']].mean()

    st.dataframe(summary)

    # ---------------- DOWNLOAD ----------------

    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="⬇️ Download Clustered Dataset",
        data=csv,
        file_name='clustered_online_retail.csv',
        mime='text/csv'
    )

else:
    st.info("👆 Upload the Online Retail dataset to begin clustering")

