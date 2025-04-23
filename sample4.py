"""
Problem 4: Sales Pattern Analysis by Country
Database tables: Customers, Orders, OrderDetails

Question:

"Cluster orders from different countries using DBSCAN.
Identify countries with unusual order patterns."

Features:

Total orders

Average order amount

Number of products per order
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import io
import base64

app = FastAPI()

# Database connection information
user = 'postgres'
password = "2630"
host = 'localhost'
port = '5432'
database = 'Gyk1'

engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}')

# SQL query to fetch order features by country - only the requested 3 features
query = """
SELECT 
    c.country,
    COUNT(DISTINCT o.order_id) as total_orders,
    AVG(od.unit_price * od.quantity) as average_order_amount,
    AVG(od.quantity) as avg_products_per_order
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_details od ON o.order_id = od.order_id
GROUP BY c.country
"""

df = pd.read_sql_query(query, engine)
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Feature matrix preparation and scaling - only the requested 3 features
X = df[['total_orders', 'average_order_amount', 'avg_products_per_order']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to find optimal eps value
def find_optimal_eps(X_scaled, min_samples=3):
    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
    distances, _ = neighbors.kneighbors(X_scaled)
    
    distances = np.sort(distances[:, min_samples-1])
    
    kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
    optimal_eps = distances[kneedle.elbow]
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.axvline(x=kneedle.elbow, color='r', linestyle='--', label=f'Optimal eps: {optimal_eps:.2f}')
    plt.xlabel('Points Sorted by Distance')
    plt.ylabel(f'{min_samples}-th Nearest Neighbor Distance')
    plt.title('Elbow Method for Optimal Eps')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return optimal_eps

# Function to find optimal min_samples value
def find_optimal_min_samples(X_scaled, min_samples_range=range(2, 11)):
    """
    Find optimal min_samples for DBSCAN by analyzing the trade-off between 
    number of clusters and noise points.
    """
    n_clusters_list = []
    n_noise_list = []
    
    # Test different min_samples values
    for min_samples in min_samples_range:
        dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        noise_ratio = n_noise / len(labels)
        
        n_clusters_list.append(n_clusters)
        n_noise_list.append(noise_ratio)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(min_samples_range, n_clusters_list, 'bo-')
    ax1.set_xlabel('Min Samples')
    ax1.set_ylabel('Number of Clusters')
    ax1.set_title('Min Samples vs Number of Clusters')
    ax1.grid(True)

    ax2.plot(min_samples_range, n_noise_list, 'ro-')
    ax2.set_xlabel('Min Samples')
    ax2.set_ylabel('Noise Points Ratio')
    ax2.set_title('Min Samples vs Noise Points Ratio')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Print results for each min_samples value
    print("\nCluster counts for different min_samples values:")
    for ms, nc, nr in zip(min_samples_range, n_clusters_list, n_noise_list):
        print(f"min_samples={ms}: {nc} clusters, {nr:.2%} noise points")

    # Find optimal min_samples using custom criteria
    best_score = float('inf')
    optimal_min_samples = min_samples_range[0]
    
    for i, min_samples in enumerate(min_samples_range):
        noise_ratio = n_noise_list[i]
        n_clusters = n_clusters_list[i]
        
        # Penalize based on:
        # 1. Too high noise ratio (>20%)
        # 2. Too few clusters (<2)
        # 3. Too many clusters (>4)
        score = noise_ratio * 2  # Weight for noise ratio
        if n_clusters < 2:
            score += 0.5  # Penalty for too few clusters
        if n_clusters > 4:
            score += 0.3  # Penalty for too many clusters
            
        if score < best_score:
            best_score = score
            optimal_min_samples = min_samples

    print(f"\nSelected optimal min_samples={optimal_min_samples}")
    idx = min_samples_range.index(optimal_min_samples)
    print(f"This results in {n_clusters_list[idx]} clusters")
    print(f"and {n_noise_list[idx]:.2%} noise points")
    
    return optimal_min_samples

# Find optimal parameters
optimal_eps = find_optimal_eps(X_scaled)
optimal_min_samples = find_optimal_min_samples(X_scaled)

print(f"\nOptimal eps value: {optimal_eps:.3f}")
print(f"Optimal min_samples value: {optimal_min_samples}")

# Apply DBSCAN algorithm
dbscan = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples)
df['cluster'] = dbscan.fit_predict(X_scaled)

# Visualization of results - 3 different visualizations for 3 features
# 1. Total Orders vs Average Order Amount
plt.figure(figsize=(12, 8))
plt.scatter(df['total_orders'], df['average_order_amount'], 
           c=df['cluster'], cmap='plasma', s=100)
for i, txt in enumerate(df['country']):
    plt.annotate(txt, (df['total_orders'].iloc[i], df['average_order_amount'].iloc[i]))
plt.xlabel("Total Orders")
plt.ylabel("Average Order Amount")
plt.title("Country Order Patterns (Total Orders vs Average Amount)")
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# 2. Total Orders vs Products per Order
plt.figure(figsize=(12, 8))
plt.scatter(df['total_orders'], df['avg_products_per_order'], 
           c=df['cluster'], cmap='plasma', s=100)
for i, txt in enumerate(df['country']):
    plt.annotate(txt, (df['total_orders'].iloc[i], df['avg_products_per_order'].iloc[i]))
plt.xlabel("Total Orders")
plt.ylabel("Average Products per Order")
plt.title("Country Order Patterns (Total Orders vs Products per Order)")
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# 3. Average Order Amount vs Products per Order
plt.figure(figsize=(12, 8))
plt.scatter(df['average_order_amount'], df['avg_products_per_order'], 
           c=df['cluster'], cmap='plasma', s=100)
for i, txt in enumerate(df['country']):
    plt.annotate(txt, (df['average_order_amount'].iloc[i], df['avg_products_per_order'].iloc[i]))
plt.xlabel("Average Order Amount")
plt.ylabel("Average Products per Order")
plt.title("Country Order Patterns (Amount vs Products per Order)")
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# Display cluster statistics
print("\nCluster statistics:")
for cluster in sorted(df['cluster'].unique()):
    cluster_data = df[df['cluster'] == cluster]
    print(f"\nCluster {cluster}:")
    print(f"Countries in this cluster: {', '.join(cluster_data['country'].tolist())}")
    print(f"Average statistics:")
    print(f"- Total orders: {cluster_data['total_orders'].mean():.2f}")
    print(f"- Order amount: {cluster_data['average_order_amount'].mean():.2f}")
    print(f"- Products per order: {cluster_data['avg_products_per_order'].mean():.2f}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Create a new figure for the web display
    plt.figure(figsize=(12, 8))
    plt.scatter(df['total_orders'], df['average_order_amount'], 
               c=df['cluster'], cmap='plasma', s=100)
    for i, txt in enumerate(df['country']):
        plt.annotate(txt, (df['total_orders'].iloc[i], df['average_order_amount'].iloc[i]))
    plt.xlabel("Total Orders")
    plt.ylabel("Average Order Amount")
    plt.title("Country Order Patterns (Total Orders vs Average Amount)")
    plt.colorbar(label='Cluster')
    plt.grid(True)
    
    # Convert plot to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    
    # Create HTML content
    html_content = f"""
    <html>
        <head>
            <title>Country Sales Pattern Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                .stats {{ margin: 20px 0; padding: 20px; background-color: #f5f5f5; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Country Sales Pattern Analysis</h1>
                <div class="plot">
                    <img src="data:image/png;base64,{img_str}" alt="Cluster Plot">
                </div>
                <div class="stats">
                    <h2>Cluster Statistics</h2>
                    {''.join([f'''
                    <div class="cluster">
                        <h3>Cluster {cluster}</h3>
                        <p>Countries: {', '.join(df[df['cluster'] == cluster]['country'].tolist())}</p>
                        <p>Average Statistics:</p>
                        <ul>
                            <li>Total orders: {df[df['cluster'] == cluster]['total_orders'].mean():.2f}</li>
                            <li>Order amount: {df[df['cluster'] == cluster]['average_order_amount'].mean():.2f}</li>
                            <li>Products per order: {df[df['cluster'] == cluster]['avg_products_per_order'].mean():.2f}</li>
                        </ul>
                    </div>
                    ''' for cluster in sorted(df['cluster'].unique())])}
                </div>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)