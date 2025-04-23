# Customer grouping based on shopping behavior and outlier detection

# Join these three tables: order_details, customers, orders

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import io
import base64

# Create FastAPI app instance
app = FastAPI(
    title="Customer Segmentation",
    description="Customer segmentation using DBSCAN clustering",
    version="1.0.0"
)

# Database connection details
user = 'postgres'
password = "2630"
host = 'localhost'
port = '5432'
database = 'Gyk1'

engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}')

# SQL query to get customer metrics
query ="""
select c.customer_id, 
		count(o.order_id) as total_orders, 
		sum(od.quantity * od.unit_price) as total_spent,
		avg(od.quantity * od.unit_price)as avg_order_value
from customers c inner join orders o
on c.customer_id = o.customer_id
inner join order_details od
on o.order_id = od.order_id
group by c.customer_id
having count(o.order_id)>0
"""

df = pd.read_sql_query(query, engine)
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Feature matrix preparation and scaling
X = df[['total_orders', 'total_spent', 'avg_order_value']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#optimize eps
def find_optimal_eps(X_scaled,min_samples=3):
    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
    distances,_ =neighbors.kneighbors(X_scaled)

    distances = np.sort(distances[:, min_samples-1])

    kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
    optimal_eps = distances[kneedle.elbow]

    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.axvline(x=kneedle.elbow, color='r', linestyle='--', label=f'Optimal eps: {optimal_eps:.2f}')
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{min_samples}-th nearest neighbor distance')
    plt.title('Elbow Method for Optimal eps')
    plt.legend()
    plt.grid(True)
    plt.show()

    return optimal_eps

optimal_eps = find_optimal_eps(X_scaled)

#optimize min_samples
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

optimal_min_samples = find_optimal_min_samples(X_scaled)

print(f"\nOptimal eps değeri: {optimal_eps:.3f}")
print(f"Optimal min_samples değeri: {optimal_min_samples}")


dbscan = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples)
df['cluster'] = dbscan.fit_predict(X_scaled)


plt.figure(figsize=(10, 6))
plt.scatter(df['total_orders'], df['total_spent'], c=df['cluster'], cmap='plasma', s=60)
plt.xlabel("Total Number of Orders")
plt.ylabel("Total Amount Spent")
plt.title("Customer Segmentation (DBSCAN)")
plt.grid(True)
plt.colorbar(label='Cluster')
plt.show()

# Analyze outliers
outliers = df[df['cluster'] == -1]
print("\nNumber of outliers:", len(outliers))
print("\nOutlier details:")
print(outliers[["customer_id", "total_orders", "total_spent"]])

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Convert the last plot to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    
    html_content = f"""
    <html>
        <head>
            <title>Customer Segmentation Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .plot {{ text-align: center; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Customer Segmentation Analysis</h1>
                <div class="plot">
                    <img src="data:image/png;base64,{img_str}" alt="Cluster Plot">
                </div>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/outliers/")
async def get_outliers():
    """Get outlier customers"""
    outliers = df[df['cluster'] == -1]
    return {
        "count": len(outliers),
        "outliers": outliers[["customer_id", "total_orders", "total_spent"]].to_dict('records')
    }

@app.get("/clusters/")
async def get_clusters():
    """Get cluster information"""
    cluster_info = df.groupby('cluster').agg({
        'customer_id': 'count',
        'total_orders': 'mean',
        'total_spent': 'mean'
    }).reset_index()
    return cluster_info.to_dict('records')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)