# Product Clustering using DBSCAN

This project implements a product clustering system using the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm to identify similar products based on their sales patterns and characteristics.

## Project Overview

The system analyzes products based on several key features:
- Average sales price
- Sales frequency
- Average quantity per order
- Number of unique customers

The goal is to:
- Group similar products together
- Identify products that are rarely sold
- Detect unusual product combinations
- Segment products for better inventory and marketing strategies

## Features

- Automated optimal parameter selection for DBSCAN
- Visualization of product clusters
- Identification of outlier products
- Web interface for viewing clustering results
- Integration with PostgreSQL database

## Requirements

- Python 3.x
- PostgreSQL database
- Required Python packages:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - psycopg2
  - SQLAlchemy
  - fastapi
  - uvicorn
  - kneed

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd DBSCAN
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Set up your PostgreSQL database and update the connection details in the code:
```python
user = 'postgres'
password = "your_password"
host = 'localhost'
port = '5432'
database = 'your_database'
```

## Usage

1. Run the application:
```bash
python sample2.py
```

2. Access the web interface at `http://localhost:8001`

## Project Structure

- `sample2.py`: Main application file containing the DBSCAN implementation and web interface
- `sample1.py`: Additional analysis scripts
- `sample4.py`: Supporting utilities

## Results

The application provides:
- Visual representation of product clusters
- List of outlier products
- Statistical analysis of product groupings
- Interactive web interface for result visualization

## Contributing

Feel free to submit issues and enhancement requests!
