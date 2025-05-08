import pandas as pd
from sklearn.cluster import KMeans

def main():
    # Load shot data
    df = pd.read_csv('NBA_2009_Shots.csv')
    coords = df[['LOC_X', 'LOC_Y']].values

    # Run K-Means clustering for 300 centroids
    kmeans = KMeans(n_clusters=300, random_state=42, n_init=10)
    kmeans.fit(coords)

    # Extract cluster centers and counts
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    counts = pd.Series(labels).value_counts().sort_index()

    # Build hotspots DataFrame
    hotspots = pd.DataFrame(centroids, columns=['LOC_X', 'LOC_Y'])
    hotspots['count'] = counts.values

    # Save to CSV
    hotspots.to_csv('shot_centroids2009.csv', index=False)
    print('Wrote shot_centroids.csv with 300 centroids.')

if __name__ == '__main__':
    main() 