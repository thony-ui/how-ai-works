"use client";

import React from 'react';
import { Point } from '@/lib/math/k-means';

interface KMeansPyTorchComparisonProps {
  points: Point[];
  numClusters: number;
  maxIterations: number;
  initMethod: 'random' | 'kmeans++';
}

export function KMeansPyTorchComparison({
  points,
  numClusters,
  maxIterations,
  initMethod
}: KMeansPyTorchComparisonProps) {
  
  const generatePythonCode = () => {
    const pointsStr = points.map(p => `[${p.x.toFixed(2)}, ${p.y.toFixed(2)}]`).join(',\n    ');
    
    return `from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Data points
X = np.array([
    ${pointsStr}
])

# Create and fit K-means model
kmeans = KMeans(
    n_clusters=${numClusters},
    max_iter=${maxIterations},
    init='${initMethod === 'kmeans++' ? 'k-means++' : 'random'}',
    n_init=1,
    random_state=42
)

# Fit the model
kmeans.fit(X)

# Get cluster assignments and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_

print(f"Number of iterations: {kmeans.n_iter_}")
print(f"Inertia: {inertia:.2f}")
print(f"\\nCentroids:")
for i, centroid in enumerate(centroids):
    print(f"  Cluster {i}: [{centroid[0]:.2f}, {centroid[1]:.2f}]")
# Predict cluster for new point
new_point = np.array([[0, 0]])
cluster = kmeans.predict(new_point)
print(f"\\nPoint [0, 0] belongs to cluster: {cluster[0]}")`;
  };

  const generateElbowCode = () => {
    return `# Elbow Method to find optimal K
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertias = []
K_range = range(1, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-', 
linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.title('Elbow Method for Optimal K')
plt.grid(True, alpha=0.3)
plt.show()`;
  };

  const handleCopy = (code: string) => {
    navigator.clipboard.writeText(code);
  };

  return (
    <div className="space-y-4">
      <div>
        <div className="flex justify-between items-start mb-2">
          <p className="text-sm text-muted-foreground">
            K-means clustering with scikit-learn:
          </p>
          <button
            onClick={() => handleCopy(generatePythonCode())}
            className="text-xs px-2 py-1 bg-gray-100 hover:bg-gray-200 rounded"
          >
            Copy Code
          </button>
        </div>
        
        <pre className="rounded-lg overflow-x-auto text-xs font-mono min-w-0 max-w-full whitespace-pre-wrap wrap-break-word">
          <code>{generatePythonCode()}</code>
        </pre>
      </div>

      <div>
        <div className="flex justify-between items-start mb-2">
          <p className="text-sm font-semibold">
            Elbow Method:
          </p>
          <button
            onClick={() => handleCopy(generateElbowCode())}
            className="text-xs px-2 py-1 bg-gray-100 hover:bg-gray-200 rounded"
          >
            Copy Code
          </button>
        </div>
        
        <pre className="rounded-lg overflow-x-auto text-xs font-mono min-w-0 max-w-full whitespace-pre-wrap wrap-break-word">
          <code>{generateElbowCode()}</code>
        </pre>
      </div>
      
      <div className="text-xs text-muted-foreground space-y-2">
        <p>
          <strong>Key differences:</strong>
        </p>
        <ul className="list-disc list-inside space-y-1 ml-2">
          <li>Scikit-learn uses optimized implementations for better performance</li>
          <li>The n_init parameter runs the algorithm multiple times with different initializations</li>
          <li>Both use the same Lloyd&apos;s algorithm for K-means clustering</li>
          <li>Results may vary slightly due to random initialization (use random_state for reproducibility)</li>
        </ul>
      </div>
    </div>
  );
}
