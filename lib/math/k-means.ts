// K-Means Clustering implementation

export interface Point {
  x: number;
  y: number;
  cluster?: number;
}

export interface Centroid {
  x: number;
  y: number;
  cluster: number;
}

export interface ClusterResult {
  centroids: Centroid[];
  points: Point[];
  iterations: number;
  inertia: number;
  converged: boolean;
}

export interface IterationStep {
  centroids: Centroid[];
  points: Point[];
  inertia: number;
}

// Calculate Euclidean distance between two points
export function euclideanDistance(p1: Point, p2: Point | Centroid): number {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  return Math.sqrt(dx * dx + dy * dy);
}

// Initialize centroids using k-means++ algorithm
export function initializeCentroidsKMeansPlusPlus(
  points: Point[],
  k: number,
): Centroid[] {
  if (points.length === 0) return [];

  const centroids: Centroid[] = [];

  // Choose first centroid randomly
  const firstIndex = Math.floor(Math.random() * points.length);
  centroids.push({
    x: points[firstIndex].x,
    y: points[firstIndex].y,
    cluster: 0,
  });

  // Choose remaining centroids
  for (let i = 1; i < k; i++) {
    const distances: number[] = [];
    let totalDistance = 0;

    // Calculate distance to nearest centroid for each point
    for (const point of points) {
      let minDist = Infinity;
      for (const centroid of centroids) {
        const dist = euclideanDistance(point, centroid);
        minDist = Math.min(minDist, dist);
      }
      const distSquared = minDist * minDist;
      distances.push(distSquared);
      totalDistance += distSquared;
    }

    // Choose next centroid with probability proportional to distance squared
    let random = Math.random() * totalDistance;
    let chosenIndex = 0;

    for (let j = 0; j < distances.length; j++) {
      random -= distances[j];
      if (random <= 0) {
        chosenIndex = j;
        break;
      }
    }

    centroids.push({
      x: points[chosenIndex].x,
      y: points[chosenIndex].y,
      cluster: i,
    });
  }

  return centroids;
}

// Initialize centroids randomly
export function initializeCentroidsRandom(
  points: Point[],
  k: number,
): Centroid[] {
  const centroids: Centroid[] = [];
  const usedIndices = new Set<number>();

  for (let i = 0; i < k; i++) {
    let index: number;
    do {
      index = Math.floor(Math.random() * points.length);
    } while (usedIndices.has(index));

    usedIndices.add(index);
    centroids.push({
      x: points[index].x,
      y: points[index].y,
      cluster: i,
    });
  }

  return centroids;
}

// Assign each point to nearest centroid
export function assignPointsToClusters(
  points: Point[],
  centroids: Centroid[],
): Point[] {
  return points.map((point) => {
    let minDist = Infinity;
    let closestCluster = 0;

    centroids.forEach((centroid, index) => {
      const dist = euclideanDistance(point, centroid);
      if (dist < minDist) {
        minDist = dist;
        closestCluster = index;
      }
    });

    return { ...point, cluster: closestCluster };
  });
}

// Update centroid positions based on assigned points
export function updateCentroids(points: Point[], k: number): Centroid[] {
  const centroids: Centroid[] = [];

  for (let i = 0; i < k; i++) {
    const clusterPoints = points.filter((p) => p.cluster === i);

    if (clusterPoints.length === 0) {
      // If no points assigned to cluster, keep previous position or random
      centroids.push({
        x: Math.random() * 10 - 5,
        y: Math.random() * 10 - 5,
        cluster: i,
      });
    } else {
      const sumX = clusterPoints.reduce((sum, p) => sum + p.x, 0);
      const sumY = clusterPoints.reduce((sum, p) => sum + p.y, 0);

      centroids.push({
        x: sumX / clusterPoints.length,
        y: sumY / clusterPoints.length,
        cluster: i,
      });
    }
  }

  return centroids;
}

// Calculate inertia (sum of squared distances to nearest centroid)
export function calculateInertia(
  points: Point[],
  centroids: Centroid[],
): number {
  let inertia = 0;

  for (const point of points) {
    if (point.cluster !== undefined) {
      const centroid = centroids[point.cluster];
      const dist = euclideanDistance(point, centroid);
      inertia += dist * dist;
    }
  }

  return inertia;
}

// Check if centroids have converged
export function haveCentroidsConverged(
  oldCentroids: Centroid[],
  newCentroids: Centroid[],
  tolerance: number = 1e-4,
): boolean {
  if (oldCentroids.length !== newCentroids.length) return false;

  for (let i = 0; i < oldCentroids.length; i++) {
    const dist = Math.sqrt(
      Math.pow(oldCentroids[i].x - newCentroids[i].x, 2) +
        Math.pow(oldCentroids[i].y - newCentroids[i].y, 2),
    );

    if (dist > tolerance) return false;
  }

  return true;
}

// Run k-means clustering algorithm
export function runKMeans(
  points: Point[],
  k: number,
  maxIterations: number = 100,
  initMethod: "random" | "kmeans++" = "kmeans++",
  tolerance: number = 1e-4,
): ClusterResult {
  if (points.length === 0 || k <= 0) {
    return {
      centroids: [],
      points: [],
      iterations: 0,
      inertia: 0,
      converged: false,
    };
  }

  // Initialize centroids
  let centroids =
    initMethod === "kmeans++"
      ? initializeCentroidsKMeansPlusPlus(points, k)
      : initializeCentroidsRandom(points, k);

  let assignedPoints = points;
  let iterations = 0;
  let converged = false;

  for (let iter = 0; iter < maxIterations; iter++) {
    iterations++;

    // Assign points to clusters
    assignedPoints = assignPointsToClusters(assignedPoints, centroids);

    // Update centroids
    const newCentroids = updateCentroids(assignedPoints, k);

    // Check convergence
    if (haveCentroidsConverged(centroids, newCentroids, tolerance)) {
      converged = true;
      centroids = newCentroids;
      break;
    }

    centroids = newCentroids;
  }

  const inertia = calculateInertia(assignedPoints, centroids);

  return {
    centroids,
    points: assignedPoints,
    iterations,
    inertia,
    converged,
  };
}

// Run k-means and capture all iteration steps
export function runKMeansWithSteps(
  points: Point[],
  k: number,
  maxIterations: number = 100,
  initMethod: "random" | "kmeans++" = "kmeans++",
): IterationStep[] {
  if (points.length === 0 || k <= 0) {
    return [];
  }

  const steps: IterationStep[] = [];

  // Initialize centroids
  let centroids =
    initMethod === "kmeans++"
      ? initializeCentroidsKMeansPlusPlus(points, k)
      : initializeCentroidsRandom(points, k);

  let assignedPoints = assignPointsToClusters(points, centroids);

  // Record initial state
  steps.push({
    centroids: JSON.parse(JSON.stringify(centroids)),
    points: JSON.parse(JSON.stringify(assignedPoints)),
    inertia: calculateInertia(assignedPoints, centroids),
  });

  for (let iter = 0; iter < maxIterations; iter++) {
    // Update centroids
    const newCentroids = updateCentroids(assignedPoints, k);

    // Assign points to new clusters
    assignedPoints = assignPointsToClusters(assignedPoints, newCentroids);

    // Record step
    steps.push({
      centroids: JSON.parse(JSON.stringify(newCentroids)),
      points: JSON.parse(JSON.stringify(assignedPoints)),
      inertia: calculateInertia(assignedPoints, newCentroids),
    });

    // Check convergence
    if (haveCentroidsConverged(centroids, newCentroids, 1e-4)) {
      break;
    }

    centroids = newCentroids;
  }

  return steps;
}

// Elbow method: calculate inertia for different k values
export function calculateElbowMethod(
  points: Point[],
  maxK: number = 10,
): { k: number; inertia: number }[] {
  const results: { k: number; inertia: number }[] = [];

  for (let k = 1; k <= maxK; k++) {
    const result = runKMeans(points, k, 100, "kmeans++");
    results.push({ k, inertia: result.inertia });
  }

  return results;
}

// Silhouette score for a single point
export function calculatePointSilhouette(
  point: Point,
  points: Point[],
  centroids: Centroid[],
): number {
  if (point.cluster === undefined || centroids.length <= 1) return 0;

  const sameCluster = points.filter(
    (p) => p.cluster === point.cluster && p !== point,
  );

  if (sameCluster.length === 0) return 0;

  // Average distance to points in same cluster
  const a =
    sameCluster.reduce((sum, p) => sum + euclideanDistance(point, p), 0) /
    sameCluster.length;

  // Average distance to points in nearest other cluster
  let minB = Infinity;
  for (let i = 0; i < centroids.length; i++) {
    if (i === point.cluster) continue;

    const clusterPoints = points.filter((p) => p.cluster === i);
    if (clusterPoints.length === 0) continue;

    const b =
      clusterPoints.reduce((sum, p) => sum + euclideanDistance(point, p), 0) /
      clusterPoints.length;
    minB = Math.min(minB, b);
  }

  if (minB === Infinity) return 0;

  return (minB - a) / Math.max(a, minB);
}

// Calculate average silhouette score
export function calculateSilhouetteScore(
  points: Point[],
  centroids: Centroid[],
): number {
  if (points.length === 0 || centroids.length <= 1) return 0;

  const scores = points.map((point) =>
    calculatePointSilhouette(point, points, centroids),
  );
  return scores.reduce((sum, score) => sum + score, 0) / scores.length;
}

// Generate sample data with clusters
export function generateClusteredData(
  numClusters: number = 3,
  pointsPerCluster: number = 30,
): Point[] {
  const points: Point[] = [];

  // Generate cluster centers
  const clusterCenters: { x: number; y: number }[] = [];
  for (let i = 0; i < numClusters; i++) {
    clusterCenters.push({
      x: (Math.random() - 0.5) * 8,
      y: (Math.random() - 0.5) * 8,
    });
  }

  // Generate points around each center
  for (let i = 0; i < numClusters; i++) {
    for (let j = 0; j < pointsPerCluster; j++) {
      const angle = Math.random() * 2 * Math.PI;
      const radius = Math.random() * 1.5;

      points.push({
        x: clusterCenters[i].x + radius * Math.cos(angle),
        y: clusterCenters[i].y + radius * Math.sin(angle),
      });
    }
  }

  return points;
}
