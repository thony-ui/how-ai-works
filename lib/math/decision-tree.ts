// Decision Tree implementation for classification
// Using ID3 algorithm with information gain and entropy

export interface DataPoint {
  features: number[];
  label: number;
}

export interface TreeNode {
  featureIndex?: number;
  threshold?: number;
  value?: number;
  left?: TreeNode;
  right?: TreeNode;
  entropy?: number;
  samples?: number;
}

export interface TreeParams {
  maxDepth: number;
  minSamplesSplit: number;
  minSamplesLeaf: number;
}

export interface EntropyStep {
  step: number;
  description: string;
  formula: string;
  values: number[];
  result?: number;
}

export interface SplitCalculation {
  featureIndex: number;
  threshold: number;
  parentEntropy: number;
  leftEntropy: number;
  rightEntropy: number;
  leftWeight: number;
  rightWeight: number;
  weightedEntropy: number;
  informationGain: number;
  leftLabels: number[];
  rightLabels: number[];
  parentLabels: number[];
}

// Calculate entropy (Shannon entropy)
export function calculateEntropy(labels: number[]): number {
  if (labels.length === 0) return 0;

  const counts = new Map<number, number>();
  labels.forEach((label) => {
    counts.set(label, (counts.get(label) || 0) + 1);
  });

  let entropy = 0;
  const total = labels.length;

  counts.forEach((count) => {
    const prob = count / total;
    if (prob > 0) {
      entropy -= prob * Math.log2(prob);
    }
  });

  return entropy;
}

// Get step-by-step entropy calculation
export function getEntropySteps(labels: number[]): EntropyStep[] {
  const steps: EntropyStep[] = [];

  if (labels.length === 0) return steps;

  // Step 1: Count labels
  const counts = new Map<number, number>();
  labels.forEach((label) => {
    counts.set(label, (counts.get(label) || 0) + 1);
  });

  const total = labels.length;
  const countArray = Array.from(counts.entries()).sort((a, b) => a[0] - b[0]);

  steps.push({
    step: 1,
    description: "Count occurrences of each class",
    formula: `Total samples: ${total}`,
    values: countArray.map(([, count]) => count),
    result: total,
  });

  // Step 2: Calculate probabilities
  const probabilities = countArray.map(([, count]) => count / total);
  steps.push({
    step: 2,
    description: "Calculate probability for each class",
    formula: "p_i = count_i / total",
    values: probabilities,
  });

  // Step 3: Calculate -p * log2(p) for each class
  const terms = probabilities.map((p) => (p > 0 ? -p * Math.log2(p) : 0));
  steps.push({
    step: 3,
    description: "Calculate -p_i × log₂(p_i) for each class",
    formula: "-p_i × log₂(p_i)",
    values: terms,
  });

  // Step 4: Sum to get entropy
  const entropy = terms.reduce((sum, term) => sum + term, 0);
  steps.push({
    step: 4,
    description: "Sum all terms to get entropy",
    formula: "H = Σ(-p_i × log₂(p_i))",
    values: [entropy],
    result: entropy,
  });

  return steps;
}

// Calculate information gain
export function calculateInformationGain(
  parentLabels: number[],
  leftLabels: number[],
  rightLabels: number[],
): number {
  const parentEntropy = calculateEntropy(parentLabels);
  const n = parentLabels.length;
  const nLeft = leftLabels.length;
  const nRight = rightLabels.length;

  if (nLeft === 0 || nRight === 0) return 0;

  const weightedEntropy =
    (nLeft / n) * calculateEntropy(leftLabels) +
    (nRight / n) * calculateEntropy(rightLabels);

  return parentEntropy - weightedEntropy;
}

// Get detailed split calculation steps
export function getSplitCalculationSteps(
  parentLabels: number[],
  leftLabels: number[],
  rightLabels: number[],
  featureIndex: number,
  threshold: number,
): SplitCalculation {
  const parentEntropy = calculateEntropy(parentLabels);
  const leftEntropy = calculateEntropy(leftLabels);
  const rightEntropy = calculateEntropy(rightLabels);

  const n = parentLabels.length;
  const nLeft = leftLabels.length;
  const nRight = rightLabels.length;

  const leftWeight = nLeft / n;
  const rightWeight = nRight / n;

  const weightedEntropy = leftWeight * leftEntropy + rightWeight * rightEntropy;
  const informationGain = parentEntropy - weightedEntropy;

  return {
    featureIndex,
    threshold,
    parentEntropy,
    leftEntropy,
    rightEntropy,
    leftWeight,
    rightWeight,
    weightedEntropy,
    informationGain,
    leftLabels,
    rightLabels,
    parentLabels,
  };
}

// Find the best split for a dataset
export function findBestSplit(
  data: DataPoint[],
  params: TreeParams,
): { featureIndex: number; threshold: number; gain: number } | null {
  if (data.length < params.minSamplesSplit) return null;

  let bestGain = 0;
  let bestFeatureIndex = 0;
  let bestThreshold = 0;

  const numFeatures = data[0].features.length;
  const parentLabels = data.map((d) => d.label);

  // Try each feature
  for (let featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
    // Get unique values for this feature
    const values = data.map((d) => d.features[featureIndex]);
    const sortedValues = [...new Set(values)].sort((a, b) => a - b);

    // Try splits between consecutive values
    for (let i = 0; i < sortedValues.length - 1; i++) {
      const threshold = (sortedValues[i] + sortedValues[i + 1]) / 2;

      // Split data
      const leftData = data.filter(
        (d) => d.features[featureIndex] <= threshold,
      );
      const rightData = data.filter(
        (d) => d.features[featureIndex] > threshold,
      );

      // Check minimum samples constraint
      if (
        leftData.length < params.minSamplesLeaf ||
        rightData.length < params.minSamplesLeaf
      ) {
        continue;
      }

      const leftLabels = leftData.map((d) => d.label);
      const rightLabels = rightData.map((d) => d.label);

      const gain = calculateInformationGain(
        parentLabels,
        leftLabels,
        rightLabels,
      );

      if (gain > bestGain) {
        bestGain = gain;
        bestFeatureIndex = featureIndex;
        bestThreshold = threshold;
      }
    }
  }

  if (bestGain === 0) return null;

  return {
    featureIndex: bestFeatureIndex,
    threshold: bestThreshold,
    gain: bestGain,
  };
}

// Build decision tree recursively
export function buildTree(
  data: DataPoint[],
  params: TreeParams,
  depth: number = 0,
): TreeNode {
  const labels = data.map((d) => d.label);
  const entropy = calculateEntropy(labels);

  // Create leaf node if stopping criteria met
  if (
    depth >= params.maxDepth ||
    data.length < params.minSamplesSplit ||
    entropy === 0
  ) {
    // Return most common label
    const labelCounts = new Map<number, number>();
    labels.forEach((label) => {
      labelCounts.set(label, (labelCounts.get(label) || 0) + 1);
    });

    let maxCount = 0;
    let mostCommonLabel = 0;
    labelCounts.forEach((count, label) => {
      if (count > maxCount) {
        maxCount = count;
        mostCommonLabel = label;
      }
    });

    return {
      value: mostCommonLabel,
      entropy,
      samples: data.length,
    };
  }

  // Find best split
  const split = findBestSplit(data, params);

  if (!split) {
    // No good split found, create leaf
    const labelCounts = new Map<number, number>();
    labels.forEach((label) => {
      labelCounts.set(label, (labelCounts.get(label) || 0) + 1);
    });

    let maxCount = 0;
    let mostCommonLabel = 0;
    labelCounts.forEach((count, label) => {
      if (count > maxCount) {
        maxCount = count;
        mostCommonLabel = label;
      }
    });

    return {
      value: mostCommonLabel,
      entropy,
      samples: data.length,
    };
  }

  // Split data
  const leftData = data.filter(
    (d) => d.features[split.featureIndex] <= split.threshold,
  );
  const rightData = data.filter(
    (d) => d.features[split.featureIndex] > split.threshold,
  );

  // Build subtrees
  return {
    featureIndex: split.featureIndex,
    threshold: split.threshold,
    entropy,
    samples: data.length,
    left: buildTree(leftData, params, depth + 1),
    right: buildTree(rightData, params, depth + 1),
  };
}

// Predict a single data point
export function predict(tree: TreeNode, features: number[]): number {
  if (tree.value !== undefined) {
    return tree.value;
  }

  if (tree.featureIndex === undefined || tree.threshold === undefined) {
    return 0; // Default prediction
  }

  if (features[tree.featureIndex] <= tree.threshold) {
    return tree.left ? predict(tree.left, features) : 0;
  } else {
    return tree.right ? predict(tree.right, features) : 0;
  }
}

// Predict multiple data points
export function predictBatch(tree: TreeNode, dataPoints: number[][]): number[] {
  return dataPoints.map((features) => predict(tree, features));
}

// Calculate accuracy
export function calculateAccuracy(
  predictions: number[],
  actual: number[],
): number {
  if (predictions.length !== actual.length || predictions.length === 0)
    return 0;

  let correct = 0;
  for (let i = 0; i < predictions.length; i++) {
    if (predictions[i] === actual[i]) correct++;
  }

  return correct / predictions.length;
}

// Generate decision boundary for 2D visualization
export function generateDecisionBoundary(
  tree: TreeNode,
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number,
  resolution: number = 50,
): number[][] {
  const boundary: number[][] = [];
  const xStep = (xMax - xMin) / resolution;
  const yStep = (yMax - yMin) / resolution;

  for (let i = 0; i <= resolution; i++) {
    const row: number[] = [];
    for (let j = 0; j <= resolution; j++) {
      const x = xMin + j * xStep;
      const y = yMin + i * yStep;
      const prediction = predict(tree, [x, y]);
      row.push(prediction);
    }
    boundary.push(row);
  }

  return boundary;
}

// Get tree depth
export function getTreeDepth(tree: TreeNode): number {
  if (tree.value !== undefined) return 1;

  const leftDepth = tree.left ? getTreeDepth(tree.left) : 0;
  const rightDepth = tree.right ? getTreeDepth(tree.right) : 0;

  return 1 + Math.max(leftDepth, rightDepth);
}

// Count leaf nodes
export function countLeaves(tree: TreeNode): number {
  if (tree.value !== undefined) return 1;

  const leftLeaves = tree.left ? countLeaves(tree.left) : 0;
  const rightLeaves = tree.right ? countLeaves(tree.right) : 0;

  return leftLeaves + rightLeaves;
}

// Generate sample data
export function generateSampleData(numSamples: number = 100): DataPoint[] {
  const data: DataPoint[] = [];

  for (let i = 0; i < numSamples; i++) {
    const x = Math.random() * 10 - 5;
    const y = Math.random() * 10 - 5;

    // Create four quadrants with some noise
    let label: number;
    if (x > 0 && y > 0) {
      label = Math.random() > 0.1 ? 0 : 1;
    } else if (x > 0 && y <= 0) {
      label = Math.random() > 0.1 ? 1 : 0;
    } else if (x <= 0 && y > 0) {
      label = Math.random() > 0.1 ? 2 : 1;
    } else {
      label = Math.random() > 0.1 ? 3 : 2;
    }

    data.push({ features: [x, y], label });
  }

  return data;
}
