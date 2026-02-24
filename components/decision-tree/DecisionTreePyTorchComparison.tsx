"use client";

import React from 'react';
import { DataPoint } from '@/lib/math/decision-tree';

interface DecisionTreePyTorchComparisonProps {
  trainingData: DataPoint[];
  maxDepth: number;
  minSamplesSplit: number;
  minSamplesLeaf: number;
}

export function DecisionTreePyTorchComparison({
  trainingData,
  maxDepth,
  minSamplesSplit,
  minSamplesLeaf
}: DecisionTreePyTorchComparisonProps) {
  
  const generatePythonCode = () => {
    const features = trainingData.map(d => `[${d.features.join(', ')}]`).join(',\n    ');
    const labels = trainingData.map(d => d.label).join(', ');
    
    return `from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Training data
X = np.array([
    ${features}
])

y = np.array([${labels}])

# Create and train decision tree
clf = DecisionTreeClassifier(
    max_depth=${maxDepth},
    min_samples_split=${minSamplesSplit},
    min_samples_leaf=${minSamplesLeaf},
    criterion='gini',
    random_state=42
)

clf.fit(X, y)

# Make predictions
predictions = clf.predict(X)

# Calculate accuracy
accuracy = accuracy_score(y, predictions)
print(f"Training Accuracy: {accuracy:.4f}")

# Get tree information
print(f"Tree Depth: {clf.get_depth()}")
print(f"Number of Leaves: {clf.get_n_leaves()}")

# Visualize tree (optional)
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=['X0', 'X1'], 
          class_names=[str(i) for i in np.unique(y)])
plt.show()

# Make prediction for new data point
new_point = np.array([[0.5, 0.5]])
prediction = clf.predict(new_point)
print(f"Prediction for [0.5, 0.5]: {prediction[0]}")`;
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(generatePythonCode());
  };

  return (
    <div className="space-y-3">
      <div className="flex justify-between items-start">
        <p className="text-sm text-muted-foreground">
          Equivalent Python code using scikit-learn:
        </p>
        <button
          onClick={handleCopy}
          className="text-xs px-2 py-1 bg-gray-100 hover:bg-gray-200 rounded shrink-0"
          type="button"
        >
          Copy Code
        </button>
      </div>
      
      <pre className="rounded-lg bg-slate-950 text-slate-50 p-4 overflow-x-auto text-xs font-mono whitespace-pre-wrap break-words">
        <code>{generatePythonCode()}</code>
      </pre>
      
      <div className="text-xs text-muted-foreground space-y-2">
        <p>
          <strong>Key differences:</strong>
        </p>
        <ul className="list-disc list-inside space-y-1 ml-2">
          <li>Scikit-learn uses optimized C/Cython implementations for speed</li>
          <li>This implementation uses CART algorithm with Gini impurity</li>
          <li>Scikit-learn supports additional features like pruning and cost-complexity</li>
          <li>Both produce similar decision boundaries for the same parameters</li>
        </ul>
      </div>
    </div>
  );
}
