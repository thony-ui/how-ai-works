"use client";

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  TreeParams,
  buildTree,
  predictBatch,
  calculateAccuracy,
  generateDecisionBoundary,
  getTreeDepth,
  countLeaves,
  generateSampleData,
  findBestSplit,
  getSplitCalculationSteps
} from '@/lib/math/decision-tree';
import { DecisionTreeVisualizer } from './DecisionTreeVisualizer';
import { DecisionBoundaryCanvas } from './DecisionBoundaryCanvas';
import { DecisionTreePyTorchComparison } from './DecisionTreePyTorchComparison';
import { TreeBuildingSteps } from './TreeBuildingSteps';
import { RotateCcwIcon, TreeDeciduousIcon, PlusIcon, EyeIcon, EyeOffIcon } from 'lucide-react';

export default function DecisionTreePlayground() {
  const [maxDepth, setMaxDepth] = useState<number>(3);
  const [minSamplesSplit, setMinSamplesSplit] = useState<number>(2);
  const [minSamplesLeaf, setMinSamplesLeaf] = useState<number>(1);
  const [numSamples, setNumSamples] = useState<number>(100);
  const [dataVersion, setDataVersion] = useState<number>(0);
  const [showSteps, setShowSteps] = useState<boolean>(true);

  // Generate training data
  const trainingData = useMemo(
    () => generateSampleData(numSamples),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [numSamples, dataVersion]
  );

  // Build decision tree
  const tree = useMemo(
    () => {
      const params: TreeParams = {
        maxDepth,
        minSamplesSplit,
        minSamplesLeaf
      };
      return buildTree(trainingData, params);
    },
    [trainingData, maxDepth, minSamplesSplit, minSamplesLeaf]
  );

  // Calculate predictions and accuracy
  const predictions = useMemo(
    () => predictBatch(tree, trainingData.map(d => d.features)),
    [tree, trainingData]
  );
  
  const accuracy = useMemo(
    () => calculateAccuracy(predictions, trainingData.map(d => d.label)),
    [predictions, trainingData]
  );

  // Generate decision boundary
  const decisionBoundary = useMemo(
    () => generateDecisionBoundary(tree, -5, 5, -5, 5, 50),
    [tree]
  );

  const treeDepth = useMemo(() => getTreeDepth(tree), [tree]);
  const leafCount = useMemo(() => countLeaves(tree), [tree]);

  // Calculate best split for step-by-step explanation
  const bestSplitCalculation = useMemo(() => {
    const params: TreeParams = {
      maxDepth,
      minSamplesSplit,
      minSamplesLeaf
    };
    const bestSplit = findBestSplit(trainingData, params);
    
    if (!bestSplit) return null;
    
    const leftData = trainingData.filter(d => d.features[bestSplit.featureIndex] <= bestSplit.threshold);
    const rightData = trainingData.filter(d => d.features[bestSplit.featureIndex] > bestSplit.threshold);
    
    return getSplitCalculationSteps(
      trainingData.map(d => d.label),
      leftData.map(d => d.label),
      rightData.map(d => d.label),
      bestSplit.featureIndex,
      bestSplit.threshold
    );
  }, [trainingData, maxDepth, minSamplesSplit, minSamplesLeaf]);

  const reset = () => {
    setMaxDepth(3);
    setMinSamplesSplit(2);
    setMinSamplesLeaf(1);
    setNumSamples(100);
  };

  const regenerateData = () => {
    setDataVersion(v => v + 1);
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">Decision Trees</h1>
        <p className="text-lg text-muted-foreground">
          Interactive visualization of decision tree classification using the CART algorithm.
        </p>
      </div>

      {/* Status Bar */}
      <Card>
        <CardContent className="">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-6">
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Tree Depth</p>
                <p className="text-2xl font-bold">{treeDepth}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Leaf Nodes</p>
                <p className="text-2xl font-bold">{leafCount}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Training Samples</p>
                <p className="text-2xl font-bold">{numSamples}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Training Accuracy</p>
                <p className="text-2xl font-bold text-green-600">
                  {(accuracy * 100).toFixed(1)}%
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="default">
                <TreeDeciduousIcon className="w-3 h-3 mr-1" />
                Classification
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Visualizations */}
        <div className="lg:col-span-2 space-y-6">
          {/* Decision Boundary */}
          <Card>
            <CardHeader>
              <CardTitle>Decision Boundary</CardTitle>
              <CardDescription>
                Visualization of how the decision tree partitions the feature space
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex justify-center overflow-x-auto">
                <DecisionBoundaryCanvas
                  dataPoints={trainingData}
                  decisionBoundary={decisionBoundary}
                  width={600}
                  height={500}
                />
              </div>
            </CardContent>
          </Card>

          {/* Tree Structure */}
          <Card>
            <CardHeader>
              <div className="flex justify-between items-start">
                <div>
                  <CardTitle>Tree Structure</CardTitle>
                  <CardDescription>
                    Visual representation of the decision tree. Purple nodes are decision nodes, green are leaf nodes.
                  </CardDescription>
                </div>
                <Button
                  onClick={() => setShowSteps(!showSteps)}
                  variant="outline"
                  size="sm"
                >
                  {showSteps ? (
                    <>
                      <EyeOffIcon className="w-4 h-4 mr-2" />
                      Hide Math
                    </>
                  ) : (
                    <>
                      <EyeIcon className="w-4 h-4 mr-2" />
                      Show Math
                    </>
                  )}
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="flex justify-center overflow-auto">
                <DecisionTreeVisualizer tree={tree} width={800} height={400} />
              </div>
            </CardContent>
          </Card>

          {/* Step-by-Step Math Explanation */}
          {showSteps && bestSplitCalculation && (
            <TreeBuildingSteps 
              splitCalculation={bestSplitCalculation} 
              showDetails={true}
            />
          )}

          {/* Algorithm Explanation */}
          <Card>
            <CardHeader>
              <CardTitle>How Decision Trees Work</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h3 className="font-semibold text-lg mb-2">Building the Tree</h3>
                <ol className="list-decimal list-inside space-y-2 text-sm text-muted-foreground">
                  <li>Start with all training data at the root node</li>
                  <li>Find the best feature and threshold to split the data (using entropy and information gain)</li>
                  <li>Create child nodes with the split data</li>
                  <li>Recursively repeat for each child until stopping criteria are met</li>
                  <li>Stopping criteria: max depth reached, minimum samples, or pure node</li>
                </ol>
              </div>
              <Separator />
              <div>
                <h3 className="font-semibold text-lg mb-2">Making Predictions</h3>
                <ol className="list-decimal list-inside space-y-2 text-sm text-muted-foreground">
                  <li>Start at the root node with your input features</li>
                  <li>Follow the decision path based on feature values and thresholds</li>
                  <li>Continue until you reach a leaf node</li>
                  <li>Return the class label stored in the leaf node</li>
                </ol>
              </div>
              <Separator />
              <div>
                <h3 className="font-semibold text-lg mb-2">Entropy & Information Gain</h3>
                <p className="text-sm text-muted-foreground mb-2">
                  Entropy measures the impurity or uncertainty in the data:
                </p>
                <div className="bg-gray-50 p-3 rounded font-mono text-sm">
                  H(S) = -Σ(p_i × log₂(p_i))
                </div>
                <p className="text-sm text-muted-foreground mt-2">
                  where p_i is the probability of class i. Higher entropy = more disorder.
                </p>
                <p className="text-sm text-muted-foreground mt-2">
                  Information Gain measures the reduction in entropy after a split. We choose the split that maximizes information gain.
                </p>
              </div>
            </CardContent>
          </Card>

          {/* PyTorch Comparison */}
          <Card>
            <CardHeader>
              <CardTitle>Python/Scikit-learn Comparison</CardTitle>
              <CardDescription>
                See how this compares to scikit-learn&apos;s implementation
              </CardDescription>
            </CardHeader>
            <CardContent>

              <DecisionTreePyTorchComparison 
                trainingData={trainingData}
                maxDepth={maxDepth}
                minSamplesSplit={minSamplesSplit}
                minSamplesLeaf={minSamplesLeaf}
              />
            </CardContent>
          </Card>
        </div>

        {/* Right Column - Controls */}
        <div className="space-y-6">
          {/* Tree Parameters */}
          <Card>
            <CardHeader>
              <CardTitle>Tree Parameters</CardTitle>
              <CardDescription>
                Adjust hyperparameters to control tree growth
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>Max Depth</Label>
                  <span className="text-sm font-mono">{maxDepth}</span>
                </div>
                <Slider
                  value={[maxDepth]}
                  onValueChange={(value) => setMaxDepth(value[0])}
                  min={1}
                  max={10}
                  step={1}
                />
                <p className="text-xs text-muted-foreground">
                  Maximum depth of the tree. Deeper trees can overfit.
                </p>
              </div>

              <Separator />

              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>Min Samples Split</Label>
                  <span className="text-sm font-mono">{minSamplesSplit}</span>
                </div>
                <Slider
                  value={[minSamplesSplit]}
                  onValueChange={(value) => setMinSamplesSplit(value[0])}
                  min={2}
                  max={20}
                  step={1}
                />
                <p className="text-xs text-muted-foreground">
                  Minimum samples required to split a node.
                </p>
              </div>

              <Separator />

              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>Min Samples Leaf</Label>
                  <span className="text-sm font-mono">{minSamplesLeaf}</span>
                </div>
                <Slider
                  value={[minSamplesLeaf]}
                  onValueChange={(value) => setMinSamplesLeaf(value[0])}
                  min={1}
                  max={10}
                  step={1}
                />
                <p className="text-xs text-muted-foreground">
                  Minimum samples required in a leaf node.
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Data Controls */}
          <Card>
            <CardHeader>
              <CardTitle>Data Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>Number of Samples</Label>
                  <span className="text-sm font-mono">{numSamples}</span>
                </div>
                <Slider
                  value={[numSamples]}
                  onValueChange={(value) => setNumSamples(value[0])}
                  min={50}
                  max={300}
                  step={10}
                />
              </div>

              <Button
                onClick={regenerateData}
                variant="outline"
                className="w-full"
              >
                <PlusIcon className="w-4 h-4 mr-2" />
                Regenerate Data
              </Button>
            </CardContent>
          </Card>

          {/* Actions */}
          <Card>
            <CardHeader>
              <CardTitle>Actions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <Button onClick={reset} variant="outline" className="w-full">
                <RotateCcwIcon className="w-4 h-4 mr-2" />
                Reset to Defaults
              </Button>
            </CardContent>
          </Card>

          {/* Info Card */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Key Concepts</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-xs text-muted-foreground">
              <div>
                <span className="font-semibold text-foreground">Overfitting:</span> Trees that
                are too deep can memorize training data. Use max depth and min samples constraints.
              </div>
              <div>
                <span className="font-semibold text-foreground">Feature Selection:</span> At each
                split, the algorithm chooses the feature and threshold that best separates the classes.
              </div>
              <div>
                <span className="font-semibold text-foreground">Non-parametric:</span> Decision
                trees make no assumptions about data distribution.
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
