"use client";

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  runKMeans,
  runKMeansWithSteps,
  generateClusteredData,
  calculateSilhouetteScore,
  calculateElbowMethod
} from '@/lib/math/k-means';
import { KMeansVisualizer } from './KMeansVisualizer';
import { ElbowChart } from './ElbowChart';
import { KMeansPyTorchComparison } from './KMeansPyTorchComparison';
import { RotateCcwIcon, PlusIcon, PlayIcon, PauseIcon } from 'lucide-react';

export default function KMeansPlayground() {
  const [numClusters, setNumClusters] = useState<number>(3);
  const [maxIterations, setMaxIterations] = useState<number>(100);
  const [initMethod, setInitMethod] = useState<'random' | 'kmeans++'>('kmeans++');
  const [numPoints, setNumPoints] = useState<number>(90);
  const [dataVersion, setDataVersion] = useState<number>(0);
  
  // Animation state
  const [isAnimating, setIsAnimating] = useState<boolean>(false);
  const [currentStep, setCurrentStep] = useState<number>(0);

  // Generate data
  const rawData = useMemo(
    () => generateClusteredData(3, numPoints / 3),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [numPoints, dataVersion]
  );

  // Run k-means
  const clusterResult = useMemo(
    () => runKMeans(rawData, numClusters, maxIterations, initMethod),
    [rawData, numClusters, maxIterations, initMethod]
  );

  // Get step-by-step iterations
  const steps = useMemo(
    () => runKMeansWithSteps(rawData, numClusters, maxIterations, initMethod),
    [rawData, numClusters, maxIterations, initMethod]
  );

  // Calculate silhouette score
  const silhouetteScore = useMemo(
    () => calculateSilhouetteScore(clusterResult.points, clusterResult.centroids),
    [clusterResult]
  );

  // Calculate elbow method data
  const elbowData = useMemo(
    () => calculateElbowMethod(rawData, Math.min(8, Math.floor(rawData.length / 2))),
    [rawData]
  );

  // Get current step for animation
  const displayStep = isAnimating ? steps[currentStep] : {
    points: clusterResult.points,
    centroids: clusterResult.centroids,
    inertia: clusterResult.inertia
  };

  // Animation control
  React.useEffect(() => {
    if (!isAnimating) return;

    const interval = setInterval(() => {
      setCurrentStep(prev => {
        if (prev >= steps.length - 1) {
          setIsAnimating(false);
          return prev;
        }
        return prev + 1;
      });
    }, 800);

    return () => clearInterval(interval);
  }, [isAnimating, steps.length]);

  const toggleAnimation = () => {
    if (isAnimating) {
      setIsAnimating(false);
    } else {
      setCurrentStep(0);
      setIsAnimating(true);
    }
  };

  const reset = () => {
    setNumClusters(3);
    setMaxIterations(100);
    setInitMethod('kmeans++');
    setNumPoints(90);
    setIsAnimating(false);
    setCurrentStep(0);
  };

  const regenerateData = () => {
    setDataVersion(v => v + 1);
    setIsAnimating(false);
    setCurrentStep(0);
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">K-Means Clustering</h1>
        <p className="text-lg text-muted-foreground">
          Interactive visualization of K-means clustering algorithm for unsupervised learning.
        </p>
      </div>

      {/* Status Bar */}
      <Card>
        <CardContent className="">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-6">
              <div className="text-center">
                <p className="text-sm text-muted-foreground">K (Clusters)</p>
                <p className="text-2xl font-bold">{numClusters}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Data Points</p>
                <p className="text-2xl font-bold">{rawData.length}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Iterations</p>
                <p className="text-2xl font-bold">{clusterResult.iterations}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Inertia</p>
                <p className="text-2xl font-bold text-blue-600">
                  {displayStep.inertia.toFixed(2)}
                </p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Silhouette</p>
                <p className="text-2xl font-bold text-green-600">
                  {silhouetteScore.toFixed(3)}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant={clusterResult.converged ? "default" : "secondary"}>
                {clusterResult.converged ? "Converged" : "Max Iterations"}
              </Badge>
              <Badge variant="outline">
                {initMethod === 'kmeans++' ? 'K-Means++' : 'Random Init'}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Visualizations */}
        <div className="lg:col-span-2 space-y-6">
          {/* Cluster Visualization */}
          <Card>
            <CardHeader>
              <div className="flex justify-between items-start">
                <div>
                  <CardTitle>Cluster Visualization</CardTitle>
                  <CardDescription>
                    {isAnimating ? `Iteration ${currentStep + 1} of ${steps.length}` : 
                     'Final clustering result - X marks centroids'}
                  </CardDescription>
                </div>
                <Button
                  onClick={toggleAnimation}
                  variant="outline"
                  size="sm"
                >
                  {isAnimating ? (
                    <>
                      <PauseIcon className="w-4 h-4 mr-2" />
                      Pause
                    </>
                  ) : (
                    <>
                      <PlayIcon className="w-4 h-4 mr-2" />
                      Animate
                    </>
                  )}
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="flex justify-center overflow-x-auto">
                <KMeansVisualizer
                  points={displayStep.points}
                  centroids={displayStep.centroids}
                  width={600}
                  height={500}
                />
              </div>
            </CardContent>
          </Card>

          {/* Elbow Method */}
          <Card>
            <CardHeader>
              <CardTitle>Elbow Method</CardTitle>
              <CardDescription>
                Find optimal K by looking for the &quot;elbow&quot; in the inertia curve
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex justify-center overflow-x-auto">
                <ElbowChart data={elbowData} currentK={numClusters} />
              </div>
              <p className="text-sm text-muted-foreground mt-4">
                The elbow point suggests the optimal number of clusters where adding more clusters 
                provides diminishing returns in reducing inertia.
              </p>
            </CardContent>
          </Card>

          {/* Algorithm Explanation */}
          <Card>
            <CardHeader>
              <CardTitle>How K-Means Works</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h3 className="font-semibold text-lg mb-2">Algorithm Steps</h3>
                <ol className="list-decimal list-inside space-y-2 text-sm text-muted-foreground">
                  <li>Initialize K centroids (randomly or using K-means++)</li>
                  <li>Assign each point to the nearest centroid</li>
                  <li>Update centroids to the mean of assigned points</li>
                  <li>Repeat steps 2-3 until convergence or max iterations</li>
                </ol>
              </div>
              <Separator />
              <div>
                <h3 className="font-semibold text-lg mb-2">K-Means++ Initialization</h3>
                <p className="text-sm text-muted-foreground">
                  K-means++ improves initialization by selecting centroids that are far apart,
                  leading to better and more consistent clustering results compared to random initialization.
                </p>
              </div>
              <Separator />
              <div>
                <h3 className="font-semibold text-lg mb-2">Key Metrics</h3>
                <ul className="list-disc list-inside space-y-2 text-sm text-muted-foreground">
                  <li>
                    <strong>Inertia:</strong> Sum of squared distances to nearest centroid (lower is better)
                  </li>
                  <li>
                    <strong>Silhouette Score:</strong> Measures cluster separation quality (-1 to 1, higher is better)
                  </li>
                </ul>
              </div>
            </CardContent>
          </Card>

          {/* Python Comparison */}
          <Card>
            <CardHeader>
              <CardTitle>Python/Scikit-learn Comparison</CardTitle>
              <CardDescription>
                See how this compares to scikit-learn&apos;s implementation
              </CardDescription>
            </CardHeader>
            <CardContent>
              <KMeansPyTorchComparison
                points={rawData}
                numClusters={numClusters}
                maxIterations={maxIterations}
                initMethod={initMethod}
              />
            </CardContent>
          </Card> 
        </div>

        {/* Right Column - Controls */}
        <div className="space-y-6">
          {/* Clustering Parameters */}
          <Card>
            <CardHeader>
              <CardTitle>Clustering Parameters</CardTitle>
              <CardDescription>
                Adjust K-means settings
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>Number of Clusters (K)</Label>
                  <span className="text-sm font-mono">{numClusters}</span>
                </div>
                <Slider
                  value={[numClusters]}
                  onValueChange={(value) => {
                    setNumClusters(value[0]);
                    setIsAnimating(false);
                    setCurrentStep(0);
                  }}
                  min={2}
                  max={8}
                  step={1}
                />
                <p className="text-xs text-muted-foreground">
                  Number of clusters to find in the data.
                </p>
              </div>

              <Separator />

              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>Max Iterations</Label>
                  <span className="text-sm font-mono">{maxIterations}</span>
                </div>
                <Slider
                  value={[maxIterations]}
                  onValueChange={(value) => {
                    setMaxIterations(value[0]);
                    setIsAnimating(false);
                    setCurrentStep(0);
                  }}
                  min={10}
                  max={300}
                  step={10}
                />
                <p className="text-xs text-muted-foreground">
                  Maximum number of iterations before stopping.
                </p>
              </div>

              <Separator />

              <div className="space-y-2">
                <Label>Initialization Method</Label>
                <div className="flex gap-2">
                  <Button
                    variant={initMethod === 'kmeans++' ? 'default' : 'outline'}
                    onClick={() => {
                      setInitMethod('kmeans++');
                      setIsAnimating(false);
                      setCurrentStep(0);
                    }}
                    className="flex-1"
                    size="sm"
                  >
                    K-Means++
                  </Button>
                  <Button
                    variant={initMethod === 'random' ? 'default' : 'outline'}
                    onClick={() => {
                      setInitMethod('random');
                      setIsAnimating(false);
                      setCurrentStep(0);
                    }}
                    className="flex-1"
                    size="sm"
                  >
                    Random
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground">
                  K-means++ provides better initial centroids.
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
                  <Label>Number of Points</Label>
                  <span className="text-sm font-mono">{numPoints}</span>
                </div>
                <Slider
                  value={[numPoints]}
                  onValueChange={(value) => setNumPoints(value[0])}
                  min={30}
                  max={300}
                  step={30}
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
                <span className="font-semibold text-foreground">Choosing K:</span> Use the elbow
                method or silhouette analysis to find the optimal number of clusters.
              </div>
              <div>
                <span className="font-semibold text-foreground">Convergence:</span> Algorithm stops
                when centroids no longer move significantly between iterations.
              </div>
              <div>
                <span className="font-semibold text-foreground">Limitations:</span> Assumes spherical
                clusters and is sensitive to outliers and initial centroid placement.
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
