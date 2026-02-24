"use client";

import React from 'react';
import { Separator } from '@/components/ui/separator';
import { 
  SplitCalculation,
  getEntropySteps 
} from '@/lib/math/decision-tree';

interface TreeBuildingStepsProps {
  splitCalculation: SplitCalculation | null;
  showDetails: boolean;
}

export function TreeBuildingSteps({ 
  splitCalculation, 
  showDetails 
}: TreeBuildingStepsProps) {
  
  if (!splitCalculation || !showDetails) return null;

  const parentEntropySteps = getEntropySteps(splitCalculation.parentLabels);
  const leftEntropySteps = getEntropySteps(splitCalculation.leftLabels);
  const rightEntropySteps = getEntropySteps(splitCalculation.rightLabels);

  const getLabelCounts = (labels: number[]) => {
    const counts = new Map<number, number>();
    labels.forEach(label => {
      counts.set(label, (counts.get(label) || 0) + 1);
    });
    return Array.from(counts.entries()).sort((a, b) => a[0] - b[0]);
  };

  const parentCounts = getLabelCounts(splitCalculation.parentLabels);
  const leftCounts = getLabelCounts(splitCalculation.leftLabels);
  const rightCounts = getLabelCounts(splitCalculation.rightLabels);

  return (
    <div className="space-y-4">
      <div className="bg-blue-50 border-2 border-blue-200 rounded-lg p-4">
        <h3 className="font-bold text-blue-900 text-lg mb-3">
          Split Analysis: Feature X{splitCalculation.featureIndex} at threshold {splitCalculation.threshold.toFixed(2)}
        </h3>
        
        {/* Step 1: Parent Node Entropy */}
        <div className="space-y-3">
          <div className="bg-white rounded p-3">
            <p className="font-semibold text-purple-700 mb-2">Step 1: Calculate Parent Node Entropy</p>
            
            <div className="ml-2 space-y-2 text-sm">
              <div>
                <p className="font-medium">Label distribution:</p>
                <p className="ml-2">
                  Total samples: {splitCalculation.parentLabels.length}
                </p>
                {parentCounts.map(([label, count]) => (
                  <p key={label} className="ml-2">
                    Class {label}: {count} samples (p = {(count / splitCalculation.parentLabels.length).toFixed(3)})
                  </p>
                ))}
              </div>

              <Separator />

              <div>
                <p className="font-medium text-blue-700">Entropy formula:</p>
                <div className="bg-blue-50 p-2 rounded font-mono text-xs mt-1">
                  H(S) = -Σ p_i × log₂(p_i)
                </div>
              </div>

              <div>
                <p className="font-medium">Step-by-step calculation:</p>
                {parentEntropySteps.map((step, idx) => (
                  <div key={idx} className="ml-2 mt-2 bg-gray-50 p-2 rounded">
                    <p className="text-blue-600 font-medium">{step.step}. {step.description}</p>
                    <p className="font-mono text-xs mt-1">{step.formula}</p>
                    {step.values.length > 1 && step.values.length < 10 && (
                      <p className="mt-1">
                        [{step.values.map(v => v.toFixed(4)).join(', ')}]
                      </p>
                    )}
                    {step.result !== undefined && (
                      <p className="font-bold text-blue-800 mt-1">
                        Result: {step.result.toFixed(4)}
                      </p>
                    )}
                  </div>
                ))}
              </div>

              <div className="bg-purple-100 p-2 rounded">
                <p className="font-bold text-purple-900">
                  Parent Entropy H(S) = {splitCalculation.parentEntropy.toFixed(4)}
                </p>
              </div>
            </div>
          </div>

          <Separator />

          {/* Step 2: Left Child Entropy */}
          <div className="bg-white rounded p-3">
            <p className="font-semibold text-green-700 mb-2">Step 2: Calculate Left Child Entropy (X{splitCalculation.featureIndex} ≤ {splitCalculation.threshold.toFixed(2)})</p>
            
            <div className="ml-2 space-y-2 text-sm">
              <div className="bg-green-50 border border-green-200 p-3 rounded mb-3">
                <p className="font-semibold text-green-800 mb-1">How the split works:</p>
                <p className="text-sm mb-1">
                  • We split the data based on feature <span className="font-mono font-bold">X{splitCalculation.featureIndex}</span> at threshold <span className="font-mono font-bold">{splitCalculation.threshold.toFixed(2)}</span>
                </p>
                <p className="text-sm mb-1">
                  • <span className="font-bold text-green-700">Left child</span> gets all samples where <span className="font-mono">X{splitCalculation.featureIndex} ≤ {splitCalculation.threshold.toFixed(2)}</span>
                </p>
                <p className="text-sm">
                  • Out of {splitCalculation.parentLabels.length} total samples, <span className="font-bold">{splitCalculation.leftLabels.length} samples</span> meet this condition
                </p>
              </div>

              <div>
                <p className="font-medium">Label distribution:</p>
                <p className="ml-2">
                  Total samples: {splitCalculation.leftLabels.length}
                </p>
                {leftCounts.map(([label, count]) => (
                  <p key={label} className="ml-2">
                    Class {label}: {count} samples (p = {(count / splitCalculation.leftLabels.length).toFixed(3)})
                  </p>
                ))}
              </div>

              <div>
                <p className="font-medium">Calculation:</p>
                {leftEntropySteps.filter(step => step.step === 4).map((step, idx) => (
                  <div key={idx} className="ml-2 bg-gray-50 p-2 rounded">
                    <p className="font-mono text-xs">{step.formula}</p>
                    {step.result !== undefined && (
                      <p className="font-bold text-green-800 mt-1">
                        H(S_left) = {step.result.toFixed(4)}
                      </p>
                    )}
                  </div>
                ))}
              </div>

              <div className="bg-green-100 p-2 rounded">
                <p className="font-bold text-green-900">
                  Left Child Entropy H(S_left) = {splitCalculation.leftEntropy.toFixed(4)}
                </p>
                <p className="text-sm mt-1">
                  Weight: {splitCalculation.leftWeight.toFixed(3)} ({splitCalculation.leftLabels.length}/{splitCalculation.parentLabels.length})
                </p>
              </div>
            </div>
          </div>

          <Separator />

          {/* Step 3: Right Child Entropy */}
          <div className="bg-white rounded p-3">
            <p className="font-semibold text-red-700 mb-2">Step 3: Calculate Right Child Entropy (X{splitCalculation.featureIndex} &gt; {splitCalculation.threshold.toFixed(2)})</p>
            
            <div className="ml-2 space-y-2 text-sm">
              <div className="bg-red-50 border border-red-200 p-3 rounded mb-3">
                <p className="font-semibold text-red-800 mb-1">How the split works:</p>
                <p className="text-sm mb-1">
                  • <span className="font-bold text-red-700">Right child</span> gets all samples where <span className="font-mono">X{splitCalculation.featureIndex} &gt; {splitCalculation.threshold.toFixed(2)}</span>
                </p>
                <p className="text-sm">
                  • Out of {splitCalculation.parentLabels.length} total samples, <span className="font-bold">{splitCalculation.rightLabels.length} samples</span> meet this condition
                </p>
              </div>

              <div>
                <p className="font-medium">Label distribution:</p>
                <p className="ml-2">
                  Total samples: {splitCalculation.rightLabels.length}
                </p>
                {rightCounts.map(([label, count]) => (
                  <p key={label} className="ml-2">
                    Class {label}: {count} samples (p = {(count / splitCalculation.rightLabels.length).toFixed(3)})
                  </p>
                ))}
              </div>

              <div>
                <p className="font-medium">Calculation:</p>
                {rightEntropySteps.filter(step => step.step === 4).map((step, idx) => (
                  <div key={idx} className="ml-2 bg-gray-50 p-2 rounded">
                    <p className="font-mono text-xs">{step.formula}</p>
                    {step.result !== undefined && (
                      <p className="font-bold text-red-800 mt-1">
                        H(S_right) = {step.result.toFixed(4)}
                      </p>
                    )}
                  </div>
                ))}
              </div>

              <div className="bg-red-100 p-2 rounded">
                <p className="font-bold text-red-900">
                  Right Child Entropy H(S_right) = {splitCalculation.rightEntropy.toFixed(4)}
                </p>
                <p className="text-sm mt-1">
                  Weight: {splitCalculation.rightWeight.toFixed(3)} ({splitCalculation.rightLabels.length}/{splitCalculation.parentLabels.length})
                </p>
              </div>
            </div>
          </div>

          <Separator />

          {/* Step 4: Information Gain */}
          <div className="bg-white rounded p-3">
            <p className="font-semibold text-orange-700 mb-2">Step 4: Calculate Information Gain</p>
            
            <div className="ml-2 space-y-2 text-sm">
              <div>
                <p className="font-medium text-orange-700">Information Gain formula:</p>
                <div className="bg-orange-50 p-2 rounded font-mono text-xs mt-1">
                  IG = H(parent) - [w_left × H(left) + w_right × H(right)]
                </div>
              </div>

              <div className="space-y-1">
                <p className="font-medium">Step-by-step:</p>
                <div className="ml-2 bg-gray-50 p-2 rounded space-y-1">
                  <p>H(parent) = {splitCalculation.parentEntropy.toFixed(4)}</p>
                  <p>w_left = {splitCalculation.leftWeight.toFixed(3)}</p>
                  <p>H(left) = {splitCalculation.leftEntropy.toFixed(4)}</p>
                  <p>w_right = {splitCalculation.rightWeight.toFixed(3)}</p>
                  <p>H(right) = {splitCalculation.rightEntropy.toFixed(4)}</p>
                </div>
              </div>

              <div className="ml-2 bg-gray-50 p-2 rounded space-y-1">
                <p className="font-medium">Weighted entropy:</p>
                <p>
                  = {splitCalculation.leftWeight.toFixed(3)} × {splitCalculation.leftEntropy.toFixed(4)} + {splitCalculation.rightWeight.toFixed(3)} × {splitCalculation.rightEntropy.toFixed(4)}
                </p>
                <p>
                  = {(splitCalculation.leftWeight * splitCalculation.leftEntropy).toFixed(4)} + {(splitCalculation.rightWeight * splitCalculation.rightEntropy).toFixed(4)}
                </p>
                <p className="font-bold">
                  = {splitCalculation.weightedEntropy.toFixed(4)}
                </p>
              </div>

              <div className="ml-2 bg-gray-50 p-2 rounded space-y-1">
                <p className="font-medium">Information Gain:</p>
                <p>
                  IG = {splitCalculation.parentEntropy.toFixed(4)} - {splitCalculation.weightedEntropy.toFixed(4)}
                </p>
                <p className="font-bold text-orange-800">
                  IG = {splitCalculation.informationGain.toFixed(4)}
                </p>
              </div>

              <div className="bg-orange-100 p-3 rounded border-2 border-orange-300">
                <p className="font-bold text-orange-900 text-lg">
                  Information Gain = {splitCalculation.informationGain.toFixed(4)}
                </p>
                <p className="text-sm mt-1 text-orange-800">
                  {splitCalculation.informationGain > 0 
                    ? 'This split reduces entropy and improves classification!' 
                    : 'This split provides no information gain.'}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Summary and interpretation */}
      <div className="bg-gray-50 border rounded-lg p-4 space-y-2 text-sm">
        <p className="font-semibold text-gray-900">📚 Interpretation:</p>
        <ul className="list-disc list-inside space-y-1 text-gray-700 ml-2">
          <li>
            <strong>Entropy</strong> measures impurity/disorder. 0 = pure (all same class), higher = mixed classes
          </li>
          <li>
            <strong>Information Gain</strong> measures how much the split reduces entropy
          </li>
          <li>
            The algorithm chooses the split with the <strong>highest information gain</strong>
          </li>
          <li>
            Lower child entropy = better separation of classes = better split
          </li>
        </ul>
      </div>
    </div>
  );
}
