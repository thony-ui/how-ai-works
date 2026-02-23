"use client";

import React from 'react';

interface TokenPredictionVizProps {
  predictions: Array<{ token: string; probability: number; logit: number }>;
}

export function TokenPredictionViz({ predictions }: TokenPredictionVizProps) {
  return (
    <div className="space-y-3">
      {predictions.map((pred, i) => (
        <div key={i} className="space-y-1">
          <div className="flex justify-between items-center text-sm">
            <span className="font-semibold">
              {i + 1}. &quot;{pred.token}&quot;
            </span>
            <span className="font-mono text-muted-foreground">
              {(pred.probability * 100).toFixed(2)}%
            </span>
          </div>
          <div className="relative h-8 bg-secondary rounded overflow-hidden">
            <div
              className="absolute top-0 left-0 h-full bg-blue-500"
              style={{ width: `${pred.probability * 100}%` }}
            />
            <div className="absolute inset-0 flex items-center justify-start px-3">
              <span className="text-xs font-mono text-foreground mix-blend-difference">
                logit: {pred.logit.toFixed(3)}
              </span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
