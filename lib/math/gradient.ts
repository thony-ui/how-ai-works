/**
 * Gradient computation utilities
 * Used for gradient descent and backpropagation
 */

import type { Vector } from "./matrix";

/**
 * Compute gradient for linear regression
 * Loss = MSE = (1/n) * Σ(y_pred - y_actual)²
 * y_pred = w * x + b
 *
 * ∂Loss/∂w = (2/n) * Σ(y_pred - y_actual) * x
 * ∂Loss/∂b = (2/n) * Σ(y_pred - y_actual)
 */
export function linearRegressionGradient(
  x: Vector,
  y: Vector,
  w: number,
  b: number,
): { dw: number; db: number } {
  const n = x.length;
  let dwSum = 0;
  let dbSum = 0;

  for (let i = 0; i < n; i++) {
    const yPred = w * x[i] + b;
    const error = yPred - y[i];
    dwSum += error * x[i];
    dbSum += error;
  }

  return {
    dw: (2 / n) * dwSum,
    db: (2 / n) * dbSum,
  };
}

/**
 * Update parameters using gradient descent
 */
export function gradientDescentStep(
  params: number[],
  gradients: number[],
  learningRate: number,
): number[] {
  return params.map((param, idx) => param - learningRate * gradients[idx]);
}

/**
 * Calculate MSE loss for linear regression
 */
export function calculateMSE(
  x: Vector,
  y: Vector,
  w: number,
  b: number,
): number {
  const n = x.length;
  let sum = 0;

  for (let i = 0; i < n; i++) {
    const yPred = w * x[i] + b;
    const error = yPred - y[i];
    sum += error * error;
  }

  return sum / n;
}

/**
 * Predict using linear model
 */
export function linearPredict(x: number, w: number, b: number): number {
  return w * x + b;
}

/**
 * Generate predictions for all x values
 */
export function linearPredictBatch(x: Vector, w: number, b: number): Vector {
  return x.map((xi) => linearPredict(xi, w, b));
}
