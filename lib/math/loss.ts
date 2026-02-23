/**
 * Loss functions and their derivatives
 * Used to compute model error and gradients
 */

import type { Matrix, Vector } from "./matrix";

/**
 * Mean Squared Error loss
 */
export function mse(predicted: Vector, actual: Vector): number {
  const sum = predicted.reduce((acc, val, idx) => {
    const diff = val - actual[idx];
    return acc + diff * diff;
  }, 0);
  return sum / predicted.length;
}

/**
 * MSE derivative with respect to predictions
 */
export function mseDerivative(predicted: Vector, actual: Vector): Vector {
  return predicted.map(
    (val, idx) => (2 / predicted.length) * (val - actual[idx]),
  );
}

/**
 * Single value squared error
 */
export function squaredError(predicted: number, actual: number): number {
  const diff = predicted - actual;
  return diff * diff;
}

/**
 * Squared error derivative
 */
export function squaredErrorDerivative(
  predicted: number,
  actual: number,
): number {
  return 2 * (predicted - actual);
}

/**
 * Calculate MSE for batch of predictions
 */
export function batchMSE(predictions: Matrix, targets: Matrix): number {
  let totalError = 0;
  for (let i = 0; i < predictions.length; i++) {
    totalError += mse(predictions[i], targets[i]);
  }
  return totalError / predictions.length;
}
