/**
 * Activation functions and their derivatives
 * Used in neural network forward and backward passes
 */

import type { Matrix } from "./matrix";

/**
 * ReLU (Rectified Linear Unit) activation function
 */
export function relu(x: number): number {
  return Math.max(0, x);
}

/**
 * ReLU derivative for backpropagation
 */
export function reluDerivative(x: number): number {
  return x > 0 ? 1 : 0;
}

/**
 * Apply ReLU to entire matrix
 */
export function reluMatrix(matrix: Matrix): Matrix {
  return matrix.map((row) => row.map(relu));
}

/**
 * Apply ReLU derivative to entire matrix
 */
export function reluDerivativeMatrix(matrix: Matrix): Matrix {
  return matrix.map((row) => row.map(reluDerivative));
}

/**
 * Sigmoid activation function
 */
export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Sigmoid derivative for backpropagation
 */
export function sigmoidDerivative(x: number): number {
  const s = sigmoid(x);
  return s * (1 - s);
}

/**
 * Linear activation (identity function)
 */
export function linear(x: number): number {
  return x;
}

/**
 * Linear derivative
 */
export function linearDerivative(): number {
  return 1;
}

/**
 * Tanh activation function
 */
export function tanh(x: number): number {
  return Math.tanh(x);
}

/**
 * Tanh derivative for backpropagation
 */
export function tanhDerivative(x: number): number {
  const t = Math.tanh(x);
  return 1 - t * t;
}
