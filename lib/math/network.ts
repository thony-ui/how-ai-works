/**
 * Two-layer neural network implementation
 * Architecture: Input(2) -> Hidden(4, ReLU) -> Output(1)
 */

import {
  matrixMultiply,
  matrixAdd,
  transpose,
  scalarMultiply,
  randomMatrix,
} from "./matrix";
import { relu, reluDerivativeMatrix } from "./activations";

export interface NetworkState {
  // Weights and biases
  w1: number[][]; // 2x4 input to hidden
  b1: number[][]; // 1x4 hidden layer bias
  w2: number[][]; // 4x1 hidden to output
  b2: number[][]; // 1x1 output bias

  // Forward pass activations
  z1?: number[][]; // Hidden layer pre-activation
  a1?: number[][]; // Hidden layer post-activation (ReLU)
  z2?: number[][]; // Output layer pre-activation
  a2?: number[][]; // Output layer post-activation (final output)

  // Input and target
  input?: number[][];
  target?: number[][];

  // Loss
  loss?: number;

  // Backward pass gradients
  dz2?: number[][]; // Gradient at output
  dw2?: number[][]; // Gradient for w2
  db2?: number[][]; // Gradient for b2
  da1?: number[][]; // Gradient flowing to hidden layer
  dz1?: number[][]; // Gradient at hidden layer
  dw1?: number[][]; // Gradient for w1
  db1?: number[][]; // Gradient for b1
}

export class TwoLayerNetwork {
  state: NetworkState;

  constructor() {
    // Initialize weights with small random values
    this.state = {
      w1: this.initializeWeights(2, 4),
      b1: [[0, 0, 0, 0]],
      w2: this.initializeWeights(4, 1),
      b2: [[0]],
    };
  }

  private initializeWeights(inputSize: number, outputSize: number): number[][] {
    // Positive initialization for better visualization
    const scale = Math.sqrt(2.0 / inputSize);
    return randomMatrix(inputSize, outputSize).map((row) =>
      row.map((val) => Math.abs(val) * scale),
    );
  }

  /**
   * Forward pass through the network
   */
  forward(input: number[][]): number[][] {
    this.state.input = input;

    // Hidden layer: z1 = input @ w1 + b1
    this.state.z1 = matrixAdd(
      matrixMultiply(input, this.state.w1),
      this.state.b1,
    );

    // Apply ReLU activation: a1 = relu(z1)
    this.state.a1 = this.state.z1.map((row) => row.map(relu));

    // Output layer: z2 = a1 @ w2 + b2
    this.state.z2 = matrixAdd(
      matrixMultiply(this.state.a1, this.state.w2),
      this.state.b2,
    );

    // No activation on output (for regression)
    this.state.a2 = this.state.z2;

    return this.state.a2;
  }

  /**
   * Compute MSE loss
   */
  computeLoss(target: number[][]): number {
    this.state.target = target;

    if (!this.state.a2) {
      throw new Error("Must run forward pass before computing loss");
    }

    const prediction = this.state.a2[0][0];
    const actual = target[0][0];
    const diff = prediction - actual;

    // Using MSE/2 for cleaner gradient (no factor of 2)
    this.state.loss = (diff * diff) / 2;
    return this.state.loss;
  }

  /**
   * Backward pass through the network
   */
  backward(): void {
    if (
      !this.state.a2 ||
      !this.state.target ||
      !this.state.a1 ||
      !this.state.z1 ||
      !this.state.input
    ) {
      throw new Error(
        "Must run forward pass and compute loss before backward pass",
      );
    }

    // Gradient of loss w.r.t output: dL/da2 = (a2 - target)
    // Using MSE/2, so gradient is simply (output - target) without the factor of 2
    const error = this.state.a2[0][0] - this.state.target[0][0];
    this.state.dz2 = [[error]];

    // Gradient w.r.t w2: dL/dw2 = a1.T @ dz2
    this.state.dw2 = matrixMultiply(transpose(this.state.a1), this.state.dz2);

    // Gradient w.r.t b2: dL/db2 = sum(dz2)
    this.state.db2 = this.state.dz2;

    // Gradient flowing back to hidden layer: da1 = dz2 @ w2.T
    this.state.da1 = matrixMultiply(this.state.dz2, transpose(this.state.w2));

    // Gradient at hidden layer (before ReLU): dz1 = da1 ⊙ relu'(z1)
    const reluGrad = reluDerivativeMatrix(this.state.z1);
    this.state.dz1 = this.state.da1.map((row, i) =>
      row.map((val, j) => val * reluGrad[i][j]),
    );

    // Gradient w.r.t w1: dL/dw1 = input.T @ dz1
    this.state.dw1 = matrixMultiply(
      transpose(this.state.input),
      this.state.dz1,
    );

    // Gradient w.r.t b1: dL/db1 = sum(dz1)
    this.state.db1 = this.state.dz1;
  }

  /**
   * Update weights using gradient descent
   */
  updateWeights(learningRate: number): void {
    if (
      !this.state.dw1 ||
      !this.state.db1 ||
      !this.state.dw2 ||
      !this.state.db2
    ) {
      throw new Error("Must run backward pass before updating weights");
    }

    // Update w1 and b1
    this.state.w1 = matrixAdd(
      this.state.w1,
      scalarMultiply(this.state.dw1, -learningRate),
    );
    this.state.b1 = matrixAdd(
      this.state.b1,
      scalarMultiply(this.state.db1, -learningRate),
    );

    // Update w2 and b2
    this.state.w2 = matrixAdd(
      this.state.w2,
      scalarMultiply(this.state.dw2, -learningRate),
    );
    this.state.b2 = matrixAdd(
      this.state.b2,
      scalarMultiply(this.state.db2, -learningRate),
    );
  }

  /**
   * Full training step: forward -> loss -> backward -> update
   */
  trainStep(
    input: number[][],
    target: number[][],
    learningRate: number,
  ): number {
    this.forward(input);
    const loss = this.computeLoss(target);
    this.backward();
    this.updateWeights(learningRate);
    return loss;
  }

  /**
   * Get current state for visualization
   */
  getState(): NetworkState {
    return { ...this.state };
  }

  /**
   * Reset network with new random weights
   */
  reset(): void {
    this.state = {
      w1: this.initializeWeights(2, 4),
      b1: [[0, 0, 0, 0]],
      w2: this.initializeWeights(4, 1),
      b2: [[0]],
    };
  }

  /**
   * Set specific weights (for manual control)
   */
  setWeights(
    w1: number[][],
    b1: number[][],
    w2: number[][],
    b2: number[][],
  ): void {
    this.state.w1 = w1;
    this.state.b1 = b1;
    this.state.w2 = w2;
    this.state.b2 = b2;
  }
}
