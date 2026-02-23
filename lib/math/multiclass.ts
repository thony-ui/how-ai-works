/**
 * Multi-class classification with a simple neural network
 * Uses a 2-layer network: Input → Hidden (ReLU) → Output (Softmax)
 */

import {
  softmax,
  crossEntropyLoss,
  softmaxCrossEntropyGradient,
} from "./softmax";
import { reluDerivative } from "./activations";
import {
  Matrix,
  matrixMultiply,
  transpose,
  matrixSubtract,
  scalarMultiply,
} from "./matrix";

export interface MultiClassDataPoint {
  x: number;
  y: number;
  class: number;
}

export interface NetworkWeights {
  W1: Matrix; // Input → Hidden
  b1: number[]; // Hidden bias
  W2: Matrix; // Hidden → Output
  b2: number[]; // Output bias
}

export interface ForwardPassResult {
  z1: number[]; // Hidden pre-activation
  a1: number[]; // Hidden activation
  z2: number[]; // Output pre-activation (logits)
  a2: number[]; // Output activation (probabilities)
}

export interface BackwardPassResult {
  dW2: Matrix;
  db2: number[];
  dW1: Matrix;
  db1: number[];
}

export interface TrainingStep {
  step: string;
  description: string;
  formula: string;
  values?: number[] | Matrix;
}

/**
 * Initialize network weights with small random values
 */
export function initializeWeights(
  inputSize: number,
  hiddenSize: number,
  outputSize: number,
): NetworkWeights {
  // Xavier/He initialization scaled down for educational purposes
  const scale1 = 0.5;
  const scale2 = 0.5;

  const W1: Matrix = Array(inputSize)
    .fill(0)
    .map(() =>
      Array(hiddenSize)
        .fill(0)
        .map(() => (Math.random() - 0.5) * scale1),
    );

  const b1: number[] = Array(hiddenSize).fill(0);

  const W2: Matrix = Array(hiddenSize)
    .fill(0)
    .map(() =>
      Array(outputSize)
        .fill(0)
        .map(() => (Math.random() - 0.5) * scale2),
    );

  const b2: number[] = Array(outputSize).fill(0);

  return { W1, b1, W2, b2 };
}

/**
 * Forward pass through the network
 */
export function forwardPass(
  input: number[],
  weights: NetworkWeights,
): ForwardPassResult {
  const { W1, b1, W2, b2 } = weights;

  // Layer 1: Input → Hidden
  const z1 = matrixMultiply([input], W1)[0].map(
    (val: number, i: number) => val + b1[i],
  );
  const a1 = z1.map((val: number) => Math.max(0, val)); // ReLU activation

  // Layer 2: Hidden → Output
  const z2 = matrixMultiply([a1], W2)[0].map(
    (val: number, i: number) => val + b2[i],
  );
  const a2 = softmax(z2);

  return { z1, a1, z2, a2 };
}

/**
 * Backward pass (backpropagation)
 */
export function backwardPass(
  input: number[],
  forward: ForwardPassResult,
  targetClass: number,
  weights: NetworkWeights,
): BackwardPassResult {
  const { z1, a1, a2 } = forward;
  const { W2 } = weights;

  // Output layer gradient (softmax + cross-entropy)
  const dz2 = softmaxCrossEntropyGradient(a2, targetClass);

  // Gradients for W2 and b2
  const dW2 = matrixMultiply(transpose([a1]), [dz2]);
  const db2 = dz2;

  // Gradient for hidden layer
  const dz1_pre = matrixMultiply([dz2], transpose(W2))[0];
  const dz1 = dz1_pre.map(
    (val: number, i: number) => val * reluDerivative(z1[i]),
  );

  // Gradients for W1 and b1
  const dW1 = matrixMultiply(transpose([input]), [dz1]);
  const db1 = dz1;

  return { dW2, db2, dW1, db1 };
}

/**
 * Update weights using gradient descent
 */
export function updateWeights(
  weights: NetworkWeights,
  gradients: BackwardPassResult,
  learningRate: number,
): NetworkWeights {
  const { W1, b1, W2, b2 } = weights;
  const { dW1, db1, dW2, db2 } = gradients;

  return {
    W1: matrixSubtract(W1, scalarMultiply(dW1, learningRate)),
    b1: b1.map((val: number, i: number) => val - learningRate * db1[i]),
    W2: matrixSubtract(W2, scalarMultiply(dW2, learningRate)),
    b2: b2.map((val: number, i: number) => val - learningRate * db2[i]),
  };
}

/**
 * Predict class for a single input
 */
export function predict(input: number[], weights: NetworkWeights): number {
  const { a2 } = forwardPass(input, weights);
  return a2.indexOf(Math.max(...a2));
}

/**
 * Compute accuracy on a dataset
 */
export function computeAccuracy(
  data: MultiClassDataPoint[],
  weights: NetworkWeights,
): number {
  let correct = 0;
  for (const point of data) {
    const predicted = predict([point.x, point.y], weights);
    if (predicted === point.class) correct++;
  }
  return correct / data.length;
}

/**
 * Train network for one epoch
 */
export function trainEpoch(
  data: MultiClassDataPoint[],
  weights: NetworkWeights,
  learningRate: number,
): { weights: NetworkWeights; loss: number } {
  let totalLoss = 0;
  let newWeights = weights;

  for (const point of data) {
    const input = [point.x, point.y];
    const forward = forwardPass(input, newWeights);
    const loss = crossEntropyLoss(forward.a2, point.class);
    totalLoss += loss;

    const gradients = backwardPass(input, forward, point.class, newWeights);
    newWeights = updateWeights(newWeights, gradients, learningRate);
  }

  return {
    weights: newWeights,
    loss: totalLoss / data.length,
  };
}

/**
 * Get step-by-step forward pass explanation
 */
export function getForwardPassSteps(
  input: number[],
  weights: NetworkWeights,
): TrainingStep[] {
  const forward = forwardPass(input, weights);

  return [
    {
      step: "1",
      description: "Input features",
      formula: `x = [${input.map((v) => v.toFixed(2)).join(", ")}]`,
      values: input,
    },
    {
      step: "2",
      description: "Hidden layer pre-activation",
      formula: `z₁ = x · W₁ + b₁`,
      values: forward.z1,
    },
    {
      step: "3",
      description: "Hidden layer activation (ReLU)",
      formula: `a₁ = ReLU(z₁) = max(0, z₁)`,
      values: forward.a1,
    },
    {
      step: "4",
      description: "Output layer pre-activation (logits)",
      formula: `z₂ = a₁ · W₂ + b₂`,
      values: forward.z2,
    },
    {
      step: "5",
      description: "Output probabilities (softmax)",
      formula: `p = softmax(z₂)`,
      values: forward.a2,
    },
  ];
}

/**
 * Get step-by-step backward pass explanation
 */
export function getBackwardPassSteps(
  input: number[],
  forward: ForwardPassResult,
  targetClass: number,
  weights: NetworkWeights,
): TrainingStep[] {
  const gradients = backwardPass(input, forward, targetClass, weights);

  return [
    {
      step: "1",
      description: "Output gradient (softmax + cross-entropy)",
      formula: `∂L/∂z₂ = p - y (where y is one-hot target)`,
      values: softmaxCrossEntropyGradient(forward.a2, targetClass),
    },
    {
      step: "2",
      description: "Output weight gradient",
      formula: `∂L/∂W₂ = a₁ᵀ · ∂L/∂z₂`,
      values: gradients.dW2,
    },
    {
      step: "3",
      description: "Hidden layer gradient",
      formula: `∂L/∂a₁ = ∂L/∂z₂ · W₂ᵀ`,
    },
    {
      step: "4",
      description: "Hidden pre-activation gradient (ReLU backprop)",
      formula: `∂L/∂z₁ = ∂L/∂a₁ ⊙ ReLU'(z₁)`,
    },
    {
      step: "5",
      description: "Input weight gradient",
      formula: `∂L/∂W₁ = xᵀ · ∂L/∂z₁`,
      values: gradients.dW1,
    },
  ];
}

/**
 * Generate synthetic dataset for classification
 */
export function generateDataset(
  numClasses: number,
  pointsPerClass: number,
  spread: number = 1.5,
): MultiClassDataPoint[] {
  const data: MultiClassDataPoint[] = [];
  const angleStep = (2 * Math.PI) / numClasses;

  for (let classIdx = 0; classIdx < numClasses; classIdx++) {
    const angle = classIdx * angleStep;
    const centerX = Math.cos(angle) * 3;
    const centerY = Math.sin(angle) * 3;

    for (let i = 0; i < pointsPerClass; i++) {
      // Add some randomness around the center
      const noise = () => (Math.random() - 0.5) * spread;
      data.push({
        x: centerX + noise(),
        y: centerY + noise(),
        class: classIdx,
      });
    }
  }

  // Shuffle
  for (let i = data.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [data[i], data[j]] = [data[j], data[i]];
  }

  return data;
}
