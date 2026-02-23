/**
 * Transformer Block
 * Combines self-attention with feed-forward network
 */

import { scaledDotProductAttention } from "./attention";
import { Matrix, matrixMultiply, matrixAdd } from "./matrix";

// Re-export Matrix type for other modules
export type { Matrix };

export interface TransformerStep {
  step: string;
  description: string;
  formula: string;
  values?: Matrix;
}

/**
 * Layer Normalization
 * Normalizes across the feature dimension
 */
export function layerNorm(x: Matrix, epsilon: number = 1e-5): Matrix {
  return x.map((row: number[]) => {
    const mean =
      row.reduce((sum: number, val: number) => sum + val, 0) / row.length;
    const variance =
      row.reduce((sum: number, val: number) => sum + (val - mean) ** 2, 0) /
      row.length;
    const std = Math.sqrt(variance + epsilon);
    return row.map((val: number) => (val - mean) / std);
  });
}

/**
 * Feed-forward network (2 layers with ReLU)
 * FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂
 */
export function feedForward(
  x: Matrix,
  W1: Matrix,
  b1: number[],
  W2: Matrix,
  b2: number[],
): Matrix {
  // First layer
  const hidden = matrixMultiply(x, W1).map((row: number[]) =>
    row.map((val: number, j: number) => Math.max(0, val + b1[j])),
  );

  // Second layer
  const output = matrixMultiply(hidden, W2).map((row: number[]) =>
    row.map((val: number, j: number) => val + b2[j]),
  );

  return output;
}

/**
 * Complete Transformer block
 * = LayerNorm(x + MultiHeadAttention(x))
 * + LayerNorm(x + FeedForward(x))
 */
export function transformerBlock(
  input: Matrix,
  W1_ff: Matrix,
  b1_ff: number[],
  W2_ff: Matrix,
  b2_ff: number[],
): { output: Matrix; attentionWeights: Matrix; intermediate: Matrix } {
  // Step 1: Self-attention with residual
  const attentionOutput = scaledDotProductAttention(input, input, input);
  const afterAttention = matrixAdd(input, attentionOutput);
  const normalized1 = layerNorm(afterAttention);

  // Step 2: Feed-forward with residual
  const ffOutput = feedForward(normalized1, W1_ff, b1_ff, W2_ff, b2_ff);
  const afterFF = matrixAdd(normalized1, ffOutput);
  const output = layerNorm(afterFF);

  // For visualization, we'll compute attention weights
  const Q = input;
  const K = input;
  const d_k = K[0].length;
  const scale = Math.sqrt(d_k);

  // Compute attention scores (Q · K for each query-key pair)
  const scores = Q.map((q) =>
    K.map((k) => q.reduce((sum, val, i) => sum + val * k[i], 0) / scale),
  );

  // Softmax to get weights
  const attentionWeights = scores.map((row) => {
    const max = Math.max(...row);
    const exp = row.map((val) => Math.exp(val - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map((val) => val / sum);
  });

  return {
    output,
    attentionWeights,
    intermediate: afterAttention,
  };
}

/**
 * Get step-by-step Transformer block computation
 */
export function getTransformerBlockSteps(
  input: Matrix,
  W1_ff: Matrix,
  b1_ff: number[],
  W2_ff: Matrix,
  b2_ff: number[],
): TransformerStep[] {
  const steps: TransformerStep[] = [];

  // Step 1: Input
  steps.push({
    step: "1",
    description: "Input embeddings",
    formula: `X ∈ ℝ^(seq_len × d_model)`,
    values: input,
  });

  // Step 2: Self-attention
  const attentionOutput = scaledDotProductAttention(input, input, input);
  steps.push({
    step: "2",
    description: "Multi-head self-attention",
    formula: `Attention(Q=X, K=X, V=X)`,
    values: attentionOutput,
  });

  // Step 3: Residual + Norm
  const afterAttention = matrixAdd(input, attentionOutput);
  steps.push({
    step: "3",
    description: "Add residual connection",
    formula: `X + Attention(X)`,
    values: afterAttention,
  });

  const normalized1 = layerNorm(afterAttention);
  steps.push({
    step: "4",
    description: "Layer normalization",
    formula: `LayerNorm(X + Attention(X))`,
    values: normalized1,
  });

  // Step 4: Feed-forward
  const ffOutput = feedForward(normalized1, W1_ff, b1_ff, W2_ff, b2_ff);
  steps.push({
    step: "5",
    description: "Feed-forward network",
    formula: `FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂`,
    values: ffOutput,
  });

  // Step 5: Final residual + norm
  const afterFF = matrixAdd(normalized1, ffOutput);
  steps.push({
    step: "6",
    description: "Add residual connection",
    formula: `X + FFN(X)`,
    values: afterFF,
  });

  const output = layerNorm(afterFF);
  steps.push({
    step: "7",
    description: "Final layer normalization",
    formula: `Output = LayerNorm(X + FFN(X))`,
    values: output,
  });

  return steps;
}

/**
 * Initialize Transformer weights
 */
export function initializeTransformerWeights(
  d_model: number,
  d_ff: number,
): {
  W1_ff: Matrix;
  b1_ff: number[];
  W2_ff: Matrix;
  b2_ff: number[];
} {
  const scale = 0.1;

  const W1_ff: Matrix = Array(d_model)
    .fill(0)
    .map(() =>
      Array(d_ff)
        .fill(0)
        .map(() => (Math.random() - 0.5) * scale),
    );

  const b1_ff: number[] = Array(d_ff).fill(0);

  const W2_ff: Matrix = Array(d_ff)
    .fill(0)
    .map(() =>
      Array(d_model)
        .fill(0)
        .map(() => (Math.random() - 0.5) * scale),
    );

  const b2_ff: number[] = Array(d_model).fill(0);

  return { W1_ff, b1_ff, W2_ff, b2_ff };
}

/**
 * Stack multiple Transformer blocks
 */
export function stackedTransformer(
  input: Matrix,
  numLayers: number,
  d_model: number,
  d_ff: number,
): { output: Matrix; allAttentionWeights: Matrix[] } {
  let current = input;
  const allAttentionWeights: Matrix[] = [];

  for (let layer = 0; layer < numLayers; layer++) {
    const weights = initializeTransformerWeights(d_model, d_ff);
    const result = transformerBlock(
      current,
      weights.W1_ff,
      weights.b1_ff,
      weights.W2_ff,
      weights.b2_ff,
    );
    current = result.output;
    allAttentionWeights.push(result.attentionWeights);
  }

  return {
    output: current,
    allAttentionWeights,
  };
}
