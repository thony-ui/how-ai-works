/**
 * Attention Mechanism (Scaled Dot-Product Attention)
 * Used in Transformers for processing sequences
 */

import { Matrix, matrixMultiply, transpose, scalarMultiply } from "./matrix";
import { softmax } from "./softmax";

// Re-export Matrix type for other modules
export type { Matrix };

export interface AttentionStep {
  step: string;
  description: string;
  formula: string;
  values?: number[][] | number[];
}

/**
 * Compute scaled dot-product attention
 * Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
 *
 * @param Q Query matrix (seq_len × d_k)
 * @param K Key matrix (seq_len × d_k)
 * @param V Value matrix (seq_len × d_v)
 * @returns Attention output (seq_len × d_v)
 */
export function scaledDotProductAttention(
  Q: Matrix,
  K: Matrix,
  V: Matrix,
): Matrix {
  // 1. Compute Q * K^T
  const KT = transpose(K);
  const scores = matrixMultiply(Q, KT);

  // 2. Scale by sqrt(d_k)
  const d_k = K[0].length;
  const scale = Math.sqrt(d_k);
  const scaledScores = scalarMultiply(scores, 1 / scale);

  // 3. Apply softmax row-wise (each query attends to all keys)
  const attentionWeights = scaledScores.map((row: number[]) => softmax(row));

  // 4. Multiply by V
  const output = matrixMultiply(attentionWeights, V);

  return output;
}

/**
 * Get step-by-step attention computation
 */
export function getAttentionSteps(
  Q: Matrix,
  K: Matrix,
  V: Matrix,
): { steps: AttentionStep[]; attentionWeights: Matrix; output: Matrix } {
  const steps: AttentionStep[] = [];

  // Step 1: Input matrices
  steps.push({
    step: "1",
    description: "Input Query (Q), Key (K), Value (V) matrices",
    formula: `Q: ${Q.length}×${Q[0].length}, K: ${K.length}×${K[0].length}, V: ${V.length}×${V[0].length}`,
    values: Q,
  });

  // Step 2: Compute scores (Q * K^T)
  const KT = transpose(K);
  const scores = matrixMultiply(Q, KT);
  steps.push({
    step: "2",
    description: "Compute attention scores",
    formula: `Scores = Q · Kᵀ`,
    values: scores,
  });

  // Step 3: Scale scores
  const d_k = K[0].length;
  const scale = Math.sqrt(d_k);
  const scaledScores = scalarMultiply(scores, 1 / scale);
  steps.push({
    step: "3",
    description: "Scale by square root of key dimension",
    formula: `Scaled Scores = Scores / √d_k = Scores / √${d_k} = Scores / ${scale.toFixed(2)}`,
    values: scaledScores,
  });

  // Step 4: Apply softmax
  const attentionWeights = scaledScores.map((row: number[]) => softmax(row));
  steps.push({
    step: "4",
    description: "Apply softmax to get attention weights",
    formula: `Attention Weights = softmax(Scaled Scores)`,
    values: attentionWeights,
  });

  // Step 5: Weighted sum with values
  const output = matrixMultiply(attentionWeights, V);
  steps.push({
    step: "5",
    description: "Compute weighted sum of values",
    formula: `Output = Attention Weights · V`,
    values: output,
  });

  return { steps, attentionWeights, output };
}

/**
 * Visualize attention weights as a heatmap-friendly structure
 */
export function createAttentionHeatmap(
  attentionWeights: Matrix,
  queryLabels: string[],
  keyLabels: string[],
): { queries: string[]; keys: string[]; weights: number[][] } {
  return {
    queries: queryLabels,
    keys: keyLabels,
    weights: attentionWeights,
  };
}

/**
 * Self-attention: Q, K, V all come from the same input
 * In practice, we project the input using learned weight matrices
 */
export function selfAttention(
  input: Matrix,
  W_q: Matrix,
  W_k: Matrix,
  W_v: Matrix,
): Matrix {
  const Q = matrixMultiply(input, W_q);
  const K = matrixMultiply(input, W_k);
  const V = matrixMultiply(input, W_v);

  return scaledDotProductAttention(Q, K, V);
}

/**
 * Multi-head attention: Run multiple attention heads in parallel
 * Each head learns different aspects of the relationships
 */
export function multiHeadAttention(
  Q: Matrix,
  K: Matrix,
  V: Matrix,
  numHeads: number,
  d_model: number,
): Matrix {
  const d_k = Math.floor(d_model / numHeads);
  const heads: Matrix[] = [];

  for (let h = 0; h < numHeads; h++) {
    // Split Q, K, V into smaller dimensions for this head
    const startIdx = h * d_k;
    const endIdx = startIdx + d_k;

    const Q_h = Q.map((row) => row.slice(startIdx, endIdx));
    const K_h = K.map((row) => row.slice(startIdx, endIdx));
    const V_h = V.map((row) => row.slice(startIdx, endIdx));

    // Run attention for this head
    const head = scaledDotProductAttention(Q_h, K_h, V_h);
    heads.push(head);
  }

  // Concatenate all heads
  const concat: Matrix = heads[0].map((row, i) =>
    heads.flatMap((head) => head[i]),
  );

  return concat;
}

/**
 * Create simple embedding matrix
 * In practice, this would be learned
 */
export function createSimpleEmbedding(dim: number): Matrix {
  const vocab = ["Hello", "World", "AI", "is", "amazing"];
  return vocab.map(() =>
    Array(dim)
      .fill(0)
      .map(() => (Math.random() - 0.5) * 0.5),
  );
}

/**
 * Add positional encoding to embeddings
 * Uses sine and cosine functions of different frequencies
 */
export function addPositionalEncoding(embeddings: Matrix): Matrix {
  const d_model = embeddings[0].length;

  return embeddings.map((embedding, pos) => {
    return embedding.map((val, i) => {
      const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / d_model);
      const posEncoding = i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
      return val + posEncoding * 0.1; // Scale down for visualization
    });
  });
}

/**
 * Compute attention between specific query and all keys
 * Useful for visualization
 */
export function computeQueryAttention(
  queryIdx: number,
  Q: Matrix,
  K: Matrix,
  V: Matrix,
): { scores: number[]; weights: number[]; output: number[] } {
  const query = Q[queryIdx];
  const d_k = K[0].length;
  const scale = Math.sqrt(d_k);

  // Compute scores for this query against all keys
  const scores = K.map(
    (key) => query.reduce((sum, val, i) => sum + val * key[i], 0) / scale,
  );

  // Apply softmax to get attention weights
  const weights = softmax(scores);

  // Compute output as weighted sum of values
  const output = V[0].map((_: number, dim: number) =>
    weights.reduce(
      (sum: number, weight: number, i: number) => sum + weight * V[i][dim],
      0,
    ),
  );

  return { scores, weights, output };
}
