/**
 * Softmax and Cross-Entropy functions
 */

/**
 * Compute softmax of a vector
 * softmax(x_i) = exp(x_i) / sum(exp(x_j))
 */
export function softmax(logits: number[]): number[] {
  // Subtract max for numerical stability
  const maxLogit = Math.max(...logits);
  const expLogits = logits.map((x) => Math.exp(x - maxLogit));
  const sumExp = expLogits.reduce((a, b) => a + b, 0);
  return expLogits.map((x) => x / sumExp);
}

/**
 * Compute cross-entropy loss
 * L = -sum(y_true * log(y_pred))
 * For one-hot encoded labels, this simplifies to -log(y_pred[correct_class])
 */
export function crossEntropyLoss(
  predictions: number[],
  targetIndex: number,
): number {
  // Clip predictions to avoid log(0)
  const epsilon = 1e-10;
  const clippedPred = Math.max(
    epsilon,
    Math.min(1 - epsilon, predictions[targetIndex]),
  );
  return -Math.log(clippedPred);
}

/**
 * Compute cross-entropy loss with full probability distribution
 */
export function crossEntropyLossFull(
  predictions: number[],
  targets: number[],
): number {
  const epsilon = 1e-10;
  let loss = 0;
  for (let i = 0; i < predictions.length; i++) {
    if (targets[i] > 0) {
      const clippedPred = Math.max(
        epsilon,
        Math.min(1 - epsilon, predictions[i]),
      );
      loss -= targets[i] * Math.log(clippedPred);
    }
  }
  return loss;
}

/**
 * Compute gradient of cross-entropy loss w.r.t. logits (combined with softmax)
 * For softmax + cross-entropy, gradient is simply: predictions - targets
 */
export function softmaxCrossEntropyGradient(
  predictions: number[],
  targetIndex: number,
): number[] {
  const gradients = [...predictions];
  gradients[targetIndex] -= 1;
  return gradients;
}

/**
 * Get step-by-step softmax calculation
 */
export interface SoftmaxStep {
  step: number;
  description: string;
  values: number[];
  formula: string;
}

export function getSoftmaxSteps(logits: number[]): SoftmaxStep[] {
  const steps: SoftmaxStep[] = [];

  // Step 1: Original logits
  steps.push({
    step: 1,
    description: "Input logits (raw scores)",
    values: [...logits],
    formula: `z = [${logits.map((v) => v.toFixed(3)).join(", ")}]`,
  });

  // Step 2: Find max for numerical stability
  const maxLogit = Math.max(...logits);
  steps.push({
    step: 2,
    description: "Find maximum value (for numerical stability)",
    values: [maxLogit],
    formula: `max(z) = ${maxLogit.toFixed(3)}`,
  });

  // Step 3: Subtract max
  const shifted = logits.map((x) => x - maxLogit);
  steps.push({
    step: 3,
    description: "Subtract max from each logit",
    values: shifted,
    formula: `z' = z - max(z) = [${shifted.map((v) => v.toFixed(3)).join(", ")}]`,
  });

  // Step 4: Compute exponentials
  const expLogits = shifted.map((x) => Math.exp(x));
  steps.push({
    step: 4,
    description: "Compute exponentials",
    values: expLogits,
    formula: `exp(z') = [${expLogits.map((v) => v.toFixed(3)).join(", ")}]`,
  });

  // Step 5: Sum of exponentials
  const sumExp = expLogits.reduce((a, b) => a + b, 0);
  steps.push({
    step: 5,
    description: "Sum of exponentials",
    values: [sumExp],
    formula: `sum(exp(z')) = ${sumExp.toFixed(3)}`,
  });

  // Step 6: Divide by sum (final probabilities)
  const probabilities = expLogits.map((x) => x / sumExp);
  steps.push({
    step: 6,
    description: "Divide by sum to get probabilities",
    values: probabilities,
    formula: `softmax(z) = exp(z') / sum(exp(z')) = [${probabilities.map((v) => v.toFixed(4)).join(", ")}]`,
  });

  return steps;
}

/**
 * Get step-by-step cross-entropy calculation
 */
export interface CrossEntropyStep {
  step: number;
  description: string;
  value: number;
  formula: string;
}

export function getCrossEntropySteps(
  predictions: number[],
  targetIndex: number,
): CrossEntropyStep[] {
  const steps: CrossEntropyStep[] = [];
  const epsilon = 1e-10;

  // Step 1: Show predictions
  steps.push({
    step: 1,
    description: "Input probabilities",
    value: predictions[targetIndex],
    formula: `p = [${predictions.map((v) => v.toFixed(4)).join(", ")}]`,
  });

  // Step 2: Extract correct class probability
  steps.push({
    step: 2,
    description: `Probability of correct class (index ${targetIndex})`,
    value: predictions[targetIndex],
    formula: `p[${targetIndex}] = ${predictions[targetIndex].toFixed(4)}`,
  });

  // Step 3: Compute log
  const clippedPred = Math.max(
    epsilon,
    Math.min(1 - epsilon, predictions[targetIndex]),
  );
  const logProb = Math.log(clippedPred);
  steps.push({
    step: 3,
    description: "Compute log probability",
    value: logProb,
    formula: `log(p[${targetIndex}]) = ${logProb.toFixed(4)}`,
  });

  // Step 4: Negate to get loss
  const loss = -logProb;
  steps.push({
    step: 4,
    description: "Negate to get loss",
    value: loss,
    formula: `L = -log(p[${targetIndex}]) = ${loss.toFixed(4)}`,
  });

  return steps;
}
