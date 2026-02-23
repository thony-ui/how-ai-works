/**
 * Next-Token Prediction
 * Core task of autoregressive language models
 */

import { softmax } from "./softmax";

export interface Vocabulary {
  tokens: string[];
  tokenToId: Map<string, number>;
  idToToken: Map<number, string>;
}

export interface PredictionResult {
  topK: Array<{ token: string; probability: number; logit: number }>;
  fullDistribution: number[];
}

/**
 * Create a simple vocabulary
 */
export function createVocabulary(tokens: string[]): Vocabulary {
  const tokenToId = new Map<string, number>();
  const idToToken = new Map<number, string>();

  tokens.forEach((token, id) => {
    tokenToId.set(token, id);
    idToToken.set(id, token);
  });

  return { tokens, tokenToId, idToToken };
}

/**
 * Convert tokens to IDs
 */
export function tokenize(text: string[], vocab: Vocabulary): number[] {
  return text.map((token) => vocab.tokenToId.get(token) ?? 0);
}

/**
 * Convert IDs to tokens
 */
export function detokenize(ids: number[], vocab: Vocabulary): string[] {
  return ids.map((id) => vocab.idToToken.get(id) ?? "<UNK>");
}

/**
 * Predict next token given logits
 */
export function predictNextToken(
  logits: number[],
  vocab: Vocabulary,
  topK: number = 5,
): PredictionResult {
  // Apply softmax to get probabilities
  const probabilities = softmax(logits);

  // Get top-K predictions
  const indexed = probabilities.map((prob, idx) => ({
    token: vocab.idToToken.get(idx) ?? "<UNK>",
    probability: prob,
    logit: logits[idx],
    index: idx,
  }));

  indexed.sort((a, b) => b.probability - a.probability);

  const topKResults = indexed.slice(0, topK);

  return {
    topK: topKResults,
    fullDistribution: probabilities,
  };
}

/**
 * Temperature sampling for more diverse predictions
 */
export function applyTemperature(
  logits: number[],
  temperature: number,
): number[] {
  if (temperature === 0) {
    // Greedy: return one-hot for argmax
    const maxIdx = logits.indexOf(Math.max(...logits));
    return logits.map((_, i) => (i === maxIdx ? 100 : -100));
  }
  return logits.map((logit) => logit / temperature);
}

/**
 * Top-K filtering: keep only top K logits
 */
export function topKFiltering(logits: number[], k: number): number[] {
  const indexed = logits.map((logit, idx) => ({ logit, idx }));
  indexed.sort((a, b) => b.logit - a.logit);

  const threshold = indexed[Math.min(k - 1, indexed.length - 1)].logit;

  return logits.map((logit) => (logit >= threshold ? logit : -Infinity));
}

/**
 * Top-P (nucleus) filtering: keep top tokens with cumulative probability >= p
 */
export function topPFiltering(logits: number[], p: number): number[] {
  // Get probabilities
  const probs = softmax(logits);

  // Sort by probability
  const indexed = probs.map((prob, idx) => ({ prob, idx, logit: logits[idx] }));
  indexed.sort((a, b) => b.prob - a.prob);

  // Find cumulative probability threshold
  let cumulative = 0;
  let cutoffIdx = indexed.length;
  for (let i = 0; i < indexed.length; i++) {
    cumulative += indexed[i].prob;
    if (cumulative >= p) {
      cutoffIdx = i + 1;
      break;
    }
  }

  // Filter logits
  const allowedIndices = new Set(indexed.slice(0, cutoffIdx).map((x) => x.idx));
  return logits.map((logit, idx) =>
    allowedIndices.has(idx) ? logit : -Infinity,
  );
}

/**
 * Sample from distribution
 */
export function sampleFromDistribution(probabilities: number[]): number {
  const random = Math.random();
  let cumulative = 0;

  for (let i = 0; i < probabilities.length; i++) {
    cumulative += probabilities[i];
    if (random <= cumulative) {
      return i;
    }
  }

  return probabilities.length - 1;
}

/**
 * Generate text autoregressively
 */
export function generateText(
  initialTokenIds: number[],
  getLogits: (tokenIds: number[]) => number[],
  vocab: Vocabulary,
  maxNewTokens: number,
  temperature: number = 1.0,
  topK?: number,
  topP?: number,
): string[] {
  const generated = [...initialTokenIds];

  for (let i = 0; i < maxNewTokens; i++) {
    // Get logits for next token
    let logits = getLogits(generated);

    // Apply temperature
    logits = applyTemperature(logits, temperature);

    // Apply top-K filtering
    if (topK !== undefined) {
      logits = topKFiltering(logits, topK);
    }

    // Apply top-P filtering
    if (topP !== undefined) {
      logits = topPFiltering(logits, topP);
    }

    // Get probabilities and sample
    const probs = softmax(logits);
    const nextTokenId = sampleFromDistribution(probs);

    generated.push(nextTokenId);
  }

  return detokenize(generated, vocab);
}

/**
 * Simple mock model for demonstration
 * In practice, this would be a neural network
 */
export function mockLanguageModel(
  tokenIds: number[],
  vocab: Vocabulary,
): number[] {
  // Create simple heuristic logits based on last token
  const vocabSize = vocab.tokens.length;
  const logits = Array(vocabSize).fill(0);

  if (tokenIds.length === 0) {
    // Start of sequence - bias toward common starting words
    vocab.tokens.forEach((token, idx) => {
      if (["The", "A", "I", "She", "He"].includes(token)) {
        logits[idx] = 2.0;
      } else {
        logits[idx] = (Math.random() - 0.5) * 0.5;
      }
    });
  } else {
    // Based on last token, create biased distribution
    const lastTokenId = tokenIds[tokenIds.length - 1];
    const lastToken = vocab.idToToken.get(lastTokenId);

    // Simple co-occurrence heuristics
    const transitions: Record<string, string[]> = {
      The: ["cat", "dog", "quick", "lazy"],
      cat: ["sat", "ran", "jumped", "slept"],
      dog: ["barked", "ran", "jumped", "slept"],
      quick: ["brown", "red", "fox", "cat"],
      sat: ["on", "down", "still"],
      on: ["the", "a", "top"],
    };

    const preferredNext = transitions[lastToken || ""] || [];

    vocab.tokens.forEach((token, idx) => {
      if (preferredNext.includes(token)) {
        logits[idx] = 3.0 + Math.random();
      } else {
        logits[idx] = (Math.random() - 0.5) * 2.0;
      }
    });
  }

  return logits;
}
