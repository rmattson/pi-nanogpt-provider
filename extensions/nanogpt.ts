import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { OAuthCredentials, OAuthLoginCallbacks } from "@earendil-works/pi-ai";

// SPDX-License-Identifier: MIT

/**
 * NanoGPT Provider for Pi
 *
 * Copyright (c) 2026 Mark Gaiser <markg85@gmail.com>
 *
 * Registers the NanoGPT API (https://nano-gpt.com/api/v1) as a Pi provider.
 * NanoGPT offers 600+ models via an OpenAI-compatible Chat Completions API.
 * All models and specs are fetched dynamically from /v1/models.
 *
 * Authentication options (in priority order):
 *   1. /login nanogpt  — prompts for API key, stored in Pi's credential store
 *   2. NANOGPT_API_KEY env var
 */

// ---------------------------------------------------------------------------
// Model filtering – skip non-chat / search-only models
// ---------------------------------------------------------------------------
const SKIP_PATTERNS = [
  /^exa-/,                 // Exa search models
  /^brave/,                // Brave search models
  /^fastgpt$/,             // Kagi search model
  /^universal-summarizer$/, // Kagi summarizer
  /^sonar/,                // Perplexity search models
  /^v0-/,                  // Vercel v0 (specialized code gen)
];

// ---------------------------------------------------------------------------
// Heuristics for model capabilities (inferred from model ID naming patterns)
// ---------------------------------------------------------------------------

/** Detect reasoning models (:thinking suffix or known reasoning families) */
function isReasoningModel(id: string): boolean {
  if (/:thinking/.test(id)) return true;
  // Known reasoning model families without :thinking suffix
  if (/^openai\/o[34]/.test(id)) return true;
  if (/^deepseek-r1$|^deepseek\/deepseek-v4/.test(id)) return true;
  if (/^deepseek-ai\/DeepSeek-R1/.test(id)) return true;
  if (/^qwen\/qwen3/.test(id)) return true;
  if (/^gemini-2\.5-/.test(id)) return true;
  if (/^google\/gemini-[23]/.test(id)) return true;
  return false;
}

/** Detect vision/multimodal models from naming patterns */
function isVisionModel(id: string): boolean {
  const v: RegExp[] = [
    /[-/]vl[-/]/i, /-vl$/i, /[-_]vision/i,
    /^openai\/gpt-4o/i, /^openai\/gpt-5/i, /^openai\/gpt-4\.1/i,
    /^anthropic\/claude/i, /^claude-/i,
    /^gemini-/i, /^google\/gemini/i,
    /gpt-4o/i,
    /qvq-/i, /qwen[-/].*vl/i,
    /minimax.*vl/i,
    /mimo-v2-omni/i,
    /glm[-.]?\d+[-.]?\d*v/i, /glm-5v/i,
    /doubao.*vision/i, /ernie.*vl/i,
    /gemma.*4.*31b.*it/i,
  ];
  return v.some((p) => p.test(id));
}

/** Guess context window from model name */
function guessContextWindow(id: string): number {
  if (/gpt-4\.1|llama-4-scout|llama-4-maverick/.test(id)) return 1_048_576;
  if (/gemini/.test(id)) return 1_048_576;
  if (/gpt-5\.|^openai\/o[34]/.test(id)) return 200_000;
  if (/claude|opus|sonnet|haiku/.test(id)) return 200_000;
  if (/deepseek.*v4|deepseek-r1|qwen3|kimi-k2/.test(id)) return 128_000;
  if (/70[bB]|72[bB]|405[bB]|235[bB]|397[bB]|675[bB]/.test(id)) return 128_000;
  if (/32[bB]|27[bB]|31[bB]|24[bB]|26[bB]/.test(id)) return 32_000;
  if (/8[bB]|9[bB]|12[bB]|14[bB]/.test(id)) return 16_000;
  if (/3[bB]|4[bB]/.test(id)) return 8_000;
  return 128_000;
}

/** Guess max output tokens from model name */
function guessMaxTokens(id: string): number {
  if (/gemini/.test(id)) return 65_536;
  if (/^openai\/o[34]/.test(id)) return 100_000;
  if (/gpt-5|gpt-4\.1/.test(id)) return 32_768;
  if (/deepseek.*v4|deepseek-r1/.test(id)) return 16_384;
  return 16_384;
}

/** Format a human-readable display name from the model ID */
function formatDisplayName(id: string): string {
  return id
    .replace(/:thinking$/, " (thinking)")
    .replace(/:thinking:(low|medium|high|max)/, " (thinking $1)");
}

// ---------------------------------------------------------------------------
// Model discovery
// ---------------------------------------------------------------------------

async function fetchModels(
  apiKey?: string
): Promise<Array<{
  id: string;
  owned_by: string;
  context_length?: number;
  max_output_tokens?: number;
  capabilities?: { vision?: boolean; reasoning?: boolean };
}>> {
  const headers: Record<string, string> = {};
  if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;

  try {
    // ?detailed=true makes NanoGPT return per-model specs (context_length,
    // max_output_tokens, capabilities, pricing, ...). Without it the payload
    // only contains id/object/created/owned_by.
    const res = await fetch("https://nano-gpt.com/api/v1/models?detailed=true", { headers });
    if (res.ok) {
      const payload = (await res.json()) as {
        data: Array<{
          id: string;
          object: string;
          created: number;
          owned_by: string;
          context_length?: number;
          max_output_tokens?: number;
          capabilities?: { vision?: boolean; reasoning?: boolean };
        }>;
      };
      const models = payload.data ?? [];
      if (models.length > 0) return models;
    }
  } catch {
    // Network error – fall through
  }

  // If the API was unreachable, return empty — no static fallback
  return [];
}

/** True when v is a usable positive number (API values may be 0/undefined). */
function isPositiveNumber(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v) && v > 0;
}

function buildModelList(
  remoteModels: Array<{
    id: string;
    owned_by: string;
    context_length?: number;
    max_output_tokens?: number;
    capabilities?: { vision?: boolean; reasoning?: boolean };
  }>
) {
  return remoteModels
    .filter((m) => !SKIP_PATTERNS.some((p) => p.test(m.id)))
    .map((m) => {
      const caps = m.capabilities;
      return {
        id: m.id,
        name: formatDisplayName(m.id),
        // Prefer the API's capability flags. The :thinking ID suffix is kept
        // as a hard signal (the API misses a few of those), and the name
        // heuristics only apply when the payload has no capability info.
        reasoning:
          caps?.reasoning === true ||
          /:thinking/.test(m.id) ||
          (caps?.reasoning === undefined && isReasoningModel(m.id)),
        input:
          (caps?.vision === undefined ? isVisionModel(m.id) : caps.vision)
            ? (["text", "image"] as const)
            : (["text"] as const),
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        // Prefer the values reported by NanoGPT's API payload; fall back to
        // the name-based heuristics when they're missing or invalid.
        contextWindow: isPositiveNumber(m.context_length)
          ? m.context_length
          : guessContextWindow(m.id),
        maxTokens: isPositiveNumber(m.max_output_tokens)
          ? m.max_output_tokens
          : guessMaxTokens(m.id),
      };
    });
}

// ---------------------------------------------------------------------------
// Main extension
// ---------------------------------------------------------------------------
export default async function (pi: ExtensionAPI) {
  const envApiKey = process.env.NANOGPT_API_KEY || "";

  // Discover models using whatever credentials we have right now.
  // The /login flow (below) will re-register the provider with a fresh key,
  // so we don't need to block on OAuth here.
  const remoteModels = await fetchModels(envApiKey || undefined);
  const models = buildModelList(remoteModels);

  pi.registerProvider("nanogpt", {
    name: "NanoGPT",
    baseUrl: "https://nano-gpt.com/api/v1",
    apiKey: "$NANOGPT_API_KEY",
    api: "openai-completions",
    authHeader: true,
    models,

    // ── Interactive /login support ──────────────────────────────────────
    // Lets users run `/login nanogpt` to enter (or update) their API key.
    // Pi stores the resulting credentials in ~/.pi/agent/auth.json and
    // resolves them before every request — no env var needed.
    oauth: {
      name: "NanoGPT",

      async login(callbacks: OAuthLoginCallbacks): Promise<OAuthCredentials> {
        const apiKey = await callbacks.onPrompt({
          message: "Enter your NanoGPT API key (from https://nano-gpt.com/settings/api-keys):",
        });

        if (!apiKey) throw new Error("Login cancelled");

        // Validate the key by trying to fetch the model list
        const res = await fetch("https://nano-gpt.com/api/v1/models", {
          headers: { Authorization: `Bearer ${apiKey}` },
        });

        if (!res.ok) {
          throw new Error(
            `API key validation failed (HTTP ${res.status}). Check your key at https://nano-gpt.com/settings/api-keys`
          );
        }

        // Store the API key as a long-lived credential.
        // NanoGPT keys don't expire, so we set a far-future expiry.
        const FAR_FUTURE = Date.now() + 10 * 365 * 24 * 60 * 60 * 1000; // ~10 years

        return {
          refresh: apiKey,
          access: apiKey,
          expires: FAR_FUTURE,
        };
      },

      async refreshToken(credentials: OAuthCredentials): Promise<OAuthCredentials> {
        // NanoGPT API keys are static — just return them as-is.
        return credentials;
      },

      getApiKey(credentials: OAuthCredentials): string {
        return credentials.access;
      },
    },
  });
}
