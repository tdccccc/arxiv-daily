import OpenAI from "openai";
import { retry } from "../utils/retry";
import type { Logger } from "../services/logger";
import type { LlmSettings } from "../settings/types";
import { isCancellationError, throwIfCancelled } from "../services/cancellation";

const LLM_TIMEOUT_MS = 300_000; // 5 minutes
export const LLM_TEMPERATURE = 0.1;

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface CallOptions {
  /** Overrides default temperature. Ignored when thinkingMode = true. */
  temperature?: number;
  signal?: AbortSignal;
}

const KNOWN_MODEL_BASE_SUFFIXES = [
  "/api/claudecode",
  "/api/anthropic",
  "/apps/anthropic",
  "/api/coding",
  "/claudecode",
  "/anthropic",
  "/step_plan",
  "/coding",
  "/claude",
];

export function buildModelUrlCandidates(baseUrl: string): string[] {
  const normalized = baseUrl.replace(/\/+$/, "");
  const candidates: string[] = [];
  const addCandidate = (url: string): void => {
    if (!candidates.includes(url)) candidates.push(url);
  };

  if (normalized.endsWith("/v1")) {
    addCandidate(`${normalized}/models`);
  } else {
    addCandidate(`${normalized}/v1/models`);
  }

  for (const suffix of KNOWN_MODEL_BASE_SUFFIXES) {
    if (normalized.endsWith(suffix)) {
      const stripped = normalized.slice(0, -suffix.length);
      addCandidate(`${stripped}/v1/models`);
      break;
    }
  }

  addCandidate(`${normalized}/models`);
  return candidates;
}

export class LlmClient {
  private client: OpenAI;

  constructor(private settings: LlmSettings, private logger: Logger) {
    this.client = new OpenAI({
      apiKey: settings.apiKey,
      baseURL: settings.baseUrl,
      timeout: LLM_TIMEOUT_MS,
      maxRetries: 0,
      dangerouslyAllowBrowser: true,
    });
  }

  async testConnection(): Promise<{ success: boolean; error?: string }> {
    try {
      await this.client.chat.completions.create({
        model: this.settings.model,
        messages: [{ role: "user", content: "Hello" }],
        max_tokens: 5,
      });
      return { success: true };
    } catch (e) {
      return { success: false, error: (e as Error).message };
    }
  }

  async fetchModels(): Promise<string[]> {
    const baseUrl = this.settings.baseUrl.replace(/\/+$/, "");
    const apiKey = this.settings.apiKey;

    if (!baseUrl || !apiKey) {
      throw new Error("Please fill in API Base URL and API Key first");
    }

    const candidates = buildModelUrlCandidates(baseUrl);

    for (const url of candidates) {
      try {
        const response = await fetch(url, {
          method: "GET",
          headers: {
            "Authorization": `Bearer ${apiKey}`,
            "Content-Type": "application/json",
          },
        });

        if (response.ok) {
          const data = await response.json();
          return this.parseModelList(data);
        }
      } catch {
        // Try next candidate
        continue;
      }
    }

    throw new Error("Failed to fetch models from any endpoint");
  }

  private parseModelList(data: unknown): string[] {
    if (
      data &&
      typeof data === "object" &&
      "data" in data &&
      Array.isArray((data as { data: unknown }).data)
    ) {
      return (data as { data: Array<{ id?: string }> }).data
        .map((model) => model.id)
        .filter((id): id is string => Boolean(id));
    }
    if (Array.isArray(data)) {
      return data
        .map((model: { id?: string; name?: string }) => model.id || model.name)
        .filter((id): id is string => Boolean(id));
    }
    throw new Error("Invalid model list format");
  }

  async call(messages: ChatMessage[], opts: CallOptions = {}): Promise<string> {
    return retry(
      async () => {
        throwIfCancelled(opts.signal);
        const params: Record<string, unknown> = {
          model: this.settings.model,
          messages,
          stream: true,
        };
        if (this.settings.thinkingMode) {
          if (this.settings.provider === "anthropic") {
            // Anthropic extended thinking via OpenAI-compat proxy
            const budgets: Record<string, number> = {
              low: 2048,
              medium: 8192,
              high: 16384,
            };
            (params as any).extra_body = {
              thinking: {
                type: "enabled",
                budget_tokens: budgets[this.settings.reasoningEffort] ?? 8192,
              },
            };
          } else {
            params.reasoning_effort = this.settings.reasoningEffort;
            (params as any).extra_body = { thinking: { type: "enabled" } };
          }
        } else {
          params.temperature = opts.temperature ?? LLM_TEMPERATURE;
        }
        const stream = await this.client.chat.completions.create(params as any, {
          signal: opts.signal,
        } as any);
        const chunks: string[] = [];
        for await (const chunk of stream as any) {
          throwIfCancelled(opts.signal);
          const delta = chunk.choices?.[0]?.delta?.content;
          if (delta) chunks.push(delta);
        }
        throwIfCancelled(opts.signal);
        return chunks.join("");
      },
      {
        maxAttempts: 3,
        baseDelayMs: 5000,
        signal: opts.signal,
        shouldRetry: (err) => !isCancellationError(err),
        onRetry: (err, attempt, wait) =>
          this.logger.warn(
            `LLM retry #${attempt} after ${wait}ms: ${(err as Error).message}`,
          ),
      },
    );
  }
}
