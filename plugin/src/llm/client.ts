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
