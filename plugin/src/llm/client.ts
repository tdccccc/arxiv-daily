import OpenAI from "openai";
import { retry } from "../utils/retry";
import type { Logger } from "../services/logger";
import type { LlmSettings } from "../settings/types";

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface CallOptions {
  /** Overrides settings.temperature. Ignored when thinkingMode = true. */
  temperature?: number;
}

export class LlmClient {
  private client: OpenAI;

  constructor(private settings: LlmSettings, private logger: Logger) {
    this.client = new OpenAI({
      apiKey: settings.apiKey,
      baseURL: settings.baseUrl,
      timeout: settings.timeoutMs,
      maxRetries: 0,
      dangerouslyAllowBrowser: true,
    });
  }

  async call(messages: ChatMessage[], opts: CallOptions = {}): Promise<string> {
    return retry(
      async () => {
        const params: Record<string, unknown> = {
          model: this.settings.model,
          messages,
          stream: true,
        };
        if (this.settings.thinkingMode) {
          params.reasoning_effort = this.settings.reasoningEffort;
          (params as any).extra_body = { thinking: { type: "enabled" } };
        } else {
          params.temperature = opts.temperature ?? this.settings.temperature;
        }
        const stream = await this.client.chat.completions.create(params as any);
        const chunks: string[] = [];
        for await (const chunk of stream as any) {
          const delta = chunk.choices?.[0]?.delta?.content;
          if (delta) chunks.push(delta);
        }
        return chunks.join("");
      },
      {
        maxAttempts: 3,
        baseDelayMs: 5000,
        onRetry: (err, attempt, wait) =>
          this.logger.warn(
            `LLM retry #${attempt} after ${wait}ms: ${(err as Error).message}`,
          ),
      },
    );
  }
}
