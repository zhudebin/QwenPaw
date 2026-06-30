/** Single token usage record (per date + provider + model). */
export interface TokenUsageRecord {
  date: string; // YYYY-MM-DD
  provider_id: string;
  model: string;
  prompt_tokens: number;
  completion_tokens: number;
  call_count: number;
  /** Tokens written into Anthropic prompt cache (paid 1.25x). */
  cache_creation_tokens?: number;
  /** Tokens read from Anthropic prompt cache (paid 0.1x — i.e. saved). */
  cache_read_tokens?: number;
}

/** Per-model (has provider_id, model) or per-date (no provider_id, model) stats. */
export interface TokenUsageStats {
  provider_id?: string;
  model?: string;
  prompt_tokens: number;
  completion_tokens: number;
  call_count: number;
  cache_creation_tokens?: number;
  cache_read_tokens?: number;
}

export interface TokenUsageSummary {
  total_prompt_tokens: number;
  total_completion_tokens: number;
  total_calls: number;
  total_cache_creation_tokens?: number;
  total_cache_read_tokens?: number;
  by_model: Record<string, TokenUsageStats>;
  by_date: Record<string, TokenUsageStats>;
}
