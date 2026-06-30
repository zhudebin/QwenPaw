import { useMemo } from "react";
import type { TokenUsageRecord } from "../../../../api/types/tokenUsage";

interface ModelStats {
  model: string;
  provider_id: string;
  prompt_tokens: number;
  completion_tokens: number;
  call_count: number;
  cache_creation_tokens: number;
  cache_read_tokens: number;
}

interface DateStats {
  prompt_tokens: number;
  completion_tokens: number;
  call_count: number;
  cache_creation_tokens: number;
  cache_read_tokens: number;
}

interface AggregatedData {
  total_prompt_tokens: number;
  total_completion_tokens: number;
  total_calls: number;
  total_cache_creation_tokens: number;
  total_cache_read_tokens: number;
  by_model: Record<string, ModelStats>;
  by_date: Record<string, DateStats>;
  by_date_model: Record<string, Record<string, ModelStats>>;
}

export function useDataAggregation(records: TokenUsageRecord[]) {
  return useMemo<AggregatedData | null>(() => {
    if (records.length === 0) return null;

    const byModel: AggregatedData["by_model"] = {};
    const byDate: AggregatedData["by_date"] = {};
    const byDateModel: AggregatedData["by_date_model"] = {};

    let totalPrompt = 0;
    let totalCompletion = 0;
    let totalCalls = 0;
    let totalCacheCreation = 0;
    let totalCacheRead = 0;

    records.forEach((r) => {
      const pt = r.prompt_tokens;
      const ct = r.completion_tokens;
      const calls = r.call_count;
      const cc = r.cache_creation_tokens ?? 0;
      const cr = r.cache_read_tokens ?? 0;
      const providerId = r.provider_id;
      totalPrompt += pt;
      totalCompletion += ct;
      totalCalls += calls;
      totalCacheCreation += cc;
      totalCacheRead += cr;

      const modelKey = `${providerId}:${r.model}`;
      if (!byModel[modelKey]) {
        byModel[modelKey] = {
          model: r.model,
          provider_id: providerId,
          prompt_tokens: 0,
          completion_tokens: 0,
          call_count: 0,
          cache_creation_tokens: 0,
          cache_read_tokens: 0,
        };
      }
      byModel[modelKey].prompt_tokens += pt;
      byModel[modelKey].completion_tokens += ct;
      byModel[modelKey].call_count += calls;
      byModel[modelKey].cache_creation_tokens += cc;
      byModel[modelKey].cache_read_tokens += cr;

      if (!byDate[r.date]) {
        byDate[r.date] = {
          prompt_tokens: 0,
          completion_tokens: 0,
          call_count: 0,
          cache_creation_tokens: 0,
          cache_read_tokens: 0,
        };
      }
      byDate[r.date].prompt_tokens += pt;
      byDate[r.date].completion_tokens += ct;
      byDate[r.date].call_count += calls;
      byDate[r.date].cache_creation_tokens += cc;
      byDate[r.date].cache_read_tokens += cr;

      if (!byDateModel[r.date]) {
        byDateModel[r.date] = {};
      }
      if (!byDateModel[r.date][modelKey]) {
        byDateModel[r.date][modelKey] = {
          model: r.model,
          provider_id: providerId,
          prompt_tokens: 0,
          completion_tokens: 0,
          call_count: 0,
          cache_creation_tokens: 0,
          cache_read_tokens: 0,
        };
      }
      byDateModel[r.date][modelKey].prompt_tokens += pt;
      byDateModel[r.date][modelKey].completion_tokens += ct;
      byDateModel[r.date][modelKey].call_count += calls;
      byDateModel[r.date][modelKey].cache_creation_tokens += cc;
      byDateModel[r.date][modelKey].cache_read_tokens += cr;
    });

    return {
      total_prompt_tokens: totalPrompt,
      total_completion_tokens: totalCompletion,
      total_calls: totalCalls,
      total_cache_creation_tokens: totalCacheCreation,
      total_cache_read_tokens: totalCacheRead,
      by_model: byModel,
      by_date: byDate,
      by_date_model: byDateModel,
    };
  }, [records]);
}
