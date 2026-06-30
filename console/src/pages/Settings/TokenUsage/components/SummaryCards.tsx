import { Card } from "@agentscope-ai/design";
import { useTranslation } from "react-i18next";
import { formatCompact } from "../../../../utils/formatNumber";
import styles from "../index.module.less";

interface SummaryCardsProps {
  totalCalls: number;
  totalPromptTokens: number;
  totalCompletionTokens: number;
  totalTokens: number;
  totalCacheCreationTokens: number;
  totalCacheReadTokens: number;
}

export function SummaryCards({
  totalCalls,
  totalPromptTokens,
  totalCompletionTokens,
  totalTokens,
  totalCacheCreationTokens,
  totalCacheReadTokens,
}: SummaryCardsProps) {
  const { t } = useTranslation();

  // Cache hit rate = cache_read / (cache_read + prompt_tokens)
  // (cache_creation is excluded — those tokens *built* the cache, they
  //  didn't hit it. prompt_tokens are the uncached portion of input.)
  const cacheDenom = totalCacheReadTokens + totalPromptTokens;
  const cacheHitRate =
    cacheDenom > 0 ? (totalCacheReadTokens / cacheDenom) * 100 : 0;
  // Anthropic charges cache_read at ~10% of normal input price.
  // So each cache_read token saves ~0.9 of one normal token's cost.
  const savedTokens = Math.round(totalCacheReadTokens * 0.9);
  const showCacheStats = totalCacheReadTokens > 0 || totalCacheCreationTokens > 0;

  return (
    <div className={styles.summaryCards}>
      <Card className={styles.card}>
        <div className={styles.cardValue}>{formatCompact(totalCalls)}</div>
        <div className={styles.cardLabel}>{t("tokenUsage.totalCalls")}</div>
      </Card>
      <Card className={styles.card}>
        <div className={styles.cardValue}>
          {formatCompact(totalPromptTokens)}
        </div>
        <div className={styles.cardLabel}>{t("tokenUsage.promptTokens")}</div>
      </Card>
      <Card className={styles.card}>
        <div className={styles.cardValue}>
          {formatCompact(totalCompletionTokens)}
        </div>
        <div className={styles.cardLabel}>
          {t("tokenUsage.completionTokens")}
        </div>
      </Card>
      <Card className={styles.card}>
        <div className={styles.cardValue}>{formatCompact(totalTokens)}</div>
        <div className={styles.cardLabel}>{t("tokenUsage.totalTokens")}</div>
      </Card>
      {showCacheStats && (
        <>
          <Card className={styles.card}>
            <div className={styles.cardValue}>{cacheHitRate.toFixed(1)}%</div>
            <div className={styles.cardLabel}>
              {t("tokenUsage.cacheHitRate")}
            </div>
          </Card>
          <Card className={styles.card}>
            <div className={styles.cardValue}>
              {formatCompact(savedTokens)}
            </div>
            <div className={styles.cardLabel}>
              {t("tokenUsage.savedTokens")}
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
