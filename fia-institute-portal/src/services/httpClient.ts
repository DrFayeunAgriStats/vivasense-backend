export interface ResilientRequestOptions extends RequestInit {
  timeoutMs?: number;
  retries?: number;
  retryDelayMs?: number;
  retryBackoff?: number;
  retryStatuses?: number[];
}

const DEFAULT_TIMEOUT_MS = 30000;
const DEFAULT_RETRY_DELAY_MS = 500;
const DEFAULT_RETRY_BACKOFF = 2;
const RETRYABLE_STATUSES = [408, 425, 429, 500, 502, 503, 504];

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetryableError(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  if (err.name === "AbortError") return true;
  return true;
}

function isIdempotentMethod(method?: string): boolean {
  const normalized = (method ?? "GET").toUpperCase();
  return normalized === "GET" || normalized === "HEAD" || normalized === "OPTIONS";
}

export async function requestWithResilience(
  url: string,
  options: ResilientRequestOptions = {}
): Promise<Response> {
  const {
    timeoutMs = DEFAULT_TIMEOUT_MS,
    retries,
    retryDelayMs = DEFAULT_RETRY_DELAY_MS,
    retryBackoff = DEFAULT_RETRY_BACKOFF,
    retryStatuses = RETRYABLE_STATUSES,
    ...init
  } = options;

  const method = (init.method ?? "GET").toUpperCase();
  const maxRetries = retries ?? (isIdempotentMethod(method) ? 1 : 0);

  let attempt = 0;
  let delayMs = retryDelayMs;
  let lastError: Error | null = null;

  while (attempt <= maxRetries) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(url, {
        ...init,
        method,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok && retryStatuses.includes(response.status) && attempt < maxRetries) {
        await sleep(delayMs);
        delayMs *= retryBackoff;
        attempt += 1;
        continue;
      }

      return response;
    } catch (err) {
      clearTimeout(timeoutId);
      const error = err instanceof Error ? err : new Error(String(err));
      lastError = error;

      if (!isRetryableError(error) || attempt >= maxRetries) {
        if (error.name === "AbortError") {
          throw new Error(`Request timed out after ${timeoutMs}ms`);
        }
        throw error;
      }

      await sleep(delayMs);
      delayMs *= retryBackoff;
      attempt += 1;
    }
  }

  throw lastError ?? new Error("Request failed");
}
