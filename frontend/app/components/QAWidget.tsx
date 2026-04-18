"use client";

import { useEffect, useState } from "react";

type Source = {
  source_id: number;
  source_file: string;
  chunk_index: number;
  score: number;
  period_label: string | null;
  title: string | null;
  url: string | null;
};

type QueryResponse = {
  answer: string;
  sources: Source[];
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const DOCS_URL = "https://github.com/cedricconol/bspQA#readme";

/** Strip any trailing "Sources:" block the LLM might still emit. */
function stripSourcesBlock(text: string): string {
  return text.replace(/\n{0,2}Sources:\s*[\s\S]*$/i, "").trimEnd();
}

/** Render answer text, turning [N] into styled superscript spans. */
function AnswerText({ text }: { text: string }) {
  const parts = text.split(/(\[\d+\])/g);
  return (
    <p className="text-sm leading-7 text-zinc-800 dark:text-zinc-200 whitespace-pre-wrap">
      {parts.map((part, i) => {
        const match = part.match(/^\[(\d+)\]$/);
        if (match) {
          return (
            <sup key={i} className="text-xs font-semibold text-zinc-500 dark:text-zinc-400">
              [{match[1]}]
            </sup>
          );
        }
        return part;
      })}
    </p>
  );
}

/** Format a source's display label as "Monetary Policy Report <Period>". */
function sourceLabel(source: Source): string {
  if (source.title) return source.title;
  if (source.period_label) return `Monetary Policy Report ${source.period_label}`;
  return source.source_file;
}

export default function QAWidget() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Wake up the Render backend on mount so cold-start happens in the background.
  useEffect(() => {
    fetch(`${API_URL}/health`).catch(() => {});
  }, []);

  async function handleSubmit(e: React.SyntheticEvent) {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(
          (data as { detail?: string }).detail ?? `Request failed (${res.status})`
        );
      }

      const data: QueryResponse = await res.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="w-full max-w-3xl mx-auto px-4 py-12 flex flex-col gap-8">
      {/* Cold-start notice */}
      <div className="rounded-md border border-amber-200 bg-amber-50 dark:border-amber-800 dark:bg-amber-950/40 px-4 py-2.5 text-xs text-amber-700 dark:text-amber-400">
        Your first query each session may take 30–60 seconds while the server wakes up.
      </div>

      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
            BSP Monetary Policy Q&amp;A
          </h1>
          <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
            Ask questions grounded in official BSP monetary policy reports.
          </p>
        </div>
        <a
          href={DOCS_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="shrink-0 mt-1 flex items-center gap-1.5 text-sm font-medium text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100 transition-colors"
        >
          <svg viewBox="0 0 16 16" aria-hidden="true" className="w-4 h-4 fill-current">
            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
          </svg>
          Repo ↗
        </a>
      </div>

      {/* Query form */}
      <form onSubmit={handleSubmit} className="flex flex-col gap-3">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g. What was the BSP's policy rate decision in Q3 2024?"
          rows={3}
          disabled={loading}
          className="w-full rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 px-4 py-3 text-sm text-zinc-900 dark:text-zinc-100 placeholder:text-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-900 dark:focus:ring-zinc-400 resize-none disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="self-end rounded-lg bg-zinc-900 dark:bg-zinc-50 px-5 py-2.5 text-sm font-medium text-white dark:text-zinc-900 hover:bg-zinc-700 dark:hover:bg-zinc-200 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? "Searching…" : "Ask"}
        </button>
      </form>

      {/* Error */}
      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 dark:border-red-900 dark:bg-red-950 px-4 py-3 text-sm text-red-700 dark:text-red-400">
          {error}
        </div>
      )}

      {/* Loading skeleton */}
      {loading && (
        <div className="flex flex-col gap-3 animate-pulse">
          <div className="h-4 bg-zinc-200 dark:bg-zinc-700 rounded w-3/4" />
          <div className="h-4 bg-zinc-200 dark:bg-zinc-700 rounded w-full" />
          <div className="h-4 bg-zinc-200 dark:bg-zinc-700 rounded w-5/6" />
          <div className="h-4 bg-zinc-200 dark:bg-zinc-700 rounded w-2/3" />
        </div>
      )}

      {/* Result */}
      {result && !loading && (
        <div className="flex flex-col gap-6">
          {/* Answer card */}
          <div className="rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 px-5 py-4">
            <p className="text-xs font-medium uppercase tracking-wider text-zinc-400 dark:text-zinc-500 mb-3">
              Answer
            </p>
            <AnswerText text={stripSourcesBlock(result.answer)} />
          </div>

          {/* Sources */}
          {result.sources.length > 0 && (
            <div>
              <p className="text-xs font-medium uppercase tracking-wider text-zinc-400 dark:text-zinc-500 mb-3">
                Sources
              </p>
              <div className="flex flex-col gap-2">
                {result.sources.map((source) => (
                  <div
                    key={source.source_id}
                    className="flex items-center gap-3 rounded-lg border border-zinc-100 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/50 px-4 py-3"
                  >
                    <span className="shrink-0 text-xs font-mono font-semibold text-zinc-400 dark:text-zinc-500">
                      [{source.source_id}]
                    </span>
                    {source.url ? (
                      <a
                        href={source.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-sm font-medium text-blue-600 dark:text-blue-400 underline underline-offset-2 hover:text-blue-800 dark:hover:text-blue-300 truncate"
                      >
                        {sourceLabel(source)}
                      </a>
                    ) : (
                      <span className="text-sm font-medium text-zinc-700 dark:text-zinc-300 truncate">
                        {sourceLabel(source)}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
