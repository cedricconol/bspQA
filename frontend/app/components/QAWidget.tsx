"use client";

import { useState } from "react";

type Source = {
  source_id: number;
  source_file: string;
  chunk_index: number;
  score: number;
};

type QueryResponse = {
  answer: string;
  sources: Source[];
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function QAWidget() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
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
      {/* Header */}
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
          BSP Monetary Policy Q&amp;A
        </h1>
        <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
          Ask questions grounded in official BSP monetary policy and inflation reports.
        </p>
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
            <p className="text-sm leading-7 text-zinc-800 dark:text-zinc-200 whitespace-pre-wrap">
              {result.answer}
            </p>
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
                    className="flex items-start gap-3 rounded-lg border border-zinc-100 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/50 px-4 py-3"
                  >
                    <span className="mt-0.5 shrink-0 text-xs font-mono font-semibold text-zinc-400 dark:text-zinc-500">
                      [{source.source_id}]
                    </span>
                    <div className="flex flex-col gap-0.5 min-w-0">
                      <span className="text-sm font-medium text-zinc-700 dark:text-zinc-300 truncate">
                        {source.source_file}
                      </span>
                      <span className="text-xs text-zinc-400 dark:text-zinc-500">
                        chunk {source.chunk_index} · score {source.score.toFixed(3)}
                      </span>
                    </div>
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
