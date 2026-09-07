"use client";

import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";

type BidStatus = "awaiting_approval" | "submitting" | "submitted" | "manual_intervention";

type ReviewSection = {
  section: string;
  requirement: string;
  answer: string;
  source_snippet: string;
};

type BidSnapshot = {
  id: string;
  portal_bid_id: string;
  review_url: string;
  status: BidStatus;
  comparison: ReviewSection[];
  approved: boolean;
  submitted: boolean;
  manual_intervention_required: boolean;
  error: string | null;
};

const POLL_INTERVAL_MS = 5000;
const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "").replace(/\/$/, "");

function apiUrl(path: string) {
  return `${API_BASE_URL}${path}`;
}

function statusCopy(status: BidStatus) {
  switch (status) {
    case "awaiting_approval":
      return { label: "Ready for review", tone: "border-amber-200 bg-amber-50 text-amber-800" };
    case "submitting":
      return { label: "Submitting", tone: "border-blue-200 bg-blue-50 text-blue-800" };
    case "submitted":
      return { label: "Submitted", tone: "border-emerald-200 bg-emerald-50 text-emerald-800" };
    case "manual_intervention":
      return { label: "Manual intervention", tone: "border-red-200 bg-red-50 text-red-800" };
  }
}

function StatusIcon({ status }: { status: BidStatus }) {
  if (status === "submitted") {
    return <span aria-hidden="true" className="text-base">✓</span>;
  }
  if (status === "manual_intervention") {
    return <span aria-hidden="true" className="text-base">!</span>;
  }
  return <span aria-hidden="true" className="text-[10px]">●</span>;
}

function EmptyState({ title, body }: { title: string; body: string }) {
  return (
    <main className="mx-auto flex min-h-screen max-w-6xl items-center justify-center px-6 py-12">
      <section className="w-full max-w-xl rounded-3xl border border-line bg-white p-8 text-center shadow-card">
        <div className="mx-auto mb-5 flex h-12 w-12 items-center justify-center rounded-2xl bg-blue-50 text-xl text-cobalt">↗</div>
        <h1 className="text-2xl font-semibold tracking-tight text-ink">{title}</h1>
        <p className="mt-3 text-sm leading-6 text-slate-500">{body}</p>
      </section>
    </main>
  );
}

function ReviewContent() {
  const searchParams = useSearchParams();
  const bidId = searchParams.get("bid") || process.env.NEXT_PUBLIC_BID_ID || "";
  const [bid, setBid] = useState<BidSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [approving, setApproving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadBid = useCallback(async (background = false) => {
    if (!bidId) return;
    if (background) setRefreshing(true);
    else setLoading(true);
    try {
      const response = await fetch(apiUrl(`/api/bids/${encodeURIComponent(bidId)}`), {
        cache: "no-store",
      });
      if (!response.ok) throw new Error(`Unable to load bid (${response.status}).`);
      const snapshot = (await response.json()) as BidSnapshot;
      setBid(snapshot);
      setError(null);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to load this bid.");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [bidId]);

  useEffect(() => {
    void loadBid();
    if (!bidId) return;
    const interval = window.setInterval(() => void loadBid(true), POLL_INTERVAL_MS);
    return () => window.clearInterval(interval);
  }, [bidId, loadBid]);

  const approveBid = async () => {
    if (!bidId || !bid || bid.status !== "awaiting_approval") return;
    setApproving(true);
    setError(null);
    try {
      const response = await fetch(apiUrl(`/api/bids/${encodeURIComponent(bidId)}/approve`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || `Approval failed (${response.status}).`);
      }
      setBid((await response.json()) as BidSnapshot);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Approval failed.");
    } finally {
      setApproving(false);
    }
  };

  const currentStatus = useMemo(() => bid && statusCopy(bid.status), [bid]);

  if (!bidId) {
    return <EmptyState title="No bid selected" body="Provide a dashboard bid ID with ?bid=<id>, or set NEXT_PUBLIC_BID_ID in the dashboard environment." />;
  }

  if (loading && !bid) {
    return <EmptyState title="Loading bid review" body="Fetching the staged response from the FastAPI backend…" />;
  }

  if (!bid) {
    return <EmptyState title="Could not load this bid" body={error ?? "The backend did not return a review snapshot."} />;
  }

  return (
    <main className="min-h-screen px-4 py-6 sm:px-6 lg:px-10 lg:py-10">
      <div className="mx-auto max-w-6xl">
        <header className="mb-8 flex flex-col gap-5 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <div className="mb-4 flex items-center gap-2 text-xs font-bold uppercase tracking-[0.18em] text-cobalt">
              <span className="h-2 w-2 rounded-full bg-cobalt" aria-hidden="true" /> Anakin bid desk
            </div>
            <h1 className="text-3xl font-semibold tracking-[-0.04em] text-ink sm:text-4xl">Review staged proposal</h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">Compare the RFP requirements with the grounded proposal before releasing the final submission.</p>
          </div>
          <div className="flex items-center gap-3 self-start">
            {refreshing && <span className="text-xs text-slate-400">Updating…</span>}
            {currentStatus && (
              <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-semibold ${currentStatus.tone}`}>
                <StatusIcon status={bid.status} /> {currentStatus.label}
              </span>
            )}
          </div>
        </header>

        {error && (
          <div role="alert" className="mb-6 flex items-start justify-between gap-4 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
            <span>{error}</span>
            <button className="font-semibold underline underline-offset-4" onClick={() => void loadBid()}>Retry</button>
          </div>
        )}

        <section className="mb-7 grid gap-4 sm:grid-cols-3">
          <div className="rounded-2xl border border-line bg-white px-5 py-4 shadow-card">
            <p className="text-[11px] font-bold uppercase tracking-[0.16em] text-slate-400">Review sections</p>
            <p className="mt-2 text-2xl font-semibold text-ink">{bid.comparison.length}</p>
          </div>
          <div className="rounded-2xl border border-line bg-white px-5 py-4 shadow-card">
            <p className="text-[11px] font-bold uppercase tracking-[0.16em] text-slate-400">Bid reference</p>
            <p className="mt-2 truncate font-mono text-sm text-ink" title={bid.id}>{bid.id}</p>
          </div>
          <div className="rounded-2xl border border-line bg-white px-5 py-4 shadow-card">
            <p className="text-[11px] font-bold uppercase tracking-[0.16em] text-slate-400">Approval gate</p>
            <p className="mt-2 text-sm font-semibold text-ink">{bid.approved ? "Released" : "Waiting for you"}</p>
          </div>
        </section>

        <div className="space-y-5">
          {bid.comparison.map((item) => (
            <article key={item.section} className="overflow-hidden rounded-3xl border border-line bg-white shadow-card">
              <div className="border-b border-line px-5 py-4 sm:px-7">
                <p className="text-xs font-bold uppercase tracking-[0.16em] text-cobalt">{item.section}</p>
              </div>
              <div className="grid lg:grid-cols-2">
                <div className="border-b border-line p-5 sm:p-7 lg:border-b-0 lg:border-r">
                  <p className="mb-3 text-[11px] font-bold uppercase tracking-[0.16em] text-slate-400">RFP requirement</p>
                  <p className="whitespace-pre-wrap text-sm leading-7 text-slate-700">{item.requirement}</p>
                </div>
                <div className="bg-[#fbfcff] p-5 sm:p-7">
                  <p className="mb-3 text-[11px] font-bold uppercase tracking-[0.16em] text-cobalt">Proposal answer</p>
                  <p className="whitespace-pre-wrap text-sm leading-7 text-ink">{item.answer}</p>
                  {item.source_snippet && (
                    <div className="mt-6 rounded-2xl border border-blue-100 bg-blue-50/60 p-4">
                      <p className="mb-2 text-[10px] font-bold uppercase tracking-[0.16em] text-blue-700">Grounding source</p>
                      <blockquote className="border-l-2 border-blue-300 pl-3 text-sm italic leading-6 text-blue-950">“{item.source_snippet}”</blockquote>
                    </div>
                  )}
                </div>
              </div>
            </article>
          ))}
        </div>

        <section className="mt-7 rounded-3xl bg-ink p-5 text-white shadow-card sm:p-7">
          <div className="flex flex-col gap-5 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-lg font-semibold">Ready to release this bid?</p>
              <p className="mt-1 text-sm leading-6 text-slate-300">
                {bid.status === "awaiting_approval" ? "Your approval unlocks the staged submission." : currentStatus?.label}
              </p>
              {bid.error && <p className="mt-2 text-sm text-red-300">{bid.error}</p>}
            </div>
            <button
              type="button"
              onClick={() => void approveBid()}
              disabled={approving || bid.status !== "awaiting_approval"}
              className="inline-flex min-h-12 items-center justify-center gap-2 rounded-xl bg-white px-5 py-3 text-sm font-bold text-ink transition hover:bg-blue-50 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {approving ? "Approving…" : bid.status === "awaiting_approval" ? "Approve & Submit" : currentStatus?.label}
              {bid.status === "awaiting_approval" && <span aria-hidden="true">→</span>}
            </button>
          </div>
        </section>

        <footer className="flex flex-col gap-2 py-6 text-xs text-slate-400 sm:flex-row sm:items-center sm:justify-between">
          <span>Review data is refreshed automatically while this page is open.</span>
          <a className="underline underline-offset-4 hover:text-slate-600" href={bid.review_url} target="_blank" rel="noreferrer">Open staged portal review ↗</a>
        </footer>
      </div>
    </main>
  );
}

export default function ReviewPage() {
  return (
    <Suspense fallback={<EmptyState title="Loading bid review" body="Preparing the review workspace…" />}>
      <ReviewContent />
    </Suspense>
  );
}
