"use client";

import { useRouter, useSearchParams } from "next/navigation";
import {
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

type BidStatus =
  | "awaiting_approval"
  | "submitting"
  | "submitted"
  | "manual_intervention";

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

type GenerationStage = "crawling" | "drafting" | "staging";
type EnvelopeId = "technical" | "knowledge" | "security" | "commercial";

type ProcurementEnvelope = {
  id: EnvelopeId;
  number: string;
  title: string;
  description: string;
};

const AGENCY_NAME = "Global Enterprise Solutions Directorate";
const SOLICITATION_REFERENCE = "#2026-ENT-092";
const POLL_INTERVAL_MS = 5000;
const REQUEST_TIMEOUT_MS = 12000;
const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8001"
).replace(/\/$/, "");

const GENERATION_STAGES: { key: GenerationStage; label: string }[] = [
  { key: "crawling", label: "RFP intake" },
  { key: "drafting", label: "Response drafting" },
  { key: "staging", label: "Portal staging" },
];

const PROCESS_STEPS = [
  "RFP received",
  "Responses prepared",
  "Bid staged",
  "Human authorization",
  "Submitted",
] as const;

const PROCUREMENT_ENVELOPES: ProcurementEnvelope[] = [
  {
    id: "technical",
    number: "1.0",
    title: "Technical Environment & APIs",
    description: "Integration architecture, service interfaces, and platform operation.",
  },
  {
    id: "knowledge",
    number: "2.0",
    title: "Knowledge Ingestion & Traceability",
    description: "Content ingestion, evidence provenance, and response traceability.",
  },
  {
    id: "security",
    number: "3.0",
    title: "Security & Compliance",
    description: "Identity, data protection, auditability, and governance controls.",
  },
  {
    id: "commercial",
    number: "4.0",
    title: "Commercials & Support",
    description: "Pricing, implementation terms, service support, and maintenance.",
  },
];

function apiUrl(path: string) {
  return `${API_BASE_URL}${path}`;
}

function requestErrorMessage(error: unknown, action: string) {
  if (error instanceof TypeError) {
    return `Backend unavailable at ${API_BASE_URL}. Start the FastAPI service and try again.`;
  }
  if (error instanceof DOMException && error.name === "TimeoutError") {
    return `${action} timed out. Confirm the local services are available and try again.`;
  }
  return error instanceof Error
    ? error.message
    : `${action} failed. Please try again.`;
}

function Icon({
  name,
  size = 18,
}: {
  name: "arrow" | "check" | "chevron" | "external" | "lock" | "warning";
  size?: number;
}) {
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.8,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
  };
  if (name === "check") {
    return <svg {...common}><path d="m5 12 4.2 4.2L19 6.8" /></svg>;
  }
  if (name === "arrow") {
    return <svg {...common}><path d="M5 12h13" /><path d="m13 6 6 6-6 6" /></svg>;
  }
  if (name === "chevron") {
    return <svg {...common}><path d="m7 10 5 5 5-5" /></svg>;
  }
  if (name === "external") {
    return (
      <svg {...common}>
        <path d="M14 5h5v5" /><path d="M19 5 11 13" />
        <path d="M18 13v5a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5" />
      </svg>
    );
  }
  if (name === "lock") {
    return <svg {...common}><rect x="5" y="10" width="14" height="10" rx="1" /><path d="M8 10V7a4 4 0 0 1 8 0v3" /></svg>;
  }
  return (
    <svg {...common}>
      <path d="M10.3 4.2 2.1 18a2 2 0 0 0 1.7 3h16.4a2 2 0 0 0 1.7-3L13.7 4.2a2 2 0 0 0-3.4 0Z" />
      <path d="M12 9v4" /><path d="M12 17h.01" />
    </svg>
  );
}

function statusCopy(status: BidStatus) {
  switch (status) {
    case "awaiting_approval":
      return {
        label: "Awaiting authorized sign-off",
        short: "PENDING APPROVAL",
        detail: "The response is staged. Final submission remains locked.",
      };
    case "submitting":
      return {
        label: "Authorized for submission",
        short: "SUBMITTING",
        detail: "Authorization is recorded and the mock portal submission is in progress.",
      };
    case "submitted":
      return {
        label: "Submission confirmed",
        short: "SUBMITTED",
        detail: "The mock procurement portal confirmed receipt of the response.",
      };
    case "manual_intervention":
      return {
        label: "Manual intervention required",
        short: "ACTION REQUIRED",
        detail: "Automated processing stopped. An authorized reviewer must inspect the record.",
      };
  }
}

function statusStyles(status: BidStatus) {
  if (status === "submitted") return "border-[#2f6b46] bg-[#e8f4ec] text-[#234f35]";
  if (status === "manual_intervention") return "border-[#a33b3b] bg-[#fff0f0] text-[#842e2e]";
  if (status === "submitting") return "border-[#315f8c] bg-[#eaf2fa] text-[#214c75]";
  return "border-[#9b7419] bg-[#fff8df] text-[#6f520d]";
}

function isCapabilityMissing(answer: string) {
  return answer.trim().toUpperCase().startsWith("CAPABILITY_NOT_FOUND");
}

function envelopeFor(item: ReviewSection): EnvelopeId {
  const text = `${item.section} ${item.requirement}`.toLowerCase();
  if (/pricing|commercial|fee|subscription|tax|support|maintenance|service level/.test(text)) return "commercial";
  if (/security|compliance|saml|oidc|encryption|audit|access control|governance|data location|backup/.test(text)) return "security";
  if (/knowledge|ingestion|document|source|traceability|citation|evidence|response quality/.test(text)) return "knowledge";
  return "technical";
}

function GenerationProgress({ stage }: { stage: GenerationStage }) {
  const activeIndex = GENERATION_STAGES.findIndex((item) => item.key === stage);
  return (
    <section className="mb-5 border border-[#9aa8b5] bg-white px-5 py-4 shadow-sm" aria-live="polite" aria-label="Bid generation progress">
      <div className="mb-3 flex items-center justify-between gap-4">
        <div>
          <p className="text-xs font-bold uppercase tracking-[0.12em] text-[#173f68]">Preparing response record</p>
          <p className="mt-1 text-sm text-slate-600">The procurement response is being prepared for authorized review.</p>
        </div>
        <span className="h-3 w-3 animate-pulse rounded-full bg-[#1f5b8f]" />
      </div>
      <ol className="grid border border-slate-300 sm:grid-cols-3">
        {GENERATION_STAGES.map((item, index) => (
          <li key={item.key} className={`flex items-center gap-3 border-slate-300 px-3 py-2.5 text-sm sm:border-r sm:last:border-r-0 ${index <= activeIndex ? "bg-[#edf4fa] text-[#173f68]" : "bg-slate-50 text-slate-500"}`}>
            <span className="grid h-6 w-6 shrink-0 place-items-center border border-current text-xs font-bold">{index < activeIndex ? "✓" : index + 1}</span>
            {item.label}
          </li>
        ))}
      </ol>
    </section>
  );
}

function EmptyState({ title, body, warning = false, onRetry, action }: { title: string; body: string; warning?: boolean; onRetry?: () => void; action?: ReactNode }) {
  return (
    <main className="grid min-h-screen place-items-center bg-[#e8edf2] px-5 py-12 text-slate-900">
      <section className="w-full max-w-xl border border-[#8c9aa7] bg-white shadow-[0_12px_32px_rgba(15,35,55,0.12)]">
        <div className="border-b-4 border-[#b58a2a] bg-[#123a60] px-6 py-5 text-white">
          <p className="text-xs font-bold uppercase tracking-[0.14em] text-blue-100">{AGENCY_NAME}</p>
          <h1 className="mt-2 font-serif text-2xl">Supplier Response Portal</h1>
        </div>
        <div className="p-6 sm:p-8">
          <div className={`mb-5 flex h-11 w-11 items-center justify-center border ${warning ? "border-red-700 bg-red-50 text-red-800" : "border-[#315f8c] bg-[#edf4fa] text-[#173f68]"}`}>
            <Icon name={warning ? "warning" : "lock"} size={20} />
          </div>
          <h2 className="font-serif text-2xl font-bold text-[#173f68]">{title}</h2>
          <p className="mt-3 text-base leading-7 text-slate-600">{body}</p>
          <div className="mt-6 flex flex-wrap gap-3">
            {onRetry && <button type="button" onClick={onRetry} className="border border-[#173f68] bg-[#173f68] px-4 py-2.5 text-sm font-bold text-white hover:bg-[#0d2d4c]">Retry connection</button>}
            {action}
          </div>
        </div>
      </section>
    </main>
  );
}

function GenerateNewBidButton({ disabled, onClick }: { disabled: boolean; onClick: () => void }) {
  return (
    <button type="button" onClick={onClick} disabled={disabled} className="inline-flex min-h-11 items-center justify-center gap-2 border border-[#b9c4ce] bg-white px-4 py-2 text-sm font-bold text-[#173f68] shadow-sm hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60">
      {disabled ? <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-[#173f68]" /> : <span className="text-base" aria-hidden="true">＋</span>}
      {disabled ? "Preparing new response…" : "Start new response"}
    </button>
  );
}

function ProcessStrip({ status }: { status: BidStatus }) {
  const activeIndex = status === "submitted" ? 4 : status === "submitting" ? 3 : status === "awaiting_approval" ? 3 : 2;
  return (
    <nav className="border-b border-slate-300 bg-slate-50" aria-label="Submission workflow">
      <ol className="grid sm:grid-cols-5">
        {PROCESS_STEPS.map((step, index) => {
          const complete = index < activeIndex || status === "submitted";
          const current = index === activeIndex && status !== "submitted";
          return (
            <li key={step} className={`flex min-h-12 items-center gap-2 border-b border-slate-200 px-3 py-2 text-xs font-bold sm:border-b-0 sm:border-r sm:last:border-r-0 ${current ? "bg-[#fff8df] text-[#6f520d]" : complete ? "text-[#285c3d]" : "text-slate-500"}`}>
              <span className={`grid h-5 w-5 shrink-0 place-items-center border text-[11px] ${complete ? "border-[#3d7753] bg-[#e8f4ec]" : current ? "border-[#9b7419] bg-white" : "border-slate-300 bg-white"}`}>{complete ? "✓" : index + 1}</span>
              {step}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

function RequirementField({ item, fieldIndex, reference, editedAnswer, editable, onAnswerChange, onReset }: { item: ReviewSection; fieldIndex: number; reference: string; editedAnswer: string; editable: boolean; onAnswerChange: (value: string) => void; onReset: () => void }) {
  const [citationOpen, setCitationOpen] = useState(false);
  const unmet = isCapabilityMissing(item.answer);
  const grounded = !unmet && Boolean(item.source_snippet.trim());
  const hasEdit = editedAnswer !== item.answer;
  const fieldId = `response-field-${fieldIndex}`;
  const citationId = `citation-${fieldIndex}`;

  return (
    <article className="border-t border-slate-300 first:border-t-0">
      <div className="grid lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
        <div className="border-b border-slate-300 bg-[#f7f8fa] p-5 lg:border-b-0 lg:border-r lg:p-6">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <span className="font-mono text-xs font-bold uppercase tracking-[0.08em] text-slate-500">Requirement {reference}</span>
            <span className="border border-slate-300 bg-white px-2 py-1 text-xs font-bold text-slate-600">{item.section}</span>
          </div>
          <h3 className="text-sm font-bold uppercase tracking-[0.04em] text-[#173f68]">Issuing authority requirement</h3>
          <p className="mt-3 text-base leading-7 text-slate-800">{item.requirement}</p>
        </div>

        <div className="p-5 lg:p-6">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
            <label htmlFor={fieldId} className="text-sm font-bold text-[#173f68]">Bidder response <span className="text-red-700" aria-hidden="true">*</span></label>
            {grounded ? (
              <button type="button" aria-expanded={citationOpen} aria-controls={citationId} onClick={() => setCitationOpen((open) => !open)} className="inline-flex items-center gap-2 border border-[#2f6b46] bg-[#e8f4ec] px-2.5 py-1.5 text-xs font-extrabold text-[#234f35] hover:bg-[#d9ecdf]" title="Show supporting capability citation">
                <Icon name="check" size={14} /> [VERIFIED / GROUNDED]
                <span className={citationOpen ? "rotate-180" : ""}><Icon name="chevron" size={13} /></span>
              </button>
            ) : unmet ? (
              <span className="inline-flex items-center gap-2 border border-[#a33b3b] bg-[#fff0f0] px-2.5 py-1.5 text-xs font-extrabold text-[#842e2e]"><Icon name="warning" size={14} /> [CAPABILITY_NOT_FOUND — UNMET]</span>
            ) : (
              <span className="inline-flex items-center gap-2 border border-[#9b7419] bg-[#fff8df] px-2.5 py-1.5 text-xs font-extrabold text-[#6f520d]"><Icon name="warning" size={14} /> [SOURCE REVIEW REQUIRED]</span>
            )}
          </div>

          <textarea
            id={fieldId}
            name={`response_${fieldIndex + 1}`}
            value={editedAnswer}
            onChange={(event) => onAnswerChange(event.target.value)}
            readOnly={!editable}
            rows={6}
            aria-describedby={`${fieldId}-help`}
            className={`w-full resize-y border p-3.5 text-base leading-7 text-slate-900 shadow-inner outline-none transition focus:border-[#1f5b8f] focus:ring-2 focus:ring-[#1f5b8f]/20 ${hasEdit ? "border-[#9b7419] bg-[#fffdf4]" : "border-slate-400 bg-white"} ${!editable ? "cursor-default bg-slate-50 text-slate-700" : ""}`}
          />
          <div id={`${fieldId}-help`} className="mt-2 flex flex-wrap items-start justify-between gap-2 text-xs text-slate-500">
            <span>{editable ? "Editable working field. The staged response must match before authorization." : "This response field is locked because the bid is no longer awaiting approval."}</span>
            {hasEdit && <button type="button" onClick={onReset} className="font-bold text-[#173f68] underline underline-offset-2 hover:text-[#0d2d4c]">Restore staged answer</button>}
          </div>

          {citationOpen && grounded && (
            <aside id={citationId} className="mt-4 border-l-4 border-[#3d7753] bg-[#f0f7f2] px-4 py-3" aria-label="Supporting capability citation">
              <p className="text-xs font-bold uppercase tracking-[0.08em] text-[#285c3d]">Internal capability record</p>
              <blockquote className="mt-2 text-sm leading-6 text-slate-700">“{item.source_snippet}”</blockquote>
            </aside>
          )}
        </div>
      </div>
    </article>
  );
}

function ReviewContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const urlBidId = searchParams.get("bid") || process.env.NEXT_PUBLIC_BID_ID || "";
  const [selectedBidId, setSelectedBidId] = useState(urlBidId);
  const bidId = selectedBidId;
  const [bid, setBid] = useState<BidSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [approving, setApproving] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [generationStage, setGenerationStage] = useState<GenerationStage>("crawling");
  const [approvalReleased, setApprovalReleased] = useState(false);
  const [editedAnswers, setEditedAnswers] = useState<Record<string, string>>({});
  const [authorizedRepresentative, setAuthorizedRepresentative] = useState(false);
  const [ungroundedAcknowledged, setUngroundedAcknowledged] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const initializedBidRef = useRef<string | null>(null);

  useEffect(() => setSelectedBidId(urlBidId), [urlBidId]);

  useEffect(() => {
    if (!generating) return;
    setGenerationStage("crawling");
    const draftingTimer = window.setTimeout(() => setGenerationStage("drafting"), 900);
    const stagingTimer = window.setTimeout(() => setGenerationStage("staging"), 2100);
    return () => {
      window.clearTimeout(draftingTimer);
      window.clearTimeout(stagingTimer);
    };
  }, [generating]);

  useEffect(() => {
    if (!bid || initializedBidRef.current === bid.id) return;
    initializedBidRef.current = bid.id;
    const released = bid.approved || bid.status === "submitting" || bid.status === "submitted";
    setEditedAnswers(Object.fromEntries(bid.comparison.map((item) => [item.section, item.answer])));
    setApprovalReleased(released);
    setAuthorizedRepresentative(released);
    setUngroundedAcknowledged(released);
  }, [bid]);

  const loadBid = useCallback(async (background = false) => {
    if (!bidId) {
      setLoading(false);
      return;
    }
    if (background) setRefreshing(true);
    else setLoading(true);
    try {
      const response = await fetch(apiUrl(`/api/bids/${encodeURIComponent(bidId)}`), {
        cache: "no-store",
        signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      });
      if (!response.ok) throw new Error(`Unable to load bid (${response.status}).`);
      const snapshot = (await response.json()) as BidSnapshot;
      setBid(snapshot);
      if (snapshot.approved || snapshot.status === "submitting" || snapshot.status === "submitted") setApprovalReleased(true);
      setError(null);
    } catch (requestError) {
      setError(requestErrorMessage(requestError, "Loading the bid"));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [bidId]);

  const generateBid = async () => {
    if (generating) return;
    setGenerating(true);
    setError(null);
    try {
      const response = await fetch(apiUrl("/api/bids/generate"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
        signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      });
      if (!response.ok) {
        let message = `Bid generation failed (${response.status}).`;
        try {
          const payload = (await response.json()) as { detail?: { message?: string } | string };
          if (typeof payload.detail === "string") message = payload.detail;
          else if (payload.detail?.message) message = payload.detail.message;
        } catch {
          // Preserve the stable fallback when the backend returns a non-JSON error.
        }
        throw new Error(message);
      }
      const snapshot = (await response.json()) as BidSnapshot;
      setSelectedBidId(snapshot.id);
      setBid(snapshot);
      setEditedAnswers(Object.fromEntries(snapshot.comparison.map((item) => [item.section, item.answer])));
      setApprovalReleased(false);
      setAuthorizedRepresentative(false);
      setUngroundedAcknowledged(false);
      router.replace(`/?bid=${encodeURIComponent(snapshot.id)}`, { scroll: false });
    } catch (requestError) {
      setError(requestErrorMessage(requestError, "Bid generation"));
    } finally {
      setGenerating(false);
    }
  };

  useEffect(() => {
    void loadBid();
    if (!bidId) return;
    const interval = window.setInterval(() => void loadBid(true), POLL_INTERVAL_MS);
    return () => window.clearInterval(interval);
  }, [bidId, loadBid]);

  const approveBid = async () => {
    if (!bidId || !bid || bid.status !== "awaiting_approval" || !authorizedRepresentative || !ungroundedAcknowledged) return;
    setApproving(true);
    setError(null);
    try {
      const response = await fetch(apiUrl(`/api/bids/${encodeURIComponent(bidId)}/approve`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          solicitation_reference: SOLICITATION_REFERENCE,
          session_id: bid.id,
          attestations: {
            authorized_representative: true,
            ungrounded_items_acknowledged: true,
          },
        }),
        signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || `Approval failed (${response.status}).`);
      }
      setBid((await response.json()) as BidSnapshot);
      setApprovalReleased(true);
    } catch (requestError) {
      setError(requestErrorMessage(requestError, "Approval"));
    } finally {
      setApproving(false);
    }
  };

  // Polling and UI rendering may proceed concurrently; button guards prevent duplicate
  // requests, while the backend asyncio.Event blocks final submission until approval.

  const groupedEnvelopes = useMemo(() => PROCUREMENT_ENVELOPES.map((envelope) => ({
    ...envelope,
    requirements: (bid?.comparison ?? [])
      .map((item, sourceIndex) => ({ item, sourceIndex }))
      .filter(({ item }) => envelopeFor(item) === envelope.id),
  })), [bid]);

  const isApproved = Boolean(approvalReleased || bid?.approved || bid?.status === "submitting" || bid?.status === "submitted");
  const hasEdits = Boolean(bid?.comparison.some((item) => editedAnswers[item.section] !== undefined && editedAnswers[item.section] !== item.answer));
  const canEdit = Boolean(bid && bid.status === "awaiting_approval" && !isApproved);
  const attestationsComplete = authorizedRepresentative && ungroundedAcknowledged;

  if (!bidId) {
    return <EmptyState title="No response record selected" body={error ?? "Start a new response to create a grounded procurement review session."} warning={Boolean(error)} action={<GenerateNewBidButton disabled={generating} onClick={() => void generateBid()} />} />;
  }
  if (loading && !bid) return <EmptyState title="Loading response record" body="Retrieving the staged submission from the local procurement workflow." />;
  if (!bid) return <EmptyState title="Response record unavailable" body={error ?? "The backend did not return a review snapshot."} warning onRetry={() => void loadBid()} />;

  const currentStatus = statusCopy(bid.status);
  const groundedCount = bid.comparison.filter((item) => !isCapabilityMissing(item.answer) && Boolean(item.source_snippet.trim())).length;
  const exceptionCount = bid.comparison.length - groundedCount;
  const complianceVerified = exceptionCount === 0;
  const sessionHash = bid.id.slice(0, 16).toUpperCase();
  const submitDisabled = approving || bid.status !== "awaiting_approval" || hasEdits || !attestationsComplete;

  let actionLabel = "Approve & Submit Bid";
  if (approving) actionLabel = "Recording authorization…";
  else if (isApproved) actionLabel = "Submission authorized";
  else if (hasEdits) actionLabel = "Restore staged responses to continue";
  else if (!attestationsComplete) actionLabel = "Complete certifications to continue";

  return (
    <main className="min-h-screen bg-[#e8edf2] pb-12 text-slate-900">
      <header className="border-b-4 border-[#b58a2a] bg-[#123a60] text-white shadow-md">
        <div className="mx-auto flex max-w-6xl flex-col gap-5 px-4 py-5 sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:px-8">
          <div className="flex items-center gap-4">
            <div className="grid h-14 w-14 shrink-0 place-items-center border-2 border-white/80 font-serif text-base font-bold tracking-[0.08em]">GESD</div>
            <div>
              <p className="text-xs font-bold uppercase tracking-[0.14em] text-blue-100">Official e-Procurement System</p>
              <h1 className="mt-1 font-serif text-xl font-bold sm:text-2xl">{AGENCY_NAME}</h1>
              <p className="mt-1 text-sm text-blue-100">Supplier response workspace</p>
            </div>
          </div>
          <GenerateNewBidButton disabled={generating} onClick={() => void generateBid()} />
        </div>
      </header>

      <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6 lg:px-8">
        {error && (
          <div role="alert" className="mb-5 flex items-start justify-between gap-4 border border-red-700 bg-red-50 px-4 py-3 text-sm text-red-900">
            <span className="flex items-start gap-2"><Icon name="warning" size={17} /> {error}</span>
            <button type="button" className="shrink-0 font-bold underline underline-offset-2" onClick={() => void loadBid()}>Retry</button>
          </div>
        )}

        {generating && <GenerationProgress stage={generationStage} />}

        <div className="border border-[#8292a0] bg-white shadow-[0_14px_35px_rgba(21,45,69,0.14)]">
          <section className="border-b border-slate-300 px-5 py-5 sm:px-7" aria-labelledby="form-title">
            <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
              <div>
                <p className="text-xs font-extrabold uppercase tracking-[0.12em] text-[#6f520d]">Solicitation response form</p>
                <h2 id="form-title" className="mt-2 font-serif text-3xl font-bold text-[#173f68]">Enterprise Knowledge Management & Response Automation Platform</h2>
                <p className="mt-2 text-base text-slate-600">Review and certify the staged bidder response before electronic submission.</p>
              </div>
              <div className={`shrink-0 border px-4 py-3 ${statusStyles(bid.status)}`}>
                <p className="text-xs font-extrabold uppercase tracking-[0.1em]">{currentStatus.short}</p>
                <p className="mt-1 text-sm font-semibold">{currentStatus.label}</p>
              </div>
            </div>

            <dl className="mt-6 grid border border-slate-300 sm:grid-cols-2 lg:grid-cols-4">
              {[
                ["Solicitation reference", SOLICITATION_REFERENCE],
                ["Response record", bid.portal_bid_id],
                ["Session ID hash", sessionHash],
                ["Record synchronization", refreshing ? "Checking for updates…" : `Active · ${POLL_INTERVAL_MS / 1000}s polling`],
              ].map(([label, value]) => (
                <div key={label} className="border-b border-slate-300 p-3.5 last:border-b-0 sm:border-r sm:[&:nth-child(2)]:border-r-0 lg:border-b-0 lg:[&:nth-child(2)]:border-r lg:last:border-r-0">
                  <dt className="text-xs font-bold uppercase tracking-[0.06em] text-slate-500">{label}</dt>
                  <dd className="mt-1 break-all font-mono text-sm font-bold text-slate-800">{value}</dd>
                </div>
              ))}
            </dl>
          </section>

          <div className={`flex items-start gap-3 border-b px-5 py-4 sm:px-7 ${complianceVerified ? "border-[#78a188] bg-[#e8f4ec] text-[#234f35]" : "border-[#c66a6a] bg-[#fff0f0] text-[#842e2e]"}`} role="status">
            <span className="mt-0.5"><Icon name={complianceVerified ? "check" : "warning"} size={19} /></span>
            <div>
              <p className="text-xs font-extrabold uppercase tracking-[0.1em]">Compliance verification status</p>
              <p className="mt-1 text-sm font-semibold">{complianceVerified ? "All response items are linked to supporting capability records." : `${exceptionCount} response item${exceptionCount === 1 ? " requires" : "s require"} exception review before authorization.`}</p>
            </div>
          </div>

          <ProcessStrip status={bid.status} />

          <div className="bg-[#f4f6f8] px-4 py-5 sm:px-6 sm:py-7">
            <div className="mb-5 border-l-4 border-[#173f68] bg-white px-4 py-3 text-sm leading-6 text-slate-600">Fields marked with an asterisk are mandatory. Green verification tags open the source citation used to ground the staged answer. Unmet items remain isolated for human review.</div>

            <div className="space-y-6">
              {groupedEnvelopes.map((envelope) => (
                <section key={envelope.id} className="border border-[#8d9aa6] bg-white shadow-sm" aria-labelledby={`envelope-${envelope.id}`}>
                  <div className="border-b-2 border-[#173f68] bg-[#eaf0f5] px-5 py-4">
                    <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
                      <div>
                        <p className="text-xs font-extrabold uppercase tracking-[0.12em] text-[#6f520d]">Response envelope {envelope.number}</p>
                        <h2 id={`envelope-${envelope.id}`} className="mt-1 font-serif text-xl font-bold text-[#173f68]">{envelope.number} {envelope.title}</h2>
                        <p className="mt-1 text-sm text-slate-600">{envelope.description}</p>
                      </div>
                      <span className="text-xs font-bold uppercase tracking-[0.08em] text-slate-500">{envelope.requirements.length} requirement{envelope.requirements.length === 1 ? "" : "s"}</span>
                    </div>
                  </div>

                  {envelope.requirements.length > 0 ? envelope.requirements.map(({ item, sourceIndex }, index) => (
                    <RequirementField
                      key={`${bid.id}-${item.section}`}
                      item={item}
                      fieldIndex={sourceIndex}
                      reference={`${envelope.number.slice(0, 1)}.${index + 1}`}
                      editedAnswer={editedAnswers[item.section] ?? item.answer}
                      editable={canEdit}
                      onAnswerChange={(value) => setEditedAnswers((current) => ({ ...current, [item.section]: value }))}
                      onReset={() => setEditedAnswers((current) => ({ ...current, [item.section]: item.answer }))}
                    />
                  )) : <div className="px-5 py-5 text-sm text-slate-500">No separate response fields were issued under this envelope.</div>}
                </section>
              ))}
            </div>
          </div>

          <section className="border-t-4 border-[#173f68] bg-white" aria-labelledby="legal-attestation-title">
            <div className="border-b border-slate-300 bg-[#edf1f4] px-5 py-4 sm:px-7">
              <p className="text-xs font-extrabold uppercase tracking-[0.12em] text-[#6f520d]">Mandatory certification</p>
              <h2 id="legal-attestation-title" className="mt-1 font-serif text-xl font-bold text-[#173f68]">Legal attestation and electronic authorization</h2>
            </div>

            <div className="p-5 sm:p-7">
              <div className="mb-5 border border-[#d2b45b] bg-[#fff9e5] px-4 py-3 text-sm leading-6 text-[#5f4913]">Submission constitutes an electronic certification for this response record. Both statements are required before the final procurement action can be released.</div>

              <fieldset className="space-y-3" disabled={!canEdit}>
                <legend className="sr-only">Required legal certifications</legend>
                <label className="flex cursor-pointer items-start gap-3 border border-slate-300 bg-slate-50 p-4 text-sm leading-6 text-slate-800 has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-[#1f5b8f]/30">
                  <input type="checkbox" checked={authorizedRepresentative} onChange={(event) => setAuthorizedRepresentative(event.target.checked)} className="mt-1 h-5 w-5 shrink-0 accent-[#173f68]" />
                  <span><strong>Required certification:</strong> I certify that I am an authorized representative of the bidding entity, and all responses are verified against internal capability documentation.</span>
                </label>

                <label className="flex cursor-pointer items-start gap-3 border border-slate-300 bg-slate-50 p-4 text-sm leading-6 text-slate-800 has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-[#1f5b8f]/30">
                  <input type="checkbox" checked={ungroundedAcknowledged} onChange={(event) => setUngroundedAcknowledged(event.target.checked)} className="mt-1 h-5 w-5 shrink-0 accent-[#173f68]" />
                  <span><strong>Required certification:</strong> I acknowledge that un-grounded items have been safely isolated and marked as CAPABILITY_NOT_FOUND.</span>
                </label>
              </fieldset>

              {hasEdits && <div className="mt-4 flex items-start gap-2 border border-[#9b7419] bg-[#fff8df] px-4 py-3 text-sm text-[#6f520d]" role="alert"><Icon name="warning" size={17} /><span>One or more fields differ from the immutable staged response. Restore those answers before authorization.</span></div>}
              {bid.error && <p className="mt-4 text-sm font-semibold text-red-800">{bid.error}</p>}
            </div>

            <div className="flex flex-col gap-4 border-t border-slate-400 bg-[#e9edf1] px-5 py-5 sm:px-7 lg:flex-row lg:items-center lg:justify-between">
              <div className="flex max-w-2xl items-start gap-3 text-sm text-slate-700">
                <span className="mt-0.5 text-[#173f68]"><Icon name="lock" size={18} /></span>
                <div>
                  <p className="font-bold text-slate-900">Human approval gate</p>
                  <p className="mt-1 leading-6">{isApproved ? currentStatus.detail : "The mock portal submit action remains locked until both attestations are checked and this button is activated by an authorized reviewer."}</p>
                </div>
              </div>

              <button type="button" onClick={() => void approveBid()} disabled={submitDisabled} className={`inline-flex min-h-12 min-w-[260px] items-center justify-center gap-2 border-2 px-5 py-3 text-sm font-extrabold uppercase tracking-[0.04em] transition disabled:cursor-not-allowed ${isApproved ? "border-[#2f6b46] bg-[#e8f4ec] text-[#234f35]" : "border-[#0d2d4c] bg-[#173f68] text-white hover:bg-[#0d2d4c] disabled:border-slate-400 disabled:bg-slate-300 disabled:text-slate-600"}`}>
                {approving ? <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" /> : isApproved ? <Icon name="check" size={18} /> : <Icon name="lock" size={17} />}
                {actionLabel}
              </button>
            </div>

            <div className="flex flex-col gap-3 border-t border-slate-300 px-5 py-4 text-xs text-slate-500 sm:flex-row sm:items-center sm:justify-between sm:px-7">
              <span>Controlled record · Session {sessionHash}</span>
              <a className="inline-flex items-center gap-2 font-bold text-[#173f68] underline underline-offset-2 hover:text-[#0d2d4c]" href={bid.review_url} target="_blank" rel="noreferrer">Open staged mock portal record <Icon name="external" size={13} /></a>
            </div>
          </section>
        </div>

        <footer className="flex flex-col gap-1 px-1 py-5 text-xs text-slate-500 sm:flex-row sm:items-center sm:justify-between">
          <span>{AGENCY_NAME} · Authorized procurement use only</span>
          <span>Solicitation {SOLICITATION_REFERENCE}</span>
        </footer>
      </div>
    </main>
  );
}

export default function ReviewPage() {
  return (
    <Suspense fallback={<EmptyState title="Loading response record" body="Preparing the official procurement review form." />}>
      <ReviewContent />
    </Suspense>
  );
}
