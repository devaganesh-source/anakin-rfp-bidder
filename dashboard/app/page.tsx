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
  human_override_sections?: string[];
  human_resolved_sections?: string[];
};

type GenerationStage = "crawling" | "drafting" | "staging";
type EnvelopeId = "technical" | "knowledge" | "security" | "commercial";
type MockPortalAction = "approve" | "submit";

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

function mockPortalApiUrl(bid: BidSnapshot, action: MockPortalAction) {
  const reviewUrl = new URL(bid.review_url);
  const loopbackHosts = new Set(["localhost", "127.0.0.1", "::1", "[::1]"]);
  const expectedPath = `/bids/${encodeURIComponent(bid.portal_bid_id)}/submit`;
  if (
    !["http:", "https:"].includes(reviewUrl.protocol)
    || !loopbackHosts.has(reviewUrl.hostname)
    || reviewUrl.pathname !== expectedPath
    || Boolean(reviewUrl.search)
    || Boolean(reviewUrl.hash)
  ) {
    throw new Error("Bids can only be submitted to the configured local mock API.");
  }
  return new URL(
    `/api/bids/${encodeURIComponent(bid.portal_bid_id)}/${action}`,
    reviewUrl.origin,
  ).toString();
}

function finalizedAnswers(comparison: ReviewSection[]) {
  return Object.fromEntries(
    comparison.map(({ section, answer }) => [section, answer]),
  );
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
  name: "arrow" | "check" | "chevron" | "external" | "lock" | "spark" | "warning";
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
  if (name === "warning") return (
    <svg {...common}>
      <path d="M10.3 4.2 2.1 18a2 2 0 0 0 1.7 3h16.4a2 2 0 0 0 1.7-3L13.7 4.2a2 2 0 0 0-3.4 0Z" />
      <path d="M12 9v4" /><path d="M12 17h.01" />
    </svg>
  );
  return <svg {...common}><path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z" /><path d="m19 16 .7 2.3L22 19l-2.3.7L19 22l-.7-2.3L16 19l2.3-.7L19 16Z" /></svg>;
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
    <section className="glass-panel mb-7 rounded-2xl p-5 sm:p-6" aria-live="polite" aria-label="Bid generation progress">
      <div className="mb-4 flex items-center justify-between gap-4">
        <div>
          <p className="text-[10px] font-bold uppercase tracking-[0.22em] text-indigo-300">Orchestrator in flight</p>
          <p className="mt-1 text-sm text-zinc-300">Preparing a new grounded review session</p>
        </div>
        <span className="h-2 w-2 animate-pulse rounded-full bg-indigo-300 shadow-[0_0_14px_rgba(165,180,252,.9)]" />
      </div>
      <ol className="grid gap-2 sm:grid-cols-3">
        {GENERATION_STAGES.map((item, index) => (
          <li key={item.key} className={`flex items-center gap-3 rounded-xl border px-3 py-3 text-xs ${index === activeIndex ? "border-indigo-300/35 bg-indigo-300/10 text-indigo-100" : index < activeIndex ? "border-emerald-300/20 bg-emerald-300/[0.06] text-emerald-200" : "border-zinc-800 bg-zinc-950/30 text-zinc-500"}`}>
            <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full border border-current font-mono text-[10px]">{index < activeIndex ? "✓" : `0${index + 1}`}</span>
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
    <button type="button" onClick={onClick} disabled={disabled} className="inline-flex min-h-[46px] items-center justify-center gap-2 rounded-xl border border-indigo-300/30 bg-indigo-300/[0.08] px-4 py-2.5 text-sm font-semibold text-indigo-100 transition hover:border-indigo-200/50 hover:bg-indigo-300/[0.16] disabled:cursor-not-allowed disabled:opacity-60">
      {disabled ? <span className="h-4 w-4 animate-spin rounded-full border-2 border-indigo-200/30 border-t-indigo-200" /> : <Icon name="spark" size={16} />}
      {disabled ? "Generating bid…" : "Generate new bid"}
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

function RequirementField({ item, fieldIndex, reference, editedAnswer, editable, capabilityMissing, resolved, onAnswerChange, onReset, onResolvedChange }: { item: ReviewSection; fieldIndex: number; reference: string; editedAnswer: string; editable: boolean; capabilityMissing: boolean; resolved: boolean; onAnswerChange: (value: string) => void; onReset: () => void; onResolvedChange: (checked: boolean) => void }) {
  const [citationOpen, setCitationOpen] = useState(false);
  const grounded = !capabilityMissing && Boolean(item.source_snippet.trim());
  const hasEdit = editedAnswer !== item.answer;
  const hasManualResponse = capabilityMissing && Boolean(editedAnswer.trim()) && !isCapabilityMissing(editedAnswer);
  const fieldId = `response-field-${fieldIndex}`;
  const citationId = `citation-${fieldIndex}`;
  const resolvedId = `response-resolved-${fieldIndex}`;

  return (
    <article className={`border-t first:border-t-0 ${capabilityMissing ? resolved ? "border-emerald-500 border-l-4 border-l-emerald-500 bg-emerald-50/30" : "border-red-300 border-l-4 border-l-red-500" : "border-slate-300"}`}>
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
            ) : capabilityMissing ? resolved ? (
              <span className="inline-flex items-center gap-2 border border-emerald-600 bg-emerald-50 px-2.5 py-1.5 text-xs font-extrabold text-emerald-700"><Icon name="check" size={14} /> [HUMAN OVERRIDE APPLIED]</span>
            ) : (
              <span className="inline-flex items-center gap-2 border border-red-500 bg-red-50 px-2.5 py-1.5 text-xs font-extrabold text-red-700"><Icon name="warning" size={14} /> {hasManualResponse ? "[CAPABILITY_NOT_FOUND — MANUAL REVIEW REQUIRED]" : "[CAPABILITY_NOT_FOUND — HUMAN INPUT REQUIRED]"}</span>
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
            placeholder={capabilityMissing ? "Enter a verified manual response, then mark it reviewed and resolved." : undefined}
            aria-describedby={`${fieldId}-help`}
            className={`w-full resize-y border p-3.5 text-base leading-7 text-slate-900 shadow-inner outline-none transition focus:border-[#1f5b8f] focus:ring-2 focus:ring-[#1f5b8f]/20 ${capabilityMissing ? resolved ? "border-emerald-500 bg-emerald-50" : hasManualResponse ? "border-amber-500 bg-amber-50" : "border-red-500 bg-red-50" : hasEdit ? "border-[#9b7419] bg-[#fffdf4]" : "border-slate-400 bg-white"} ${!editable ? "cursor-default text-slate-700" : ""}`}
          />
          <div id={`${fieldId}-help`} className="mt-2 flex flex-wrap items-start justify-between gap-2 text-xs text-slate-500">
            <span>{editable ? capabilityMissing ? "Enter a verified human response and confirm its review below." : "Editable working field. Review and amend the response before authorization." : "This response field is locked because the bid is no longer awaiting approval."}</span>
            {hasEdit && <button type="button" onClick={onReset} className="font-bold text-[#173f68] underline underline-offset-2 hover:text-[#0d2d4c]">Restore staged answer</button>}
          </div>

          {capabilityMissing && (
            <label htmlFor={resolvedId} className={`mt-4 flex items-center gap-3 border px-3.5 py-3 text-sm font-bold transition ${resolved ? "border-emerald-500 bg-emerald-50 text-emerald-800" : "border-amber-400 bg-amber-50 text-amber-800"} ${!editable || !hasManualResponse ? "cursor-not-allowed opacity-60" : "cursor-pointer"}`}>
              <input id={resolvedId} type="checkbox" checked={resolved} disabled={!editable || !hasManualResponse} onChange={(event) => onResolvedChange(event.target.checked)} className="h-4 w-4 shrink-0 accent-emerald-600" />
              <span aria-hidden="true">✅</span>
              <span>Manual entry reviewed and resolved</span>
            </label>
          )}

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

function StagedReview({ bidId, onBack, onSnapshot }: { bidId: string; onBack: () => void; onSnapshot: (snapshot: BidSnapshot) => void }) {
  const [bid, setBid] = useState<BidSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [approving, setApproving] = useState(false);
  const [approvalReleased, setApprovalReleased] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [editedAnswers, setEditedAnswers] = useState<Record<string, string>>({});
  const [resolvedItems, setResolvedItems] = useState<Record<string, boolean>>({});
  const [authorizedRepresentative, setAuthorizedRepresentative] = useState(false);
  const [ungroundedAcknowledged, setUngroundedAcknowledged] = useState(false);
  const initializedBidRef = useRef<string | null>(null);

  useEffect(() => {
    if (!bid || initializedBidRef.current === bid.id) return;
    initializedBidRef.current = bid.id;
    const released = bid.approved || bid.status === "submitting" || bid.status === "submitted";
    setEditedAnswers(Object.fromEntries(bid.comparison.map((item) => [item.section, item.answer])));
    setResolvedItems(Object.fromEntries((bid.human_resolved_sections ?? []).map((section) => [section, true])));
    setApprovalReleased(released);
    setIsSubmitted(bid.submitted || bid.status === "submitted");
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
      const snapshot = restoreSavedBidSnapshot((await response.json()) as BidSnapshot);
      setBid(snapshot);
      onSnapshot(snapshot);
      if (snapshot.approved || snapshot.status === "submitting" || snapshot.status === "submitted") setApprovalReleased(true);
      if (snapshot.submitted || snapshot.status === "submitted") setIsSubmitted(true);
    } catch {
      // A missing or unavailable bid leaves the review route empty by design.
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [bidId, onSnapshot]);

  useEffect(() => {
    void loadBid();
    if (!bidId) return;
    const interval = window.setInterval(() => void loadBid(true), POLL_INTERVAL_MS);
    return () => window.clearInterval(interval);
  }, [bidId, loadBid]);

  const approveAndSubmit = async () => {
    if (!bidId || !bid || bid.status !== "awaiting_approval" || !authorizedRepresentative || !ungroundedAcknowledged) return;
    setApproving(true);
    try {
      const bidWithOverrides: BidSnapshot = {
        ...bid,
        comparison: bid.comparison.map((item) => ({
          ...item,
          answer: editedAnswers[item.section] ?? item.answer,
        })),
      };
      const responseOverrides = bidWithOverrides.comparison
        .filter((item, index) => item.answer !== bid.comparison[index]?.answer)
        .map(({ section, answer }) => ({ section, answer }));
      const answers = finalizedAnswers(bidWithOverrides.comparison);
      const portalApprovalResponse = await fetch(mockPortalApiUrl(bid, "approve"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          answers,
          authorized_representative: true,
          ungrounded_items_acknowledged: true,
        }),
        signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      });
      if (!portalApprovalResponse.ok) {
        const detail = await portalApprovalResponse.text();
        throw new Error(detail || `Approval failed (${portalApprovalResponse.status}).`);
      }

      const submissionResponse = await fetch(mockPortalApiUrl(bid, "submit"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ answers }),
        signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      });
      if (!submissionResponse.ok) {
        const detail = await submissionResponse.text();
        throw new Error(detail || `Submission failed (${submissionResponse.status}).`);
      }
      const submission = (await submissionResponse.json()) as { status?: string; submitted?: boolean };
      if (submission.status !== "success" || submission.submitted !== true) {
        throw new Error("The mock procurement API did not confirm submission.");
      }

      const overrideSections = bid.comparison
        .filter((item) => isCapabilityMissing(item.answer) || bid.human_override_sections?.includes(item.section))
        .map((item) => item.section);
      const submittedSnapshot: BidSnapshot = {
        ...bidWithOverrides,
        status: "submitted",
        approved: true,
        submitted: true,
        human_override_sections: overrideSections,
        human_resolved_sections: overrideSections.filter((section) => resolvedItems[section]),
      };
      setIsSubmitted(true);
      setApprovalReleased(true);
      setBid(submittedSnapshot);
      onSnapshot(submittedSnapshot);

      // Synchronize the pipeline's HITL record after the frontend-owned JSON
      // submission. Its gated callback now observes the already-submitted API state.
      await fetch(apiUrl(`/api/bids/${encodeURIComponent(bidId)}/approve`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          bid: bidWithOverrides,
          solicitation_reference: SOLICITATION_REFERENCE,
          session_id: bid.id,
          attestations: {
            authorized_representative: true,
            ungrounded_items_acknowledged: true,
          },
          response_overrides: responseOverrides,
        }),
        signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      });
    } catch {
      // The backend remains authoritative; polling will reflect any completed action.
    } finally {
      setApproving(false);
    }
  };

  // Approval and submission run sequentially for one click. Polling may continue
  // concurrently, while the button guard prevents duplicate client requests.

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

  if (loading && !bid) return <EmptyState title="Loading response record" body="Retrieving the staged submission from the local procurement workflow." action={<button type="button" onClick={onBack} className="border border-[#173f68] px-4 py-2.5 text-sm font-bold text-[#173f68]">Back to dashboard</button>} />;
  if (!bid) return null;

  if (isSubmitted) {
    return (
      <main className="grid min-h-screen place-items-center bg-white px-6 py-16 text-slate-900">
        <section className="w-full max-w-2xl text-center" aria-labelledby="submission-confirmation-title">
          <div className="mx-auto grid h-28 w-28 place-items-center rounded-full border-4 border-emerald-700 bg-emerald-50 text-emerald-700 shadow-sm">
            <Icon name="check" size={64} />
          </div>
          <p className="mt-8 text-xs font-extrabold uppercase tracking-[0.18em] text-emerald-700">Submission confirmed</p>
          <h1 id="submission-confirmation-title" className="mt-3 font-serif text-3xl font-bold text-blue-950 sm:text-5xl">
            Official Procurement Package Submitted Successfully
          </h1>
          <p className="mx-auto mt-5 max-w-xl text-base leading-7 text-slate-600">
            The finalized response package has been recorded by the local procurement API.
          </p>
          <div className="mx-auto mt-8 max-w-lg rounded-xl border border-slate-200 bg-slate-50 px-6 py-5 shadow-sm">
            <p className="text-xs font-bold uppercase tracking-[0.14em] text-slate-500">Session ID</p>
            <p className="mt-2 break-all font-mono text-sm font-bold text-slate-900">{bid.id}</p>
          </div>
          <button type="button" onClick={onBack} className="mt-8 border-2 border-blue-900 px-6 py-3 text-sm font-bold text-blue-900 transition-colors hover:bg-blue-50">
            Return to dashboard
          </button>
        </section>
      </main>
    );
  }

  const currentStatus = statusCopy(bid.status);
  const groundedCount = bid.comparison.filter((item) => !isCapabilityMissing(item.answer) && Boolean(item.source_snippet.trim())).length;
  const exceptionCount = bid.comparison.length - groundedCount;
  const complianceVerified = exceptionCount === 0;
  const sessionHash = bid.id.slice(0, 16).toUpperCase();
  const submitDisabled = approving || isSubmitted || bid.status !== "awaiting_approval" || !authorizedRepresentative || !ungroundedAcknowledged;

  let actionLabel = "Approve & Submit";
  if (approving) actionLabel = "Recording authorization…";
  else if (isSubmitted) actionLabel = "Submitted";
  else if (isApproved) actionLabel = "Submission authorized";
  else if (!attestationsComplete) actionLabel = "Complete certifications to continue";

  return (
    <main className="min-h-screen bg-[#dfe5ea] pb-12 text-slate-900 print:bg-white print:pb-0">
      <header className="border-b-4 border-[#b58a2a] bg-[#123a60] text-white shadow-md print:border-b-2 print:border-slate-800 print:bg-white print:text-slate-900 print:shadow-none">
        <div className="mx-auto flex max-w-6xl flex-col gap-5 px-4 py-5 sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:px-8">
          <div className="flex items-center gap-4">
            <div className="grid h-14 w-14 shrink-0 place-items-center border-2 border-white/80 font-serif text-base font-bold tracking-[0.08em]">GESD</div>
            <div>
              <p className="text-xs font-bold uppercase tracking-[0.14em] text-blue-100 print:text-slate-600">Official RFP submission package</p>
              <h1 className="mt-1 font-serif text-xl font-bold sm:text-2xl">{AGENCY_NAME}</h1>
              <p className="mt-1 text-sm text-blue-100 print:text-slate-600">Procurement transmission copy · Controlled document</p>
            </div>
          </div>
          <button type="button" onClick={onBack} className="print:hidden border border-white/60 px-4 py-2.5 text-sm font-bold text-white transition hover:bg-white/10">Back to dashboard</button>
        </div>
      </header>

      <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6 lg:px-8">
        <div className="border border-[#8292a0] bg-white shadow-[0_14px_35px_rgba(21,45,69,0.14)] print:border-slate-700 print:shadow-none">
          <section className="border-b border-slate-300 px-5 py-5 sm:px-7" aria-labelledby="form-title">
            <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
              <div>
                <p className="text-xs font-extrabold uppercase tracking-[0.12em] text-[#6f520d]">Ready-Made RFP Submission Package</p>
                <h2 id="form-title" className="mt-2 font-serif text-3xl font-bold text-[#173f68]">Enterprise Knowledge Management & Response Automation Platform</h2>
                <p className="mt-2 text-base text-slate-600">Compiled final-response binder for authorized review, certification, and electronic transmission.</p>
              </div>
              <div className={`shrink-0 border px-4 py-3 ${statusStyles(bid.status)}`}>
                <p className="text-xs font-extrabold uppercase tracking-[0.1em]">{currentStatus.short}</p>
                <p className="mt-1 text-sm font-semibold">{currentStatus.label}</p>
              </div>
            </div>

            <div className="mt-6 border border-slate-300" aria-label="Submission package metadata">
              <p className="border-b border-slate-300 bg-[#173f68] px-3.5 py-2 text-xs font-extrabold uppercase tracking-[0.12em] text-white print:bg-slate-100 print:text-slate-800">Document control · Transmittal metadata</p>
              <dl className="grid sm:grid-cols-2 lg:grid-cols-4">
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
            </div>
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
                <section key={envelope.id} className="border border-[#8d9aa6] bg-white shadow-sm print:break-inside-avoid print:shadow-none" aria-labelledby={`envelope-${envelope.id}`}>
                  <div className="border-b-2 border-[#173f68] bg-[#eaf0f5] px-5 py-4">
                    <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
                      <div>
                        <p className="text-xs font-extrabold uppercase tracking-[0.12em] text-[#6f520d]">Sealed response envelope · {envelope.number}</p>
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
                      capabilityMissing={isCapabilityMissing(item.answer) || Boolean(bid.human_override_sections?.includes(item.section))}
                      resolved={Boolean(resolvedItems[item.section])}
                      onAnswerChange={(value) => {
                        setEditedAnswers((current) => ({ ...current, [item.section]: value }));
                        setResolvedItems((current) => ({ ...current, [item.section]: false }));
                      }}
                      onReset={() => {
                        setEditedAnswers((current) => ({ ...current, [item.section]: item.answer }));
                        setResolvedItems((current) => ({ ...current, [item.section]: false }));
                      }}
                      onResolvedChange={(checked) => setResolvedItems((current) => ({ ...current, [item.section]: checked }))}
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

              {hasEdits && <div className="mb-4 flex items-start gap-2 border border-[#9b7419] bg-[#fff8df] px-4 py-3 text-sm text-[#6f520d]" role="status"><Icon name="warning" size={17} /><span>Manual changes are highlighted in amber. Review each override before signing this package.</span></div>}
              {bid.error && <p className="mb-4 text-sm font-semibold text-red-800">{bid.error}</p>}

              <fieldset className="space-y-3" disabled={!canEdit}>
                <legend className="sr-only">Required legal certifications</legend>
                <label className="flex cursor-pointer items-start gap-3 border border-slate-300 bg-slate-50 p-4 text-sm leading-6 text-slate-800 has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-[#1f5b8f]/30">
                  <input type="checkbox" checked={authorizedRepresentative} onChange={(event) => setAuthorizedRepresentative(event.target.checked)} className="mt-1 h-5 w-5 shrink-0 accent-[#173f68]" />
                  <span><strong>Required certification:</strong> I certify that I am an authorized representative of the bidding entity, and all responses are verified against internal capability documentation.</span>
                </label>

                <label className="flex cursor-pointer items-start gap-3 border-2 border-[#9b7419] bg-[#fff9e5] p-4 text-sm leading-6 text-slate-900 has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-[#1f5b8f]/30">
                  <input type="checkbox" required aria-required="true" checked={ungroundedAcknowledged} onChange={(event) => setUngroundedAcknowledged(event.target.checked)} className="mt-1 h-5 w-5 shrink-0 accent-[#173f68]" />
                  <span><strong>Required certification:</strong> I acknowledge that un-grounded items have been safely isolated and marked as CAPABILITY_NOT_FOUND</span>
                </label>
              </fieldset>
            </div>

            <div className="flex flex-col gap-4 border-t border-slate-400 bg-[#e9edf1] px-5 py-5 sm:px-7 lg:flex-row lg:items-center lg:justify-between">
              <div className="flex max-w-2xl items-start gap-3 text-sm text-slate-700">
                <span className="mt-0.5 text-[#173f68]"><Icon name="lock" size={18} /></span>
                <div>
                  <p className="font-bold text-slate-900">Human approval gate</p>
                  <p className="mt-1 leading-6">{isApproved ? currentStatus.detail : "The JSON submission action remains locked until both attestations are checked and this button is activated by an authorized reviewer."}</p>
                </div>
              </div>

              <button type="button" onClick={() => void approveAndSubmit()} disabled={submitDisabled} className={`inline-flex min-h-12 min-w-[260px] items-center justify-center gap-2 border-2 px-5 py-3 text-sm font-extrabold uppercase tracking-[0.04em] transition disabled:cursor-not-allowed ${isApproved ? "border-[#2f6b46] bg-[#e8f4ec] text-[#234f35]" : "border-[#0d2d4c] bg-[#173f68] text-white hover:bg-[#0d2d4c] disabled:border-slate-400 disabled:bg-slate-300 disabled:text-slate-600"}`}>
                {approving ? <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" /> : isApproved ? <Icon name="check" size={18} /> : <Icon name="lock" size={17} />}
                {actionLabel}
              </button>
            </div>

            <div className="flex flex-col gap-3 border-t border-slate-300 px-5 py-4 text-xs text-slate-500 sm:flex-row sm:items-center sm:justify-between sm:px-7">
              <span>Controlled record · Session {sessionHash}</span>
              <span className="font-semibold text-slate-600">Final review and submission are completed in this secure workspace.</span>
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

const KNOWN_BIDS_STORAGE_KEY = "anakin-rfp-bid-ids";
const BID_SNAPSHOT_STORAGE_PREFIX = "anakin-rfp-bid-snapshot:";

function saveBidSnapshot(snapshot: BidSnapshot) {
  window.localStorage.setItem(`${BID_SNAPSHOT_STORAGE_PREFIX}${snapshot.id}`, JSON.stringify(snapshot));
}

function restoreSavedBidSnapshot(snapshot: BidSnapshot): BidSnapshot {
  try {
    const saved = JSON.parse(window.localStorage.getItem(`${BID_SNAPSHOT_STORAGE_PREFIX}${snapshot.id}`) ?? "null") as BidSnapshot | null;
    if (!saved || saved.id !== snapshot.id || !Array.isArray(saved.comparison) || saved.human_override_sections === undefined) return snapshot;
    const savedAnswers = new Map(saved.comparison.map((item) => [item.section, item.answer]));
    return {
      ...snapshot,
      comparison: snapshot.comparison.map((item) => ({
        ...item,
        answer: savedAnswers.get(item.section) ?? item.answer,
      })),
      human_override_sections: saved.human_override_sections,
      human_resolved_sections: saved.human_resolved_sections ?? [],
    };
  } catch {
    window.localStorage.removeItem(`${BID_SNAPSHOT_STORAGE_PREFIX}${snapshot.id}`);
    return snapshot;
  }
}

function dashboardStatusStyles(status: BidStatus) {
  if (status === "submitted") return {
    badge: "border-emerald-400/25 bg-emerald-400/[0.1] text-emerald-200",
    dot: "bg-emerald-300 shadow-[0_0_12px_rgba(110,231,183,.9)]",
  };
  if (status === "manual_intervention") return {
    badge: "border-rose-400/25 bg-rose-400/[0.1] text-rose-200",
    dot: "bg-rose-300 shadow-[0_0_12px_rgba(253,164,175,.9)]",
  };
  if (status === "submitting") return {
    badge: "border-indigo-400/25 bg-indigo-400/[0.1] text-indigo-200",
    dot: "bg-indigo-300 shadow-[0_0_12px_rgba(129,140,248,.9)]",
  };
  return {
    badge: "border-amber-400/20 bg-amber-400/[0.08] text-amber-200",
    dot: "bg-amber-300 shadow-[0_0_12px_rgba(252,211,77,.9)]",
  };
}

function DashboardStatusBadge({ status }: { status: BidStatus }) {
  const current = statusCopy(status);
  const styles = dashboardStatusStyles(status);
  return (
    <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-[10px] font-bold uppercase tracking-[0.16em] ${styles.badge}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${styles.dot}`} />
      {current.short}
    </span>
  );
}

function DashboardContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const reviewBidId = searchParams.get("review") || searchParams.get("bid") || "";
  const [knownBidIds, setKnownBidIds] = useState<string[]>([]);
  const [bids, setBids] = useState<BidSnapshot[]>([]);
  const [registryReady, setRegistryReady] = useState(false);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [generationStage, setGenerationStage] = useState<GenerationStage>("crawling");
  const [error, setError] = useState<string | null>(null);
  const generationRequestInFlightRef = useRef(false);

  const rememberBid = useCallback((snapshot: BidSnapshot) => {
    saveBidSnapshot(snapshot);
    setBids((current) => [snapshot, ...current.filter((item) => item.id !== snapshot.id)]);
    setKnownBidIds((current) => {
      const next = [snapshot.id, ...current.filter((id) => id !== snapshot.id)];
      window.localStorage.setItem(KNOWN_BIDS_STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  useEffect(() => {
    let storedIds: string[] = [];
    try {
      const stored = JSON.parse(window.localStorage.getItem(KNOWN_BIDS_STORAGE_KEY) ?? "[]") as unknown;
      if (Array.isArray(stored)) storedIds = stored.filter((id): id is string => typeof id === "string" && Boolean(id));
    } catch {
      window.localStorage.removeItem(KNOWN_BIDS_STORAGE_KEY);
    }
    const configuredId = process.env.NEXT_PUBLIC_BID_ID || "";
    const initialIds = Array.from(new Set([reviewBidId, configuredId, ...storedIds].filter(Boolean)));
    setKnownBidIds(initialIds);
    setRegistryReady(true);
  }, [reviewBidId]);

  const loadTrackedBids = useCallback(async (background = false) => {
    if (!registryReady || reviewBidId) return;
    if (background) setRefreshing(true);
    else setLoading(true);
    try {
      const results = await Promise.allSettled(knownBidIds.map(async (id) => {
        const response = await fetch(apiUrl(`/api/bids/${encodeURIComponent(id)}`), {
          cache: "no-store",
          signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
        });
        if (!response.ok) throw new Error(`Unable to load bid ${id.slice(0, 8)} (${response.status}).`);
        return restoreSavedBidSnapshot((await response.json()) as BidSnapshot);
      }));
      const snapshots = results.flatMap((result) => result.status === "fulfilled" ? [result.value] : []);
      setBids(snapshots);
      setError(null);
    } catch {
      setBids([]);
      setError(null);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [knownBidIds, registryReady, reviewBidId]);

  useEffect(() => {
    void loadTrackedBids();
    if (!registryReady || reviewBidId) return;
    const interval = window.setInterval(() => void loadTrackedBids(true), POLL_INTERVAL_MS);
    return () => window.clearInterval(interval);
  }, [loadTrackedBids, registryReady, reviewBidId]);

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

  const generateBid = async () => {
    if (generationRequestInFlightRef.current) return;
    generationRequestInFlightRef.current = true;
    setGenerating(true);
    setError(null);
    try {
      // Deliberately no client timeout: disconnecting does not cancel server work,
      // and a retry could otherwise start a duplicate full pipeline run.
      const response = await fetch(apiUrl("/api/bids/generate"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
      if (!response.ok) {
        let message = `Bid generation failed (${response.status}).`;
        try {
          const payload = (await response.json()) as { detail?: { message?: string } | string };
          if (typeof payload.detail === "string") message = payload.detail;
          else if (payload.detail?.message) message = payload.detail.message;
        } catch {
          // Keep the stable fallback when the backend returns a non-JSON error.
        }
        throw new Error(message);
      }
      const snapshot = (await response.json()) as BidSnapshot;
      rememberBid(snapshot);
      router.push(`/?review=${encodeURIComponent(snapshot.id)}`);
    } catch (requestError) {
      setError(requestErrorMessage(requestError, "Bid generation"));
    } finally {
      generationRequestInFlightRef.current = false;
      setGenerating(false);
    }
  };

  // Bid refreshes happen concurrently; the synchronous ref blocks duplicate
  // generation requests before React has time to commit a disabled button.

  const awaitingCount = bids.filter((bid) => bid.status === "awaiting_approval").length;
  const submittedCount = bids.filter((bid) => bid.status === "submitted").length;
  const exceptionCount = bids.reduce((total, bid) => total + bid.comparison.filter((item) => isCapabilityMissing(item.answer) || !item.source_snippet.trim()).length, 0);

  if (reviewBidId) {
    return <StagedReview key={reviewBidId} bidId={reviewBidId} onBack={() => router.push("/")} onSnapshot={rememberBid} />;
  }

  return (
    <main className="relative min-h-screen overflow-hidden px-4 py-6 text-zinc-100 sm:px-6 lg:px-10 lg:py-9">
      <div className="pointer-events-none fixed inset-0 overflow-hidden" aria-hidden="true">
        <div className="absolute left-[5%] top-[-14rem] h-[34rem] w-[34rem] rounded-full bg-indigo-600/15 blur-[130px]" />
        <div className="absolute bottom-[-16rem] right-[4%] h-[34rem] w-[34rem] rounded-full bg-emerald-500/[0.09] blur-[140px]" />
      </div>

      <div className="relative z-10 mx-auto max-w-7xl">
        <header className="mb-9 flex flex-col gap-7 border-b border-zinc-800/70 pb-8 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <div className="mb-7 flex items-center gap-3 text-[11px] font-bold uppercase tracking-[0.24em] text-zinc-400">
              <span className="grid h-8 w-8 place-items-center rounded-xl border border-indigo-300/30 bg-indigo-300/10 text-sm text-indigo-200 shadow-[0_0_25px_rgba(129,140,248,.16)]">A</span>
              <span className="text-zinc-600">/</span>
              <span>Bid operations desk</span>
            </div>
            <p className="mb-3 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.25em] text-indigo-300"><Icon name="spark" size={14} /> Grounded proposal workspace</p>
            <h1 className="max-w-3xl text-4xl font-semibold tracking-[-0.055em] text-zinc-50 sm:text-5xl lg:text-6xl">Bids in motion.<br /><span className="text-gradient">Humans in control.</span></h1>
            <p className="mt-5 max-w-2xl text-sm leading-7 text-zinc-400 sm:text-base">Track staged responses, inspect grounding exceptions, and open the formal submission binder when a bid is ready for review.</p>
          </div>
          <div className="flex flex-col items-start gap-3 lg:items-end">
            <GenerateNewBidButton disabled={generating} onClick={() => void generateBid()} />
            <p className="flex items-center gap-2 text-xs text-zinc-600"><span className={`h-1.5 w-1.5 rounded-full ${refreshing ? "animate-ping bg-indigo-300" : "bg-emerald-300"}`} /> {refreshing ? "Refreshing bid states" : `Live sync every ${POLL_INTERVAL_MS / 1000}s`}</p>
          </div>
        </header>

        {error && (
          <div role="alert" className="mb-7 flex flex-col gap-3 rounded-2xl border border-rose-400/25 bg-rose-400/[0.08] px-4 py-3 text-sm text-rose-200 sm:flex-row sm:items-center sm:justify-between">
            <span className="flex items-center gap-2"><Icon name="warning" size={16} /> {error}</span>
            <button type="button" className="shrink-0 font-semibold text-rose-100 underline decoration-rose-300/40 underline-offset-4" onClick={() => void loadTrackedBids()}>Retry</button>
          </div>
        )}

        {generating && <GenerationProgress stage={generationStage} />}

        <section className="mb-9 grid gap-3 sm:grid-cols-2 xl:grid-cols-4" aria-label="Bid metrics">
          {[
            ["Tracked bids", String(bids.length), "Stored in this browser", "text-indigo-200"],
            ["Awaiting review", String(awaitingCount), "Human approval gate locked", "text-amber-200"],
            ["Submitted", String(submittedCount), "Mock portal confirmed", "text-emerald-200"],
            ["Exceptions", String(exceptionCount), "Items requiring source review", "text-rose-200"],
          ].map(([label, value, detail, tone]) => (
            <article key={label} className="glass-panel rounded-2xl px-5 py-5">
              <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-zinc-600">{label}</p>
              <p className={`mt-3 font-mono text-3xl font-semibold tracking-tight ${tone}`}>{value}</p>
              <p className="mt-2 text-xs text-zinc-500">{detail}</p>
            </article>
          ))}
        </section>

        <section className="glass-panel overflow-hidden rounded-3xl" aria-labelledby="bid-list-title">
          <div className="flex flex-col gap-3 border-b border-zinc-800/80 px-5 py-5 sm:flex-row sm:items-end sm:justify-between sm:px-7">
            <div>
              <p className="text-[10px] font-bold uppercase tracking-[0.25em] text-zinc-600">01 / active register</p>
              <h2 id="bid-list-title" className="mt-2 text-xl font-semibold tracking-tight text-zinc-100">Bid review queue</h2>
            </div>
            <p className="text-xs text-zinc-500">Select a staged record to open its official binder.</p>
          </div>

          {loading && registryReady ? (
            <div className="flex items-center gap-3 px-6 py-12 text-sm text-zinc-500"><span className="h-4 w-4 animate-spin rounded-full border-2 border-zinc-700 border-t-indigo-300" /> Loading tracked bids…</div>
          ) : bids.length === 0 ? (
            <div className="px-6 py-14 text-center sm:px-8">
              <span className="mx-auto grid h-12 w-12 place-items-center rounded-2xl border border-indigo-300/20 bg-indigo-300/[0.08] text-indigo-200"><Icon name="spark" size={20} /></span>
              <h3 className="mt-5 text-lg font-semibold text-zinc-200">No staged bids tracked yet</h3>
              <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-zinc-500">Generate a bid here, or open a known bid link once, and it will appear in this browser&apos;s review queue.</p>
            </div>
          ) : (
            <div className="divide-y divide-zinc-800/80">
              {bids.map((bid, index) => {
                const groundedCount = bid.comparison.filter((item) => !isCapabilityMissing(item.answer) && Boolean(item.source_snippet.trim())).length;
                return (
                  <article key={bid.id} className="group grid gap-5 px-5 py-5 transition hover:bg-indigo-300/[0.035] sm:px-7 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,.75fr)_auto] lg:items-center">
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-3">
                        <span className="font-mono text-[10px] text-zinc-600">{String(index + 1).padStart(2, "0")}</span>
                        <DashboardStatusBadge status={bid.status} />
                      </div>
                      <h3 className="mt-3 truncate font-mono text-base font-semibold text-zinc-100">{bid.portal_bid_id}</h3>
                      <p className="mt-1 truncate text-xs text-zinc-600">Session {bid.id}</p>
                    </div>
                    <dl className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <dt className="text-[10px] font-bold uppercase tracking-[0.16em] text-zinc-600">Responses</dt>
                        <dd className="mt-1 font-mono text-zinc-300">{bid.comparison.length}</dd>
                      </div>
                      <div>
                        <dt className="text-[10px] font-bold uppercase tracking-[0.16em] text-zinc-600">Grounded</dt>
                        <dd className="mt-1 font-mono text-emerald-200">{groundedCount} / {bid.comparison.length}</dd>
                      </div>
                    </dl>
                    <button type="button" onClick={() => router.push(`/?review=${encodeURIComponent(bid.id)}`)} className="inline-flex min-h-11 items-center justify-center gap-2 rounded-xl bg-indigo-300 px-4 py-2.5 text-sm font-bold text-indigo-950 transition hover:bg-indigo-200 lg:min-w-44">
                      Open staged review <Icon name="arrow" size={16} />
                    </button>
                  </article>
                );
              })}
            </div>
          )}
        </section>

        <footer className="flex flex-col gap-2 py-8 text-[10px] font-semibold uppercase tracking-[0.18em] text-zinc-700 sm:flex-row sm:items-center sm:justify-between">
          <span>Anakin / grounded automation</span>
          <span>Read · Reason · Act · Human approval</span>
        </footer>
      </div>
    </main>
  );
}

export default function DashboardPage() {
  return (
    <Suspense fallback={<main className="grid min-h-screen place-items-center text-sm text-zinc-500">Loading bid operations…</main>}>
      <DashboardContent />
    </Suspense>
  );
}
