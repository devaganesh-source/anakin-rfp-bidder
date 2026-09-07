"use client";

import {
  AnimatePresence,
  motion,
  useMotionValue,
  useSpring,
} from "framer-motion";
import {
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent,
} from "react";
import { useRouter, useSearchParams } from "next/navigation";

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

type GenerationStage = "crawling" | "drafting" | "staging";

const GENERATION_STAGES: { key: GenerationStage; label: string }[] = [
  { key: "crawling", label: "Crawling RFP..." },
  { key: "drafting", label: "Drafting responses..." },
  { key: "staging", label: "Staging bid..." },
];

const TIMELINE_STEPS = [
  { label: "Crawled", detail: "RFP captured" },
  { label: "Drafted", detail: "Grounded responses ready" },
  { label: "Staged", detail: "Portal draft prepared" },
  { label: "Awaiting approval", detail: "Human review required" },
  { label: "Approved / rejected", detail: "Human decision" },
  { label: "Submitted", detail: "Portal confirmation" },
] as const;

const POLL_INTERVAL_MS = 5000;
const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8001").replace(/\/$/, "");

function apiUrl(path: string) {
  return `${API_BASE_URL}${path}`;
}

function requestErrorMessage(error: unknown, action: string) {
  if (error instanceof TypeError) {
    return `Backend unavailable at ${API_BASE_URL}. Start the FastAPI service and try again.`;
  }
  return error instanceof Error ? error.message : `${action} failed. Please try again.`;
}

function Icon({ name, size = 18 }: { name: "arrow" | "check" | "chevron" | "external" | "lock" | "spark" | "warning"; size?: number }) {
  const common = { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, "aria-hidden": true };
  if (name === "check") return <svg {...common}><path d="m5 12 4.2 4.2L19 6.8" /></svg>;
  if (name === "arrow") return <svg {...common}><path d="M5 12h13" /><path d="m13 6 6 6-6 6" /></svg>;
  if (name === "chevron") return <svg {...common}><path d="m9 18 6-6-6-6" /></svg>;
  if (name === "external") return <svg {...common}><path d="M14 5h5v5" /><path d="M19 5 11 13" /><path d="M18 13v5a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5" /></svg>;
  if (name === "lock") return <svg {...common}><rect x="5" y="10" width="14" height="10" rx="2" /><path d="M8 10V7a4 4 0 0 1 8 0v3" /></svg>;
  if (name === "warning") return <svg {...common}><path d="M10.3 4.2 2.1 18a2 2 0 0 0 1.7 3h16.4a2 2 0 0 0 1.7-3L13.7 4.2a2 2 0 0 0-3.4 0Z" /><path d="M12 9v4" /><path d="M12 17h.01" /></svg>;
  return <svg {...common}><path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z" /><path d="m19 16 .7 2.3L22 19l-2.3.7L19 22l-.7-2.3L16 19l2.3-.7L19 16Z" /></svg>;
}

function statusCopy(status: BidStatus) {
  switch (status) {
    case "awaiting_approval":
      return { label: "Staged / awaiting approval", short: "STAGED", tone: "amber", detail: "Review is locked and ready for a human decision." };
    case "submitting":
      return { label: "Approval released", short: "RELEASED", tone: "indigo", detail: "The approved draft is being submitted." };
    case "submitted":
      return { label: "Submission confirmed", short: "SUBMITTED", tone: "emerald", detail: "The mock portal confirmed the final submission." };
    case "manual_intervention":
      return { label: "Manual intervention", short: "ATTENTION", tone: "rose", detail: "The workflow stopped and needs inspection." };
  }
}

function StatusBadge({ status, refreshing }: { status: BidStatus; refreshing: boolean }) {
  const current = statusCopy(status);
  const tone = {
    amber: "border-amber-400/20 bg-amber-400/[0.08] text-amber-200",
    indigo: "border-indigo-400/25 bg-indigo-400/[0.1] text-indigo-200",
    emerald: "border-emerald-400/25 bg-emerald-400/[0.1] text-emerald-200",
    rose: "border-rose-400/25 bg-rose-400/[0.1] text-rose-200",
  }[current.tone];
  const dot = {
    amber: "bg-amber-300 shadow-[0_0_12px_rgba(252,211,77,.9)]",
    indigo: "bg-indigo-300 shadow-[0_0_12px_rgba(129,140,248,.9)]",
    emerald: "bg-emerald-300 shadow-[0_0_12px_rgba(110,231,183,.9)]",
    rose: "bg-rose-300 shadow-[0_0_12px_rgba(253,164,175,.9)]",
  }[current.tone];

  return (
    <div className={`inline-flex items-center gap-3 rounded-full border px-3.5 py-2 text-[11px] font-semibold uppercase tracking-[0.16em] ${tone}`}>
      <motion.span
        className={`h-2 w-2 rounded-full ${dot}`}
        animate={status === "submitting" || status === "awaiting_approval" ? { opacity: [1, 0.35, 1], scale: [1, 0.82, 1] } : { opacity: 1, scale: 1 }}
        transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
      />
      <span>{current.short}</span>
      <span className="hidden text-zinc-500 sm:inline">/</span>
      <span className="hidden normal-case tracking-normal text-zinc-300 sm:inline">{current.label}</span>
      {refreshing && <span className="ml-1 h-1 w-1 animate-ping rounded-full bg-current" />}
    </div>
  );
}

function generationStageIndex(stage: GenerationStage) {
  return GENERATION_STAGES.findIndex((item) => item.key === stage);
}

function GenerationProgress({ stage, compact = false }: { stage: GenerationStage; compact?: boolean }) {
  const activeIndex = generationStageIndex(stage);
  return (
    <motion.section
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      className={`glass-panel rounded-2xl ${compact ? "p-4" : "mb-7 p-5 sm:p-6"}`}
      aria-live="polite"
      aria-label="Bid generation progress"
    >
      <div className="mb-4 flex items-center justify-between gap-4">
        <div>
          <p className="text-[10px] font-bold uppercase tracking-[0.22em] text-indigo-300">Orchestrator in flight</p>
          <p className="mt-1 text-sm text-zinc-300">Preparing a new grounded review session</p>
        </div>
        <span className="h-2 w-2 animate-pulse rounded-full bg-indigo-300 shadow-[0_0_14px_rgba(165,180,252,.9)]" />
      </div>
      <div className="grid gap-2 sm:grid-cols-3">
        {GENERATION_STAGES.map((item, index) => {
          const complete = index < activeIndex;
          const active = index === activeIndex;
          return (
            <motion.div
              key={item.key}
              initial={{ opacity: 0.35 }}
              animate={{ opacity: complete || active ? 1 : 0.42 }}
              className={`flex items-center gap-3 rounded-xl border px-3 py-3 text-xs transition ${active ? "border-indigo-300/35 bg-indigo-300/10 text-indigo-100" : complete ? "border-emerald-300/20 bg-emerald-300/[0.06] text-emerald-200" : "border-zinc-800 bg-zinc-950/30 text-zinc-500"}`}
            >
              <span className={`grid h-6 w-6 shrink-0 place-items-center rounded-full border font-mono text-[10px] ${active ? "border-indigo-200/50 bg-indigo-300/20 text-indigo-100" : complete ? "border-emerald-300/40 bg-emerald-300/10 text-emerald-200" : "border-zinc-700 text-zinc-600"}`}>
                {complete ? "✓" : `0${index + 1}`}
              </span>
              <span className="font-semibold">{item.label}</span>
              {active && <span className="ml-auto h-1.5 w-1.5 animate-ping rounded-full bg-indigo-200" />}
            </motion.div>
          );
        })}
      </div>
    </motion.section>
  );
}

function timelineState(status: BidStatus) {
  if (status === "awaiting_approval") return { activeIndex: 3, tone: "amber", note: "The generated bid is staged and waiting for your decision." };
  if (status === "submitting") return { activeIndex: 4, tone: "indigo", note: "Approval was recorded; the staged draft is being released." };
  if (status === "submitted") return { activeIndex: 5, tone: "emerald", note: "The mock portal confirmed the submitted bid." };
  return { activeIndex: 4, tone: "rose", note: "The workflow stopped at the decision boundary and needs manual review." };
}

function StatusTimeline({ status }: { status: BidStatus }) {
  const state = timelineState(status);
  const currentTone = {
    amber: "border-amber-300/50 bg-amber-300/15 text-amber-100",
    indigo: "border-indigo-300/50 bg-indigo-300/15 text-indigo-100",
    emerald: "border-emerald-300/50 bg-emerald-300/15 text-emerald-100",
    rose: "border-rose-300/50 bg-rose-300/15 text-rose-100",
  }[state.tone];

  return (
    <section className="glass-panel mb-8 rounded-3xl p-6 sm:p-8" aria-label="Bid status timeline">
      <div className="mb-7 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-[10px] font-bold uppercase tracking-[0.25em] text-zinc-600">Workflow / live status</p>
          <h2 className="mt-2 text-xl font-semibold tracking-tight text-zinc-100">From RFP to submission</h2>
        </div>
        <p className="text-xs text-zinc-500">{state.note}</p>
      </div>
      <ol className="relative">
        {TIMELINE_STEPS.map((step, index) => {
          const complete = index < state.activeIndex;
          const current = index === state.activeIndex;
          return (
            <li key={step.label} className="relative flex min-h-[58px] items-start gap-4 pb-5 last:min-h-0 last:pb-0">
              {index < TIMELINE_STEPS.length - 1 && <span className={`absolute bottom-0 left-[11px] top-6 w-px ${index < state.activeIndex ? "bg-emerald-300/45" : "bg-zinc-800"}`} aria-hidden="true" />}
              <span className={`relative z-10 grid h-6 w-6 shrink-0 place-items-center rounded-full border font-mono text-[10px] ${current ? currentTone : complete ? "border-emerald-300/40 bg-emerald-300/10 text-emerald-200" : "border-zinc-700 bg-zinc-950 text-zinc-600"}`}>
                {complete ? "✓" : `0${index + 1}`}
              </span>
              <span className="relative z-10">
                <span className={`block text-xs font-semibold ${current ? "text-zinc-100" : complete ? "text-zinc-300" : "text-zinc-600"}`}>{step.label}</span>
                <span className="mt-1 block text-[11px] text-zinc-600">{step.detail}</span>
              </span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

function EmptyState({ title, body, warning = false, onRetry, action }: { title: string; body: string; warning?: boolean; onRetry?: () => void; action?: React.ReactNode }) {
  return (
    <main className="grid min-h-screen place-items-center px-6 py-12">
      <motion.section initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} className="glass-panel max-w-lg rounded-3xl p-8 text-center sm:p-12">
        <div className={`mx-auto mb-6 grid h-14 w-14 place-items-center rounded-2xl border ${warning ? "border-rose-400/25 bg-rose-400/10 text-rose-300" : "border-indigo-400/25 bg-indigo-400/10 text-indigo-300"}`}>
          <Icon name={warning ? "warning" : "spark"} size={23} />
        </div>
        <p className="mb-3 text-[10px] font-bold uppercase tracking-[0.25em] text-indigo-300">Anakin / bid operations</p>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-100">{title}</h1>
        <p className="mt-3 text-sm leading-7 text-zinc-400">{body}</p>
        {onRetry && <button type="button" onClick={onRetry} className="mt-6 rounded-xl border border-indigo-300/30 px-4 py-2 text-sm font-semibold text-indigo-200 transition hover:bg-indigo-300/10">Retry connection</button>}
        {action && <div className="mt-6">{action}</div>}
      </motion.section>
    </main>
  );
}

function MagneticButton({ disabled, onClick, children, success }: { disabled: boolean; onClick: () => void; children: React.ReactNode; success: boolean }) {
  const buttonRef = useRef<HTMLButtonElement>(null);
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const springX = useSpring(x, { stiffness: 260, damping: 18, mass: 0.25 });
  const springY = useSpring(y, { stiffness: 260, damping: 18, mass: 0.25 });

  const handleMove = (event: PointerEvent<HTMLButtonElement>) => {
    if (disabled || !buttonRef.current) return;
    const rect = buttonRef.current.getBoundingClientRect();
    x.set((event.clientX - (rect.left + rect.width / 2)) * 0.13);
    y.set((event.clientY - (rect.top + rect.height / 2)) * 0.13);
  };

  const reset = () => {
    x.set(0);
    y.set(0);
  };

  return (
    <motion.button
      ref={buttonRef}
      type="button"
      onClick={onClick}
      onPointerMove={handleMove}
      onPointerLeave={reset}
      disabled={disabled}
      style={{ x: springX, y: springY }}
      whileTap={{ scale: 0.97 }}
      className={`group relative inline-flex min-h-[52px] min-w-[200px] items-center justify-center gap-3 overflow-hidden rounded-2xl px-5 py-3 text-sm font-bold transition disabled:cursor-not-allowed ${success ? "border border-emerald-300/40 bg-emerald-300 text-zinc-950 shadow-[0_0_35px_rgba(110,231,183,.25)]" : "bg-indigo-300 text-indigo-950 shadow-[0_0_35px_rgba(129,140,248,.22)] hover:bg-indigo-200"}`}
    >
      {!success && <span className="absolute inset-0 -translate-x-full bg-white/25 transition-transform duration-700 group-hover:translate-x-full" />}
      <span className="relative flex items-center gap-3">{children}</span>
    </motion.button>
  );
}

function GenerateNewBidButton({ disabled, onClick }: { disabled: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="inline-flex min-h-[46px] items-center justify-center gap-2 rounded-xl border border-indigo-300/30 bg-indigo-300/[0.08] px-4 py-2.5 text-sm font-semibold text-indigo-100 transition hover:border-indigo-200/50 hover:bg-indigo-300/[0.16] disabled:cursor-not-allowed disabled:opacity-60"
    >
      {disabled ? <span className="h-4 w-4 animate-spin rounded-full border-2 border-indigo-200/30 border-t-indigo-200" /> : <Icon name="spark" size={16} />}
      {disabled ? "Generating bid…" : "Generate new bid"}
    </button>
  );
}

type DiffPart = { value: string; kind: "same" | "removed" | "added" };

function diffParts(original: string, edited: string): DiffPart[] {
  if (original === edited) return [{ value: original, kind: "same" }];
  const originalTokens = original.match(/\s+|[^\s]+/g) ?? [];
  const editedTokens = edited.match(/\s+|[^\s]+/g) ?? [];
  let prefix = 0;
  while (prefix < originalTokens.length && prefix < editedTokens.length && originalTokens[prefix] === editedTokens[prefix]) prefix += 1;
  let suffix = 0;
  while (
    suffix < originalTokens.length - prefix &&
    suffix < editedTokens.length - prefix &&
    originalTokens[originalTokens.length - 1 - suffix] === editedTokens[editedTokens.length - 1 - suffix]
  ) suffix += 1;

  const parts: DiffPart[] = [];
  const pushPart = (value: string, kind: DiffPart["kind"]) => {
    if (!value) return;
    const previous = parts[parts.length - 1];
    if (previous?.kind === kind) previous.value += value;
    else parts.push({ value, kind });
  };
  pushPart(originalTokens.slice(0, prefix).join(""), "same");
  pushPart(originalTokens.slice(prefix, originalTokens.length - suffix).join(""), "removed");
  pushPart(editedTokens.slice(prefix, editedTokens.length - suffix).join(""), "added");
  pushPart(originalTokens.slice(originalTokens.length - suffix).join(""), "same");
  return parts;
}

function DiffHighlight({ original, edited }: { original: string; edited: string }) {
  const parts = diffParts(original, edited);
  return (
    <div className="mt-4 rounded-2xl border border-amber-300/20 bg-amber-300/[0.045] p-4 sm:p-5">
      <div className="mb-3 flex flex-wrap items-center gap-3 text-[10px] font-bold uppercase tracking-[0.2em] text-amber-200/80">
        <span className="flex items-center gap-2"><span className="h-1.5 w-1.5 rounded-full bg-amber-300" /> HITL edit preview</span>
        <span className="font-normal tracking-normal text-zinc-600"><span className="text-rose-300 line-through">removed</span> / <span className="text-emerald-300">added</span></span>
      </div>
      <p className="whitespace-pre-wrap text-sm leading-7 text-zinc-300">
        {parts.map((part, index) => (
          <span key={`${part.kind}-${index}`} className={part.kind === "removed" ? "rounded bg-rose-400/15 text-rose-200 line-through decoration-rose-300/70" : part.kind === "added" ? "rounded bg-emerald-400/15 text-emerald-100" : "text-zinc-400"}>{part.value}</span>
        ))}
      </p>
    </div>
  );
}

function SectionDiff({ item, index, editedAnswer, editable, onAnswerChange, onReset }: { item: ReviewSection; index: number; editedAnswer: string; editable: boolean; onAnswerChange: (value: string) => void; onReset: () => void }) {
  const [citationOpen, setCitationOpen] = useState(false);
  const supported = item.answer !== "CAPABILITY_NOT_FOUND";
  const hasEdit = editedAnswer !== item.answer;
  return (
    <motion.article
      layout
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.16 }}
      transition={{ duration: 0.55, delay: index * 0.08, ease: [0.22, 1, 0.36, 1] }}
      className="group overflow-hidden rounded-3xl border border-zinc-800/90 bg-zinc-900/55 shadow-2xl shadow-black/10 backdrop-blur-xl"
    >
      <div className="flex items-center justify-between border-b border-zinc-800/80 px-5 py-4 sm:px-7">
        <div className="flex items-center gap-3">
          <span className="font-mono text-[11px] text-zinc-600">0{index + 1}</span>
          <h2 className="text-sm font-semibold tracking-wide text-zinc-100">{item.section}</h2>
        </div>
        <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.18em]">
          <span className={`h-1.5 w-1.5 rounded-full ${supported ? "bg-emerald-300 shadow-[0_0_10px_rgba(110,231,183,.8)]" : "bg-zinc-600"}`} />
          <span className={supported ? "text-emerald-300/80" : "text-zinc-500"}>{supported ? "Grounded" : "No match"}</span>
        </div>
      </div>

      <div className="grid lg:grid-cols-2">
        <div className="relative border-b border-zinc-800/80 p-6 sm:p-8 lg:border-b-0 lg:border-r">
          <div className="mb-6 flex items-center justify-between">
            <p className="text-[10px] font-bold uppercase tracking-[0.22em] text-zinc-500">RFP / requirement</p>
            <span className="rounded-md border border-zinc-800 bg-zinc-950/70 px-2 py-1 font-mono text-[10px] text-zinc-600">INPUT</span>
          </div>
          <p className="max-w-xl text-[15px] leading-8 text-zinc-300">{item.requirement}</p>
          <div className="pointer-events-none absolute bottom-0 left-0 top-0 w-px bg-gradient-to-b from-transparent via-indigo-400/50 to-transparent opacity-0 transition-opacity duration-500 group-hover:opacity-100" />
        </div>

        <div className="relative bg-gradient-to-br from-indigo-400/[0.045] via-transparent to-emerald-400/[0.035] p-6 sm:p-8">
          <div className="mb-6 flex items-center justify-between">
            <p className="text-[10px] font-bold uppercase tracking-[0.22em] text-indigo-300/80">AI / proposal answer</p>
            <span className={`flex items-center gap-1.5 rounded-md border px-2 py-1 font-mono text-[10px] ${editable ? "border-amber-300/20 bg-amber-300/[0.06] text-amber-200/80" : "border-indigo-400/15 bg-indigo-400/[0.06] text-indigo-300/70"}`}><Icon name={editable ? "spark" : "lock"} size={11} /> {editable ? "EDITABLE IN REVIEW" : "LOCKED"}</span>
          </div>
          {editable ? (
            <textarea
              value={editedAnswer}
              onChange={(event) => onAnswerChange(event.target.value)}
              aria-label={`Edit ${item.section} proposal answer`}
              className={`min-h-32 w-full resize-y rounded-2xl border bg-zinc-950/40 p-4 text-[15px] leading-8 outline-none transition focus:border-indigo-300/50 focus:ring-2 focus:ring-indigo-300/10 ${hasEdit ? "border-amber-300/40 text-amber-50" : "border-zinc-800 text-zinc-100"}`}
            />
          ) : (
            <p className={`whitespace-pre-wrap text-[15px] leading-8 ${supported ? "text-zinc-100" : "font-mono text-zinc-500"}`}>{editedAnswer}</p>
          )}
          <AnimatePresence initial={false}>
            {hasEdit && (
              <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}>
                <DiffHighlight original={item.answer} edited={editedAnswer} />
                <button type="button" onClick={onReset} className="mt-3 text-xs font-semibold text-zinc-500 underline decoration-zinc-700 underline-offset-4 transition hover:text-zinc-200">Reset to AI answer</button>
              </motion.div>
            )}
          </AnimatePresence>
          {item.source_snippet && (
            <div className="mt-7 rounded-2xl border border-emerald-400/20 bg-emerald-400/[0.055] shadow-[inset_0_1px_0_rgba(110,231,183,.06)]">
              <button type="button" onClick={() => setCitationOpen((open) => !open)} aria-expanded={citationOpen} className="flex w-full items-center justify-between gap-4 p-4 text-left sm:p-5">
                <span className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.2em] text-emerald-300/80"><span className="h-1.5 w-1.5 rounded-full bg-emerald-300 shadow-[0_0_8px_rgba(110,231,183,.8)]" /> Grounded in capability docs</span>
                <span className={`text-emerald-200 transition-transform ${citationOpen ? "rotate-90" : ""}`}><Icon name="chevron" size={15} /></span>
              </button>
              <AnimatePresence initial={false}>
                {citationOpen && (
                  <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} className="overflow-hidden">
                    <blockquote className="mx-4 mb-4 border-l-2 border-emerald-300/50 pl-4 font-mono text-[12px] leading-7 text-emerald-100/80 sm:mx-5 sm:mb-5">{item.source_snippet}</blockquote>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}
        </div>
      </div>
    </motion.article>
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
  const [error, setError] = useState<string | null>(null);
  const initializedBidRef = useRef<string | null>(null);

  useEffect(() => {
    setSelectedBidId(urlBidId);
  }, [urlBidId]);

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
    setEditedAnswers(Object.fromEntries(bid.comparison.map((item) => [item.section, item.answer])));
    setApprovalReleased(bid.approved || bid.status === "submitting" || bid.status === "submitted");
  }, [bid]);

  const loadBid = useCallback(async (background = false) => {
    if (!bidId) return;
    if (background) setRefreshing(true);
    else setLoading(true);
    try {
      const response = await fetch(apiUrl(`/api/bids/${encodeURIComponent(bidId)}`), { cache: "no-store" });
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
      setSelectedBidId(snapshot.id);
      setBid(snapshot);
      setEditedAnswers(Object.fromEntries(snapshot.comparison.map((item) => [item.section, item.answer])));
      setApprovalReleased(false);
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
      setApprovalReleased(true);
    } catch (requestError) {
      setError(requestErrorMessage(requestError, "Approval"));
    } finally {
      setApproving(false);
    }
  };

  const currentStatus = useMemo(() => bid && statusCopy(bid.status), [bid]);
  const isApproved = approvalReleased || bid?.approved || bid?.status === "submitting" || bid?.status === "submitted";
  const hasEdits = Boolean(bid?.comparison.some((item) => editedAnswers[item.section] !== undefined && editedAnswers[item.section] !== item.answer));
  const canEdit = Boolean(bid && bid.status === "awaiting_approval" && !isApproved);

  if (!bidId) return <EmptyState title="No bid selected" body={error ?? "Start the orchestrator from the dashboard to create a grounded review session."} warning={Boolean(error)} action={generating ? <GenerationProgress stage={generationStage} compact /> : <GenerateNewBidButton disabled={generating} onClick={() => void generateBid()} />} />;
  if (loading && !bid) return <EmptyState title="Loading bid review" body="Fetching the staged response from the FastAPI backend…" />;
  if (!bid) return <EmptyState title="Could not load this bid" body={error ?? "The backend did not return a review snapshot."} warning onRetry={() => void loadBid()} />;

  return (
    <main className="relative min-h-screen overflow-hidden px-4 py-5 sm:px-6 lg:px-10 lg:py-8">
      <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
        <div className="absolute left-[7%] top-[-14rem] h-[32rem] w-[32rem] rounded-full bg-indigo-600/10 blur-[130px]" />
        <div className="absolute bottom-[-16rem] right-[5%] h-[32rem] w-[32rem] rounded-full bg-emerald-500/[0.07] blur-[140px]" />
      </div>

      <div className="relative z-10 mx-auto max-w-7xl">
        <header className="mb-10 flex flex-col gap-7 border-b border-zinc-800/70 pb-7 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className="mb-7 flex items-center gap-3 text-[11px] font-bold uppercase tracking-[0.24em] text-zinc-400">
              <span className="grid h-8 w-8 place-items-center rounded-xl border border-indigo-300/30 bg-indigo-300/10 text-sm text-indigo-200 shadow-[0_0_25px_rgba(129,140,248,.16)]">A</span>
              <span className="text-zinc-600">/</span>
              <span>Bid operations desk</span>
            </div>
            <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
              <p className="mb-3 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.25em] text-indigo-300"><Icon name="spark" size={14} /> Grounded response review</p>
              <h1 className="max-w-3xl text-4xl font-semibold tracking-[-0.055em] text-zinc-50 sm:text-5xl lg:text-6xl">Approve with<br /><span className="text-gradient">evidence.</span></h1>
              <p className="mt-5 max-w-2xl text-sm leading-7 text-zinc-400 sm:text-base">A human checkpoint for a proposal assembled from your capability library. Inspect every requirement, answer, and source before release.</p>
            </motion.div>
          </div>
          <div className="flex flex-col items-start gap-3 lg:items-end">
            <GenerateNewBidButton disabled={generating} onClick={() => void generateBid()} />
            <StatusBadge status={bid.status} refreshing={refreshing} />
            <p className="text-xs text-zinc-600">Live sync <span className="mx-1 text-zinc-700">·</span> every {POLL_INTERVAL_MS / 1000}s</p>
          </div>
        </header>

        <AnimatePresence initial={false}>
          {error && (
            <motion.div initial={{ opacity: 0, height: 0, y: -8 }} animate={{ opacity: 1, height: "auto", y: 0 }} exit={{ opacity: 0, height: 0 }} role="alert" className="mb-7 flex items-start justify-between gap-4 overflow-hidden rounded-2xl border border-rose-400/25 bg-rose-400/[0.08] px-4 py-3 text-sm text-rose-200">
              <span className="flex items-center gap-2"><Icon name="warning" size={16} /> {error}</span>
              <button className="shrink-0 font-semibold text-rose-100 underline decoration-rose-300/40 underline-offset-4" onClick={() => void loadBid()}>Retry</button>
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence initial={false}>
          {generating && <GenerationProgress stage={generationStage} />}
        </AnimatePresence>

        <StatusTimeline status={bid.status} />

        <motion.section initial="hidden" animate="show" variants={{ hidden: {}, show: { transition: { staggerChildren: 0.08 } } }} className="mb-12 grid gap-3 sm:grid-cols-3">
          {[
            ["Review sections", String(bid.comparison.length), "RFP coverage"],
            ["Proposal state", isApproved ? "Released" : "Staged", isApproved ? "Approval gate open" : "Human gate locked"],
            ["Session", bid.id.slice(0, 12), "Live workspace ID"],
          ].map(([label, value, detail]) => (
            <motion.div key={label} variants={{ hidden: { opacity: 0, y: 12 }, show: { opacity: 1, y: 0 } }} className="glass-panel rounded-2xl px-5 py-4">
              <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-zinc-600">{label}</p>
              <p className="mt-2 truncate font-mono text-xl font-semibold tracking-tight text-zinc-100">{value}</p>
              <p className="mt-1 text-xs text-zinc-500">{detail}</p>
            </motion.div>
          ))}
        </motion.section>

        <div className="mb-5 flex items-end justify-between gap-4">
          <div>
            <p className="text-[10px] font-bold uppercase tracking-[0.25em] text-zinc-600">01 / proposal diff</p>
            <h2 className="mt-2 text-xl font-semibold tracking-tight text-zinc-100">Requirement by requirement</h2>
          </div>
          <span className="hidden items-center gap-2 text-xs text-zinc-600 sm:flex"><span className="h-1.5 w-1.5 rounded-full bg-emerald-300" /> Source-linked answers</span>
        </div>

        <div className="space-y-5">
          {bid.comparison.map((item, index) => (
            <SectionDiff
              key={`${bid.id}-${item.section}`}
              item={item}
              index={index}
              editedAnswer={editedAnswers[item.section] ?? item.answer}
              editable={canEdit}
              onAnswerChange={(value) => setEditedAnswers((current) => ({ ...current, [item.section]: value }))}
              onReset={() => setEditedAnswers((current) => ({ ...current, [item.section]: item.answer }))}
            />
          ))}
        </div>

        <motion.section initial={{ opacity: 0, y: 22 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="glass-panel mt-8 overflow-hidden rounded-3xl">
          <div className="flex flex-col gap-7 p-6 sm:p-8 lg:flex-row lg:items-center lg:justify-between">
            <div className="max-w-xl">
              <div className="mb-4 flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.22em] text-emerald-300/80"><span className="grid h-5 w-5 place-items-center rounded-md border border-emerald-300/20 bg-emerald-300/10"><Icon name="lock" size={11} /></span> Human approval gate</div>
              <h2 className="text-2xl font-semibold tracking-[-0.03em] text-zinc-100">{isApproved ? "Release acknowledged." : "Ready to release this bid?"}</h2>
              <p className="mt-2 text-sm leading-7 text-zinc-400">{isApproved ? currentStatus?.detail : hasEdits ? "Your edits are highlighted for review. Reset them before releasing the immutable staged draft." : "Your approval unlocks the staged submission. This action cannot be triggered by the agent."}</p>
              {bid.error && <p className="mt-3 text-sm text-rose-300">{bid.error}</p>}
            </div>
            <div className="shrink-0">
              <MagneticButton disabled={approving || bid.status !== "awaiting_approval" || hasEdits} onClick={() => void approveBid()} success={Boolean(isApproved)}>
                <AnimatePresence mode="wait" initial={false}>
                  {approving ? (
                    <motion.span key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex items-center gap-3"><span className="h-4 w-4 animate-spin rounded-full border-2 border-indigo-950/30 border-t-indigo-950" /> Releasing approval…</motion.span>
                  ) : isApproved ? (
                    <motion.span key="success" initial={{ opacity: 0, scale: 0.85 }} animate={{ opacity: 1, scale: 1 }} className="flex items-center gap-2"><Icon name="check" size={18} /> Approval released</motion.span>
                  ) : hasEdits ? (
                    <motion.span key="edited" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-center gap-3">Reset edits to approve <Icon name="warning" size={17} /></motion.span>
                  ) : (
                    <motion.span key="ready" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-center gap-3">Approve &amp; submit <Icon name="arrow" size={17} /></motion.span>
                  )}
                </AnimatePresence>
              </MagneticButton>
            </div>
          </div>
          <div className="flex flex-col gap-3 border-t border-zinc-800/80 px-6 py-4 text-xs text-zinc-600 sm:flex-row sm:items-center sm:justify-between sm:px-8">
            <span>Review data is refreshed automatically while this page is open.</span>
            <a className="group inline-flex items-center gap-2 text-zinc-400 transition hover:text-indigo-200" href={bid.review_url} target="_blank" rel="noreferrer">Open staged portal review <Icon name="external" size={13} /></a>
          </div>
        </motion.section>

        <footer className="flex flex-col gap-2 py-8 text-[10px] font-semibold uppercase tracking-[0.18em] text-zinc-700 sm:flex-row sm:items-center sm:justify-between">
          <span>Anakin / grounded automation</span>
          <span>Session {bid.id}</span>
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
