"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  AlertTriangle,
  ArrowLeft,
  BarChart3,
  Database,
  FileUp,
  Layers3,
  LogOut,
  Plus,
  Shuffle,
  ShieldCheck,
  Target,
  TimerReset,
  TrendingUp,
  UploadCloud,
  Wallet,
  Clock3,
} from "lucide-react";

type TradeSource = "TradingView" | "NinjaTrader";
type TabKey = "overview" | "tracker";
type Outcome = "Pass" | "Fail" | "Active";
type TradeRow = { timestamp: string; pnl: number; tradeId: string; dayId: string; source: TradeSource };
type DayBucket = { dayId: string; trades: TradeRow[] };
type TrackerEntry = {
  id: string;
  firm: string;
  accountLabel: string;
  attemptedAt: string;
  openedAt: string;
  outcome: Outcome;
  evalCost: number;
  activationFee: number;
  notes: string;
  strategyVersion: string;
  source: TradeSource;
};
type Rules = {
  profitTarget: number;
  drawdown: number;
  evalCost: number;
  activationFee: number;
  fundedAccountValue: number;
  oneDayPass: boolean;
};

const STORAGE_KEY = "passplan-tracker-v1";
const DEFAULT_RULES: Rules = { profitTarget: 250, drawdown: 180, evalCost: 149, activationFee: 99, fundedAccountValue: 1500, oneDayPass: true };
const SAMPLE_TRADES: TradeRow[] = [
  { timestamp: "2026-03-24T18:14:00-04:00", pnl: 112, tradeId: "TV-001", dayId: "2026-03-24", source: "TradingView" },
  { timestamp: "2026-03-24T19:11:00-04:00", pnl: 138, tradeId: "TV-002", dayId: "2026-03-24", source: "TradingView" },
  { timestamp: "2026-03-24T20:03:00-04:00", pnl: -74, tradeId: "TV-003", dayId: "2026-03-24", source: "TradingView" },
  { timestamp: "2026-03-25T18:12:00-04:00", pnl: 106, tradeId: "TV-004", dayId: "2026-03-25", source: "TradingView" },
  { timestamp: "2026-03-25T18:44:00-04:00", pnl: 129, tradeId: "TV-005", dayId: "2026-03-25", source: "TradingView" },
  { timestamp: "2026-03-25T19:20:00-04:00", pnl: 42, tradeId: "TV-006", dayId: "2026-03-25", source: "TradingView" },
  { timestamp: "2026-03-25T20:12:00-04:00", pnl: -83, tradeId: "TV-007", dayId: "2026-03-25", source: "TradingView" },
  { timestamp: "2026-03-26T18:08:00-04:00", pnl: 96, tradeId: "TV-011", dayId: "2026-03-26", source: "TradingView" },
  { timestamp: "2026-03-26T18:29:00-04:00", pnl: 114, tradeId: "TV-012", dayId: "2026-03-26", source: "TradingView" },
  { timestamp: "2026-03-26T19:14:00-04:00", pnl: 73, tradeId: "TV-013", dayId: "2026-03-26", source: "TradingView" },
  { timestamp: "2026-03-26T19:52:00-04:00", pnl: -61, tradeId: "TV-014", dayId: "2026-03-26", source: "TradingView" },
];
const SAMPLE_TRACKER: TrackerEntry[] = [
  { id: "attempt-1", firm: "Topstep", accountLabel: "50K eval #1", attemptedAt: "2026-03-24T18:00", openedAt: "2026-03-24T17:40", outcome: "Pass", evalCost: 149, activationFee: 99, notes: "Two clustered wins before the first loss.", strategyVersion: "PassPlan v1", source: "TradingView" },
  { id: "attempt-2", firm: "Apex", accountLabel: "75K eval #2", attemptedAt: "2026-03-25T18:00", openedAt: "2026-03-25T17:55", outcome: "Fail", evalCost: 167, activationFee: 0, notes: "Stopped after the first red trade.", strategyVersion: "PassPlan v1", source: "NinjaTrader" },
];

function makeId(prefix: string) { return `${prefix}-${Math.random().toString(36).slice(2, 10)}`; }
function money(value: number) { return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(value); }
function pct(value: number) { return `${value.toFixed(1)}%`; }
function n(value: number, digits = 1) { return new Intl.NumberFormat("en-US", { maximumFractionDigits: digits }).format(value); }
function localDateTime(date = new Date()) { const offset = date.getTimezoneOffset() * 60000; return new Date(date.getTime() - offset).toISOString().slice(0, 16); }
function sessionDayId(timestamp: string) {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "";
  const parts = new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit" }).formatToParts(date);
  const map = Object.fromEntries(parts.map((part) => [part.type, part.value])) as Record<string, string>;
  const bucket = new Date(Date.UTC(Number(map.year), Number(map.month) - 1, Number(map.day), 12));
  if (Number(map.hour) < 18) bucket.setUTCDate(bucket.getUTCDate() - 1);
  return bucket.toISOString().slice(0, 10);
}
function parseCsv(text: string) {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];
    if (ch === '"') { if (quoted && next === '"') { cell += '"'; i += 1; } else { quoted = !quoted; } continue; }
    if (!quoted && (ch === "," || ch === "\t")) { row.push(cell.trim()); cell = ""; continue; }
    if (!quoted && (ch === "\n" || ch === "\r")) { if (ch === "\r" && next === "\n") i += 1; row.push(cell.trim()); if (row.some((v) => v.length > 0)) rows.push(row); row = []; cell = ""; continue; }
    cell += ch;
  }
  row.push(cell.trim());
  if (row.some((v) => v.length > 0)) rows.push(row);
  return rows;
}
function parseNumber(value: string) { const cleaned = value.replace(/[$,]/g, "").replace(/\((.*)\)/, "-$1").trim(); const parsed = Number(cleaned); return Number.isFinite(parsed) ? parsed : 0; }
function findColumn(headers: string[], keys: string[]) { return headers.findIndex((header) => { const value = header.toLowerCase().trim(); return keys.some((key) => value === key || value.includes(key)); }); }
function normalizeTrades(rows: string[][], source: TradeSource) {
  if (rows.length < 2) return { trades: [] as TradeRow[], warnings: ["CSV did not contain enough rows."] };
  const headers = rows[0];
  const tsIndex = findColumn(headers, ["timestamp", "time", "date", "datetime", "fill time"]);
  const pnlIndex = findColumn(headers, ["pnl", "profit/loss", "profit", "net profit", "realized pnl", "p/l", "pl"]);
  const idIndex = findColumn(headers, ["trade id", "tradeid", "id", "order id", "execution id"]);
  const warnings: string[] = [];
  if (tsIndex < 0) warnings.push("Timestamp column was not found.");
  if (pnlIndex < 0) warnings.push("PnL column was not found.");
  const trades: TradeRow[] = [];
  rows.slice(1).forEach((row, index) => {
    const timestamp = tsIndex >= 0 ? row[tsIndex] ?? "" : "";
    const dayId = timestamp ? sessionDayId(timestamp) : "";
    const pnl = pnlIndex >= 0 ? parseNumber(row[pnlIndex] ?? "") : 0;
    if (!timestamp || !dayId) return;
    trades.push({ timestamp, pnl, tradeId: idIndex >= 0 ? row[idIndex] ?? `${source}-${index + 1}` : `${source}-${index + 1}`, dayId, source });
  });
  if (!trades.length) warnings.push("No trades were normalized from the selected CSV.");
  return { trades, warnings };
}
function sortTrades(trades: TradeRow[]) { return [...trades].sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()); }
function groupByDay(trades: TradeRow[]) {
  const map = new Map<string, TradeRow[]>();
  sortTrades(trades).forEach((trade) => { if (!map.has(trade.dayId)) map.set(trade.dayId, []);
    map.get(trade.dayId)?.push(trade);
  });
  return Array.from(map.entries()).map(([dayId, dayTrades]) => ({ dayId, trades: dayTrades }));
}
function simulateDay(trades: TradeRow[], rules: Rules) {
  let pnl = 0, passes = 0, firstPassAt: number | null = null, tradeCount = 0;
  for (const trade of sortTrades(trades)) {
    pnl += trade.pnl; tradeCount += 1;
    if (pnl >= rules.profitTarget) { passes += 1; firstPassAt ??= tradeCount; if (!rules.oneDayPass) break; pnl = 0; tradeCount = 0; continue; }
    if (trade.pnl < 0 || pnl <= -rules.drawdown) break;
  }
  return { passes, firstPassAt };
}
function analyze(dayBuckets: DayBucket[], rules: Rules) {
  const perDay = dayBuckets.map((day) => {
    let winsBeforeLoss = 0, pnlBeforeLoss = 0;
    for (const trade of sortTrades(day.trades)) { if (trade.pnl < 0) break; if (trade.pnl > 0) winsBeforeLoss += 1; pnlBeforeLoss += trade.pnl; }
    const sim = simulateDay(day.trades, rules);
    return { dayId: day.dayId, tradeCount: day.trades.length, winsBeforeLoss, pnlBeforeLoss, passes: sim.passes, firstPassAt: sim.firstPassAt };
  });
  const totalDays = Math.max(perDay.length, 1);
  const avgWins = perDay.reduce((sum, row) => sum + row.winsBeforeLoss, 0) / totalDays;
  const avgBeforeLoss = perDay.reduce((sum, row) => sum + row.pnlBeforeLoss, 0) / totalDays;
  const avgPasses = perDay.reduce((sum, row) => sum + row.passes, 0) / totalDays;
  const passRate = perDay.filter((row) => row.passes > 0).length / totalDays;
  const zeroPassRate = 1 - passRate;
  const thresholds = [1, 2, 3].map((threshold) => ({ threshold, rate: perDay.filter((row) => row.winsBeforeLoss >= threshold).length / totalDays }));
  const winDistribution = [0, 1, 2, 3, 4].map((bucket) => ({ label: bucket < 4 ? String(bucket) : "4+", value: perDay.filter((row) => (bucket < 4 ? row.winsBeforeLoss === bucket : row.winsBeforeLoss >= 4)).length }));
  const passDistribution = [0, 1, 2, 3, 4].map((bucket) => ({ label: bucket < 4 ? String(bucket) : "4+", value: perDay.filter((row) => (bucket < 4 ? row.passes === bucket : row.passes >= 4)).length }));
  return { perDay, avgWins, avgBeforeLoss, avgPasses, passRate, zeroPassRate, thresholds, winDistribution, passDistribution };
}
function lcg(seed: number) { let value = seed >>> 0; return () => { value = (Math.imul(1664525, value) + 1013904223) >>> 0; return value / 0x100000000; }; }
function monteCarlo(dayBuckets: DayBucket[], rules: Rules, runs = 2000, horizon = 20) {
  if (!dayBuckets.length) return { zeroPassRate: 0, passRate: 0, expectedPasses: 0, costPerFundedAccount: 0, netEV: 0, cumulative: Array.from({ length: horizon }, () => 0), distribution: [] as Array<{ label: string; value: number }> };
  const rand = lcg(918273), dist = new Map<number, number>(), cumulative = Array.from({ length: horizon }, () => 0);
  let totalPasses = 0, zeroPassDays = 0, totalDays = 0;
  for (let run = 0; run < runs; run += 1) {
    let passCount = 0;
    for (let day = 0; day < horizon; day += 1) {
      const bucket = dayBuckets[Math.floor(rand() * dayBuckets.length)];
      const result = simulateDay(bucket.trades, rules);
      passCount += result.passes; totalPasses += result.passes; totalDays += 1;
      if (result.passes === 0) zeroPassDays += 1;
      dist.set(result.passes, (dist.get(result.passes) ?? 0) + 1);
      cumulative[day] += passCount;
    }
  }
  const expectedPasses = totalPasses / totalDays;
  const passRate = 1 - zeroPassDays / totalDays;
  const zeroPassRate = 1 - passRate;
  const costPerFundedAccount = expectedPasses > 0 ? rules.evalCost / expectedPasses + rules.activationFee : rules.evalCost + rules.activationFee;
  const netEV = expectedPasses * rules.fundedAccountValue - rules.evalCost - expectedPasses * rules.activationFee;
  const distribution = Array.from(dist.entries()).sort((a, b) => a[0] - b[0]).slice(0, 5).map(([key, value]) => ({ label: key >= 4 ? "4+" : String(key), value: Math.round((value / totalDays) * 1000) / 10 }));
  return { zeroPassRate, passRate, expectedPasses, costPerFundedAccount, netEV, cumulative: cumulative.map((value) => value / runs), distribution };
}
function outcomeClass(outcome: Outcome) { if (outcome === "Pass") return "border-emerald-400/20 bg-emerald-400/10 text-emerald-200"; if (outcome === "Fail") return "border-rose-400/20 bg-rose-400/10 text-rose-200"; return "border-white/10 bg-white/[0.03] text-white/60"; }
function nowLabel() { return localDateTime(); }

export default function PassPlanPage() {
  const router = useRouter();
  const [tab, setTab] = useState<TabKey>("overview");
  const [source, setSource] = useState<TradeSource>("TradingView");
  const [uploadName, setUploadName] = useState("sample-sequence.csv");
  const [warnings, setWarnings] = useState<string[]>([]);
  const [trades, setTrades] = useState<TradeRow[]>(SAMPLE_TRADES);
  const [rules, setRules] = useState(DEFAULT_RULES);
  const [tracker, setTracker] = useState<TrackerEntry[]>(SAMPLE_TRACKER);
  const [hydrated, setHydrated] = useState(false);
  const [draft, setDraft] = useState<Omit<TrackerEntry, "id">>({ firm: "Topstep", accountLabel: "50K eval", attemptedAt: nowLabel(), openedAt: nowLabel(), outcome: "Active", evalCost: 149, activationFee: 99, notes: "", strategyVersion: "PassPlan v1", source: "TradingView" });

  useEffect(() => {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try { setTracker(JSON.parse(stored) as TrackerEntry[]); } catch { /* ignore */ }
    }
    setHydrated(true);
  }, []);
  useEffect(() => { if (hydrated) window.localStorage.setItem(STORAGE_KEY, JSON.stringify(tracker)); }, [tracker, hydrated]);

  const dayBuckets = useMemo(() => groupByDay(trades), [trades]);
  const stats = useMemo(() => analyze(dayBuckets, rules), [dayBuckets, rules]);
  const mc = useMemo(() => monteCarlo(dayBuckets, rules), [dayBuckets, rules]);
  const totalTrackerFees = tracker.reduce((sum, row) => sum + row.evalCost + row.activationFee, 0);
  const passTracker = tracker.filter((row) => row.outcome === "Pass").length;
  const cards = [
    { label: "Pass probability", value: pct(mc.passRate * 100), detail: `${dayBuckets.length} session buckets`, icon: ShieldCheck },
    { label: "0-pass risk", value: pct(mc.zeroPassRate * 100), detail: "Sessions that never stack", icon: AlertTriangle },
    { label: "Expected passes/day", value: n(mc.expectedPasses, 2), detail: "Sequential same-day switching", icon: Shuffle },
    { label: "Cost per funded account", value: money(mc.costPerFundedAccount), detail: "Eval fee plus activation drag", icon: Wallet },
    { label: "Net EV / day", value: money(mc.netEV), detail: "Uses estimated funded value", icon: TrendingUp },
    { label: "Tracker wins", value: `${passTracker}/${tracker.length}`, detail: `${money(totalTrackerFees)} in logged fees`, icon: Database },
  ];
  const ruleInputs: Array<{ key: keyof Rules; label: string; value: number }> = [
    { key: "profitTarget", label: "Profit target", value: rules.profitTarget },
    { key: "drawdown", label: "Drawdown", value: rules.drawdown },
    { key: "evalCost", label: "Eval cost", value: rules.evalCost },
    { key: "activationFee", label: "Activation fee", value: rules.activationFee },
    { key: "fundedAccountValue", label: "Funded account value", value: rules.fundedAccountValue },
  ];

  async function onUpload(file: File | null) {
    if (!file) return;
    const normalized = normalizeTrades(parseCsv(await file.text()), source);
    setWarnings(normalized.warnings);
    if (normalized.trades.length) { setTrades(normalized.trades); setUploadName(file.name); }
  }
  function addTrackerEntry() { setTracker((current) => [{ id: makeId("attempt"), ...draft }, ...current]); setDraft((current) => ({ ...current, notes: "", attemptedAt: nowLabel() })); }
  function resetSample() { setTrades(SAMPLE_TRADES); setRules(DEFAULT_RULES); setWarnings([]); setUploadName("sample-sequence.csv"); }
  function logout() { fetch("/api/logout", { method: "POST" }).finally(() => router.push("/login")); }

  return (
    <div className="min-h-screen bg-[#08090c] text-white flex">
      <aside className="hidden lg:flex w-[250px] flex-col border-r border-white/6 bg-[#0c0d11]">
        <div className="h-16 px-5 flex items-center gap-3 border-b border-white/6">
          <div className="h-9 w-9 rounded-2xl bg-gradient-to-br from-[#FF7A18] to-[#FFB347] flex items-center justify-center">
            <BarChart3 className="h-4 w-4 text-white" />
          </div>
          <div>
            <div className="text-sm font-bold tracking-tight">Pass<span className="text-gradient">Plan</span></div>
            <div className="text-[10px] uppercase tracking-[0.22em] text-white/25">Account acquisition engine</div>
          </div>
        </div>
        <div className="p-4 space-y-4 flex-1">
          <div className="rounded-2xl border border-white/6 bg-[#111217] p-4">
            <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.22em] text-white/28">
              <Clock3 className="h-3.5 w-3.5" />
              Session boundary
            </div>
            <div className="mt-3 text-sm font-semibold">6:00 PM ET to 5:00 PM ET</div>
            <div className="mt-2 text-xs text-white/40 leading-relaxed">Day grouping follows the CME futures session so the first-loss stop and same-day switch logic stay path dependent.</div>
          </div>
          <button type="button" onClick={() => setTab("overview")} className={`w-full flex items-center gap-3 rounded-xl px-4 py-3 text-sm transition-all ${tab === "overview" ? "bg-[#17191f] border border-[#FF7A18]/25 text-white" : "border border-transparent text-white/45 hover:text-white hover:bg-white/[0.03]"}`}><Layers3 className="h-4 w-4 text-[#FFB347]" />Overview</button>
          <button type="button" onClick={() => setTab("tracker")} className={`w-full flex items-center gap-3 rounded-xl px-4 py-3 text-sm transition-all ${tab === "tracker" ? "bg-[#17191f] border border-[#FF7A18]/25 text-white" : "border border-transparent text-white/45 hover:text-white hover:bg-white/[0.03]"}`}><Database className="h-4 w-4 text-[#FFB347]" />Tracker</button>
          <div className="rounded-2xl border border-white/6 bg-[#111217] p-4 space-y-3">
            <Link href="/simulator" className="flex items-center gap-2 text-sm text-white/45 hover:text-white transition-colors"><ArrowLeft className="h-4 w-4" />Back to ProfitPlan</Link>
            <Link href="/discover" className="flex items-center gap-2 text-sm text-white/45 hover:text-white transition-colors"><Target className="h-4 w-4" />Discover</Link>
          </div>
        </div>
        <div className="p-4 border-t border-white/6">
          <button type="button" onClick={logout} className="w-full rounded-xl border border-white/8 bg-white/[0.03] px-4 py-3 text-sm text-white/60 hover:text-white hover:border-white/15 flex items-center justify-center gap-2">
            <LogOut className="h-4 w-4" />
            Log out
          </button>
        </div>
      </aside>

      <div className="flex-1 min-w-0 flex flex-col">
        <header className="h-16 border-b border-white/6 bg-[#0b0c10]/95 backdrop-blur-xl flex items-center justify-between px-5 sm:px-7">
          <div>
            <div className="flex items-center gap-2.5">
              <h1 className="text-lg font-semibold tracking-tight">PassPlan</h1>
              <span className="rounded-full border border-[#FF7A18]/25 bg-[#FF7A18]/10 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-[#FFB347]">Acquisition engine</span>
            </div>
            <p className="text-xs text-white/35 mt-1">Evaluate eval stacking, clustered wins, and real-world tracking.</p>
          </div>
          <div className="hidden sm:flex items-center gap-2 rounded-full border border-[#FF7A18]/20 bg-[#FF7A18]/8 px-3 py-2 text-xs text-[#FFB347]">
            <Clock3 className="h-3.5 w-3.5" />
            6 PM ET session day
          </div>
        </header>

        <main className="flex-1 overflow-y-auto">
          <div className="max-w-7xl mx-auto px-5 sm:px-7 py-6 space-y-6">
            <section className="relative overflow-hidden rounded-3xl border border-white/8 bg-[radial-gradient(circle_at_top_left,_rgba(255,122,24,0.16),_transparent_28%),linear-gradient(180deg,_rgba(18,19,24,0.98),_rgba(11,12,16,0.98))] p-6 sm:p-8">
              <div className="absolute -right-20 -top-20 h-56 w-56 rounded-full bg-[#FF7A18]/10 blur-3xl" />
              <div className="relative grid gap-6 lg:grid-cols-[1.4fr_0.9fr]">
                <div className="space-y-5">
                  <div className="inline-flex items-center gap-2 rounded-full border border-[#FF7A18]/20 bg-[#FF7A18]/10 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.2em] text-[#FFB347]"><ShieldCheck className="h-3.5 w-3.5" />Account acquisition engine</div>
                  <div>
                    <h2 className="text-3xl sm:text-5xl font-bold tracking-tight leading-tight">Model how clustered wins <span className="text-gradient">stack prop evals</span></h2>
                    <p className="mt-4 max-w-2xl text-sm sm:text-base text-white/45 leading-relaxed">PassPlan focuses on path dependence, first-loss stopping, same-day switching, and real-world validation. It is built for disposable evals, not normal backtesting.</p>
                  </div>
                  <div className="flex flex-wrap gap-3">
                    <button type="button" onClick={() => setTab("overview")} className="rounded-xl bg-gradient-to-r from-[#FF7A18] to-[#FFB347] px-5 py-3 text-sm font-semibold text-white">Inspect engine</button>
                    <button type="button" onClick={() => setTab("tracker")} className="rounded-xl border border-white/8 bg-white/[0.03] px-5 py-3 text-sm font-semibold text-white/75 hover:text-white">Open tracker</button>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-2xl border border-white/8 bg-[#0f1015]/90 p-4"><div className="flex items-center justify-between"><span className="text-[11px] uppercase tracking-[0.2em] text-white/25">Session rule</span><TimerReset className="h-4 w-4 text-[#FFB347]" /></div><div className="mt-4 text-2xl font-bold">First loss stops</div><div className="mt-2 text-xs leading-relaxed text-white/40">A negative trade ends the session unless the account already passed.</div></div>
                  <div className="rounded-2xl border border-white/8 bg-[#0f1015]/90 p-4"><div className="flex items-center justify-between"><span className="text-[11px] uppercase tracking-[0.2em] text-white/25">Stack mode</span><Shuffle className="h-4 w-4 text-[#FFB347]" /></div><div className="mt-4 text-2xl font-bold">{rules.oneDayPass ? "Same-day on" : "Same-day off"}</div><div className="mt-2 text-xs leading-relaxed text-white/40">{rules.oneDayPass ? "A pass can unlock the next account inside the same day." : "The engine stops after the first pass for a conservative view."}</div></div>
                </div>
              </div>
            </section>

            <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {cards.map(({ label, value, detail, icon: Icon }) => (
                <div key={label} className="rounded-2xl border border-white/8 bg-[#101116] p-5">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-[11px] uppercase tracking-[0.22em] text-white/25">{label}</div>
                      <div className="mt-3 text-3xl font-bold tracking-tight tabular-nums">{value}</div>
                    </div>
                    <div className="h-11 w-11 rounded-2xl bg-[#FF7A18]/10 border border-[#FF7A18]/15 flex items-center justify-center"><Icon className="h-5 w-5 text-[#FFB347]" /></div>
                  </div>
                  <div className="mt-3 text-sm text-white/42 leading-relaxed">{detail}</div>
                </div>
              ))}
            </section>

            {tab === "overview" ? (
              <section className="grid gap-6 xl:grid-cols-[1.25fr_0.75fr]">
                <div className="rounded-3xl border border-white/8 bg-[#101116] p-5 sm:p-6">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <div className="text-[11px] uppercase tracking-[0.22em] text-white/28">Import layer</div>
                      <h3 className="mt-2 text-xl font-semibold">Normalize your trade CSVs</h3>
                    </div>
                    <button type="button" onClick={resetSample} className="rounded-xl border border-white/8 bg-white/[0.03] px-4 py-2 text-sm text-white/65 hover:text-white">Reset sample</button>
                  </div>
                  <div className="mt-5 grid gap-4 lg:grid-cols-[1.15fr_0.85fr]">
                    <div className="rounded-2xl border border-dashed border-white/10 bg-black/15 p-5">
                      <div className="flex flex-wrap items-center gap-3">
                        <label className="text-xs uppercase tracking-[0.18em] text-white/30">Source</label>
                        <select value={source} onChange={(e) => setSource(e.target.value as TradeSource)} className="rounded-lg border border-white/10 bg-[#0c0d11] px-3 py-2 text-sm text-white outline-none">
                          <option value="TradingView">TradingView</option>
                          <option value="NinjaTrader">NinjaTrader</option>
                        </select>
                        <div className="text-xs text-white/30">Supports timestamp, pnl, and trade id columns.</div>
                      </div>
                      <label className="mt-5 flex cursor-pointer flex-col items-center justify-center rounded-2xl border border-white/8 bg-[#0e0f13] px-6 py-8 text-center hover:border-[#FF7A18]/30 hover:bg-[#11131a]">
                        <input type="file" accept=".csv,text/csv" className="hidden" onChange={(e) => onUpload(e.target.files?.[0] ?? null)} />
                        <UploadCloud className="h-8 w-8 text-[#FFB347]" />
                        <div className="mt-3 text-base font-semibold">Upload a trade export</div>
                        <div className="mt-1 text-sm text-white/40">TradingView or NinjaTrader CSV</div>
                        <div className="mt-3 inline-flex items-center gap-2 rounded-full border border-white/8 bg-white/[0.03] px-3 py-1.5 text-xs text-white/45"><FileUp className="h-3.5 w-3.5" />{uploadName}</div>
                      </label>
                      {warnings.length > 0 && <div className="mt-4 space-y-2">{warnings.map((warning) => <div key={warning} className="rounded-xl border border-amber-400/20 bg-amber-400/10 px-4 py-3 text-sm text-amber-100">{warning}</div>)}</div>}
                    </div>
                    <div className="rounded-2xl border border-white/8 bg-black/20 p-5">
                      <div className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-white/30"><Target className="h-3.5 w-3.5" />Prop rule config</div>
                      <div className="mt-4 space-y-4">
                        {ruleInputs.map((item) => (
                          <label key={item.key} className="block">
                            <div className="mb-2 text-xs uppercase tracking-[0.16em] text-white/28">{item.label}</div>
                            <input
                              type="number"
                              value={item.value}
                              onChange={(e) =>
                                setRules((current) => ({
                                  ...current,
                                  [item.key]: Number(e.target.value),
                                }))
                              }
                              className="w-full rounded-xl border border-white/10 bg-[#0c0d11] px-4 py-3 text-white outline-none focus:border-[#FF7A18]/40"
                            />
                          </label>
                        ))}
                        <label className="flex items-center justify-between rounded-xl border border-white/8 bg-white/[0.03] px-4 py-3">
                          <span className="text-sm text-white/70">One-day pass allowed</span>
                          <input type="checkbox" checked={rules.oneDayPass} onChange={(e) => setRules((current) => ({ ...current, oneDayPass: e.target.checked }))} className="h-4 w-4 accent-[#FF7A18]" />
                        </label>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="rounded-3xl border border-white/8 bg-[#101116] p-5 sm:p-6">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <div className="text-[11px] uppercase tracking-[0.22em] text-white/28">Sequence analyzer</div>
                      <h3 className="mt-2 text-xl font-semibold">What happens before the first loss?</h3>
                    </div>
                    <div className="rounded-full border border-[#FF7A18]/20 bg-[#FF7A18]/10 px-3 py-1 text-xs text-[#FFB347]">{dayBuckets.length} sessions</div>
                  </div>
                  <div className="mt-5 grid gap-3 sm:grid-cols-2">
                    <div className="rounded-2xl border border-white/8 bg-black/20 p-4"><div className="text-[11px] uppercase tracking-[0.18em] text-white/28">Avg wins before first loss</div><div className="mt-3 text-3xl font-bold tabular-nums">{n(stats.avgWins, 2)}</div></div>
                    <div className="rounded-2xl border border-white/8 bg-black/20 p-4"><div className="text-[11px] uppercase tracking-[0.18em] text-white/28">Avg PnL before first loss</div><div className="mt-3 text-3xl font-bold tabular-nums">{money(stats.avgBeforeLoss)}</div></div>
                  </div>
                  <div className="mt-5 space-y-4">
                    {stats.thresholds.map((item) => (
                      <div key={item.threshold}>
                        <div className="flex items-center justify-between text-sm mb-2">
                          <span className="text-white/55">P(win cluster &gt;= {item.threshold})</span>
                          <span className="text-white tabular-nums">{pct(item.rate * 100)}</span>
                        </div>
                        <div className="h-2 rounded-full bg-white/5 overflow-hidden"><div className="h-full rounded-full bg-gradient-to-r from-[#FF7A18] to-[#FFB347]" style={{ width: `${Math.max(item.rate * 100, 6)}%` }} /></div>
                      </div>
                    ))}
                  </div>
                  <div className="mt-6 grid gap-4 md:grid-cols-2">
                    <div className="rounded-2xl border border-white/8 bg-black/20 p-4">
                      <div className="text-[11px] uppercase tracking-[0.18em] text-white/28 mb-3">Win clusters</div>
                      <div className="space-y-2">
                        {stats.winDistribution.map((item) => (
                          <div key={item.label} className="flex items-center gap-3 text-xs">
                            <span className="w-8 text-white/45">{item.label}</span>
                            <div className="h-2 flex-1 rounded-full bg-white/5 overflow-hidden"><div className="h-full rounded-full bg-[#FF9B34]" style={{ width: `${Math.max((item.value / Math.max(dayBuckets.length, 1)) * 100, 6)}%` }} /></div>
                            <span className="w-8 text-right text-white/45 tabular-nums">{item.value}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div className="rounded-2xl border border-white/8 bg-black/20 p-4">
                      <div className="text-[11px] uppercase tracking-[0.18em] text-white/28 mb-3">Pass distribution</div>
                      <div className="space-y-2">
                        {stats.passDistribution.map((item) => (
                          <div key={item.label} className="flex items-center gap-3 text-xs">
                            <span className="w-8 text-white/45">{item.label}</span>
                            <div className="h-2 flex-1 rounded-full bg-white/5 overflow-hidden"><div className="h-full rounded-full bg-[#74e09d]" style={{ width: `${Math.max((item.value / Math.max(dayBuckets.length, 1)) * 100, 6)}%` }} /></div>
                            <span className="w-8 text-right text-white/45 tabular-nums">{item.value}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                  <div className="mt-6 rounded-2xl border border-white/8 bg-black/20 p-4">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-white/45">Monte Carlo summary</span>
                      <span className="text-white/70">2,000 runs over 20 sampled days</span>
                    </div>
                    <div className="mt-4 grid gap-3 sm:grid-cols-2">
                      <div><div className="text-[11px] uppercase tracking-[0.18em] text-white/28">Expected passes/day</div><div className="mt-2 text-2xl font-bold tabular-nums">{n(mc.expectedPasses, 2)}</div></div>
                      <div><div className="text-[11px] uppercase tracking-[0.18em] text-white/28">Net EV / day</div><div className="mt-2 text-2xl font-bold tabular-nums">{money(mc.netEV)}</div></div>
                    </div>
                  </div>
                </div>
              </section>
            ) : (
              <section className="grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
                <div className="rounded-3xl border border-white/8 bg-[#101116] p-5 sm:p-6">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <div className="text-[11px] uppercase tracking-[0.22em] text-white/28">Tracking dashboard</div>
                      <h3 className="mt-2 text-xl font-semibold">Log real eval attempts</h3>
                    </div>
                    <Database className="h-5 w-5 text-[#FFB347]" />
                  </div>
                  <div className="mt-5 space-y-4">
                    <div className="grid gap-4 sm:grid-cols-2">
                      <label className="block"><div className="mb-2 text-xs uppercase tracking-[0.16em] text-white/28">Firm</div><input value={draft.firm} onChange={(e) => setDraft((current) => ({ ...current, firm: e.target.value }))} className="w-full rounded-xl border border-white/10 bg-[#0c0d11] px-4 py-3 text-white outline-none focus:border-[#FF7A18]/40" /></label>
                      <label className="block"><div className="mb-2 text-xs uppercase tracking-[0.16em] text-white/28">Account label</div><input value={draft.accountLabel} onChange={(e) => setDraft((current) => ({ ...current, accountLabel: e.target.value }))} className="w-full rounded-xl border border-white/10 bg-[#0c0d11] px-4 py-3 text-white outline-none focus:border-[#FF7A18]/40" /></label>
                      <label className="block"><div className="mb-2 text-xs uppercase tracking-[0.16em] text-white/28">Attempted at</div><input type="datetime-local" value={draft.attemptedAt} onChange={(e) => setDraft((current) => ({ ...current, attemptedAt: e.target.value }))} className="w-full rounded-xl border border-white/10 bg-[#0c0d11] px-4 py-3 text-white outline-none focus:border-[#FF7A18]/40" /></label>
                      <label className="block"><div className="mb-2 text-xs uppercase tracking-[0.16em] text-white/28">Opened at</div><input type="datetime-local" value={draft.openedAt} onChange={(e) => setDraft((current) => ({ ...current, openedAt: e.target.value }))} className="w-full rounded-xl border border-white/10 bg-[#0c0d11] px-4 py-3 text-white outline-none focus:border-[#FF7A18]/40" /></label>
                    </div>
                    <div className="grid gap-4 sm:grid-cols-2">
                      <label className="block"><div className="mb-2 text-xs uppercase tracking-[0.16em] text-white/28">Outcome</div><select value={draft.outcome} onChange={(e) => setDraft((current) => ({ ...current, outcome: e.target.value as Outcome }))} className="w-full rounded-xl border border-white/10 bg-[#0c0d11] px-4 py-3 text-white outline-none focus:border-[#FF7A18]/40"><option value="Active">Active</option><option value="Pass">Pass</option><option value="Fail">Fail</option></select></label>
                      <label className="block"><div className="mb-2 text-xs uppercase tracking-[0.16em] text-white/28">Source</div><select value={draft.source} onChange={(e) => setDraft((current) => ({ ...current, source: e.target.value as TradeSource }))} className="w-full rounded-xl border border-white/10 bg-[#0c0d11] px-4 py-3 text-white outline-none focus:border-[#FF7A18]/40"><option value="TradingView">TradingView</option><option value="NinjaTrader">NinjaTrader</option></select></label>
                    </div>
                    <div className="grid gap-4 sm:grid-cols-2">
                      <label className="block"><div className="mb-2 text-xs uppercase tracking-[0.16em] text-white/28">Eval cost</div><input type="number" value={draft.evalCost} onChange={(e) => setDraft((current) => ({ ...current, evalCost: Number(e.target.value) }))} className="w-full rounded-xl border border-white/10 bg-[#0c0d11] px-4 py-3 text-white outline-none focus:border-[#FF7A18]/40" /></label>
                      <label className="block"><div className="mb-2 text-xs uppercase tracking-[0.16em] text-white/28">Activation fee</div><input type="number" value={draft.activationFee} onChange={(e) => setDraft((current) => ({ ...current, activationFee: Number(e.target.value) }))} className="w-full rounded-xl border border-white/10 bg-[#0c0d11] px-4 py-3 text-white outline-none focus:border-[#FF7A18]/40" /></label>
                    </div>
                    <label className="block"><div className="mb-2 text-xs uppercase tracking-[0.16em] text-white/28">Strategy notes</div><textarea rows={4} value={draft.notes} onChange={(e) => setDraft((current) => ({ ...current, notes: e.target.value }))} className="w-full rounded-2xl border border-white/10 bg-[#0c0d11] px-4 py-3 text-white outline-none focus:border-[#FF7A18]/40 resize-none" /></label>
                    <button type="button" onClick={addTrackerEntry} className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-[#FF7A18] to-[#FFB347] px-5 py-3 text-sm font-semibold text-white"><Plus className="h-4 w-4" />Add attempt</button>
                  </div>
                </div>

                <div className="rounded-3xl border border-white/8 bg-[#101116] p-5 sm:p-6">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <div className="text-[11px] uppercase tracking-[0.22em] text-white/28">Validation log</div>
                      <h3 className="mt-2 text-xl font-semibold">What happened in the real world?</h3>
                    </div>
                    <div className="rounded-full border border-white/8 bg-white/[0.03] px-3 py-1 text-xs text-white/45">{tracker.length} records</div>
                  </div>
                  <div className="mt-5 space-y-3">
                    {tracker.map((entry) => (
                      <div key={entry.id} className="rounded-2xl border border-white/8 bg-black/20 p-4">
                        <div className="flex items-start justify-between gap-4">
                          <div>
                            <div className="flex items-center gap-2 flex-wrap">
                              <div className="text-sm font-semibold">{entry.firm}</div>
                              <span className="rounded-full border border-white/8 bg-white/[0.03] px-2.5 py-1 text-[10px] uppercase tracking-[0.16em] text-white/45">{entry.accountLabel}</span>
                              <span className={`rounded-full border px-2.5 py-1 text-[10px] uppercase tracking-[0.16em] ${outcomeClass(entry.outcome)}`}>{entry.outcome}</span>
                            </div>
                            <div className="mt-2 text-xs text-white/40">Attempted {entry.attemptedAt.replace("T", " ")} | Opened {entry.openedAt.replace("T", " ")} | {entry.source}</div>
                          </div>
                          <div className="text-right">
                            <div className="text-sm font-semibold tabular-nums">{money(entry.evalCost + entry.activationFee)}</div>
                            <div className="mt-1 text-xs text-white/35">fees logged</div>
                          </div>
                        </div>
                        {entry.notes && <p className="mt-3 text-sm text-white/42 leading-relaxed">{entry.notes}</p>}
                        <div className="mt-3 flex flex-wrap gap-2 text-xs text-white/45">
                          <span className="rounded-full border border-white/8 bg-white/[0.03] px-2.5 py-1">Strategy: {entry.strategyVersion}</span>
                          {entry.activationFee > 0 ? <span className="rounded-full border border-white/8 bg-white/[0.03] px-2.5 py-1">Activation fee {money(entry.activationFee)}</span> : <span className="rounded-full border border-white/8 bg-white/[0.03] px-2.5 py-1">No activation fee</span>}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </section>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
