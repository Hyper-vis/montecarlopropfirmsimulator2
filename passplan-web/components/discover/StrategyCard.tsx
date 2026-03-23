"use client";

import { motion } from "framer-motion";
import { Strategy, RiskLevel } from "@/app/discover/page";

// ─── Mini MC chart ───────────────────────────────────────────────────────────

/** Simple LCG pseudo-random number generator (deterministic by seed). */
function makeLcg(seed: number) {
  let s = seed | 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) | 0;
    return (s >>> 0) / 0xffffffff;
  };
}

interface ChartPath {
  points: { x: number; y: number }[];
  win: boolean; // ended above target line
}

function generatePaths(seed: number, count = 18, steps = 40): ChartPath[] {
  const rand = makeLcg(seed);
  const paths: ChartPath[] = [];

  for (let i = 0; i < count; i++) {
    const points: { x: number; y: number }[] = [];
    let equity = 0;
    for (let s = 0; s <= steps; s++) {
      const pct = s / steps;
      const dx = (rand() - 0.47) * 5.5; // slight positive drift
      equity = equity + dx;
      equity = Math.max(-40, Math.min(80, equity));
      points.push({ x: pct, y: equity });
    }
    paths.push({ points, win: equity >= 30 });
  }
  return paths;
}

function MiniChart({ seed }: { seed: number }) {
  const W = 280;
  const H = 82;
  const PAD = { top: 6, bottom: 6, left: 4, right: 4 };

  const paths = generatePaths(seed);

  const scaleX = (v: number) => PAD.left + v * (W - PAD.left - PAD.right);
  const scaleY = (v: number) =>
    PAD.top + (1 - (v + 40) / 120) * (H - PAD.top - PAD.bottom);

  const toD = (pts: { x: number; y: number }[]) =>
    pts
      .map((p, i) => `${i === 0 ? "M" : "L"}${scaleX(p.x).toFixed(1)},${scaleY(p.y).toFixed(1)}`)
      .join(" ");

  const targetY = scaleY(30).toFixed(1);
  const stopY = scaleY(-30).toFixed(1);

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full"
      style={{ height: H }}
      aria-hidden
    >
      {/* Target line */}
      <line
        x1={PAD.left}
        y1={targetY}
        x2={W - PAD.right}
        y2={targetY}
        stroke="#22c55e"
        strokeWidth="0.8"
        strokeDasharray="3,3"
        opacity="0.4"
      />
      {/* Stop line */}
      <line
        x1={PAD.left}
        y1={stopY}
        x2={W - PAD.right}
        y2={stopY}
        stroke="#ef4444"
        strokeWidth="0.8"
        strokeDasharray="3,3"
        opacity="0.35"
      />
      {/* Zero line */}
      <line
        x1={PAD.left}
        y1={scaleY(0)}
        x2={W - PAD.right}
        y2={scaleY(0)}
        stroke="#ffffff"
        strokeWidth="0.6"
        opacity="0.1"
      />

      {/* Equity paths */}
      {paths.map((path, idx) => (
        <path
          key={idx}
          d={toD(path.points)}
          fill="none"
          stroke={path.win ? "#22c55e" : "#ef4444"}
          strokeWidth="0.85"
          opacity={path.win ? 0.32 : 0.18}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      ))}
    </svg>
  );
}

// ─── Risk pill ────────────────────────────────────────────────────────────────

const RISK_STYLE: Record<RiskLevel, string> = {
  "Very Low": "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20",
  Low: "bg-green-500/10 text-green-400 border border-green-500/20",
  Medium: "bg-yellow-500/10 text-yellow-400 border border-yellow-500/20",
  High: "bg-red-500/10 text-red-400 border border-red-500/20",
};

function passColor(p: number) {
  if (p >= 65) return "text-green-400";
  if (p >= 45) return "text-yellow-400";
  return "text-red-400";
}

// ─── Card ────────────────────────────────────────────────────────────────────

interface Props {
  strategy: Strategy;
}

export default function StrategyCard({ strategy }: Props) {
  const { name, instrument, style, passProbability, expectedPayout, blowRisk, avgDays, risk, seed, tag } =
    strategy;

  return (
    <motion.div
      whileHover={{ y: -3, borderColor: "rgba(255, 122, 24, 0.35)" }}
      transition={{ duration: 0.2, ease: "easeOut" }}
      className="group relative flex flex-col rounded-2xl border border-white/6 bg-[#111116] overflow-hidden
                 shadow-[0_1px_24px_rgba(0,0,0,0.45)] cursor-pointer"
    >
      {/* Tag badge */}
      {tag && (
        <div className="absolute top-3.5 right-3.5 z-10 rounded-full bg-accent/15 border border-accent/30 px-2.5 py-0.5 text-[10px] font-bold text-accent uppercase tracking-wider">
          {tag}
        </div>
      )}

      {/* Header */}
      <div className="px-5 pt-5 pb-3">
        <div className="flex items-start gap-3">
          {/* Instrument badge */}
          <div className="mt-0.5 flex-shrink-0 rounded-lg bg-white/5 px-2 py-1 text-[10px] font-mono font-bold text-white/50 border border-white/8">
            {instrument.split(" ")[1]}
          </div>
          <div>
            <h3 className="text-[15px] font-bold text-white leading-tight group-hover:text-accent-light transition-colors duration-200">
              {name}
            </h3>
            <p className="text-[11px] text-white/35 mt-0.5 font-medium">
              {instrument} · {style}
            </p>
          </div>
        </div>

        {/* Key metrics row */}
        <div className="mt-4 flex items-center gap-3">
          {/* Pass probability */}
          <div className="flex flex-col">
            <span className="text-[10px] text-white/30 uppercase tracking-wider mb-0.5">Pass Rate</span>
            <span className={`text-3xl font-black tabular-nums leading-none ${passColor(passProbability)}`}>
              {passProbability}
              <span className="text-lg font-bold">%</span>
            </span>
          </div>

          <div className="h-9 w-px bg-white/8 mx-1" />

          {/* Expected payout */}
          <div className="flex flex-col">
            <span className="text-[10px] text-white/30 uppercase tracking-wider mb-0.5">Exp. Payout</span>
            <span className="text-xl font-bold text-white tabular-nums">
              ${expectedPayout.toLocaleString()}
            </span>
          </div>

          {/* Risk pill — pushed to right */}
          <div className="ml-auto">
            <span className={`text-[10px] font-semibold px-2.5 py-1 rounded-full ${RISK_STYLE[risk]}`}>
              {risk}
            </span>
          </div>
        </div>
      </div>

      {/* Mini MC chart strip */}
      <div className="px-1 bg-gradient-to-b from-[#111116] to-[#0c0c10]">
        <MiniChart seed={seed} />
      </div>

      {/* Footer stats */}
      <div className="px-5 py-3.5 flex items-center gap-0 border-t border-white/5 bg-[#0c0c10]">
        <StatPill label="Blow Risk" value={`${blowRisk}%`} danger />
        <div className="h-7 w-px bg-white/6 mx-3" />
        <StatPill label="Avg Days" value={`${avgDays}d`} />
        <div className="flex-1" />
        {/* CTA arrow */}
        <motion.button
          whileTap={{ scale: 0.9 }}
          className="flex items-center gap-1.5 text-[11px] font-semibold text-white/40
                     hover:text-accent transition-colors duration-200 group/cta"
          aria-label={`Simulate ${name}`}
        >
          Simulate
          <svg
            className="w-3.5 h-3.5 group-hover/cta:translate-x-0.5 transition-transform duration-150"
            fill="none" viewBox="0 0 14 14" stroke="currentColor" strokeWidth="1.8"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M1 7h12M8 3l4 4-4 4" />
          </svg>
        </motion.button>
      </div>
    </motion.div>
  );
}

function StatPill({ label, value, danger = false }: { label: string; value: string; danger?: boolean }) {
  return (
    <div className="flex flex-col">
      <span className="text-[9px] text-white/25 uppercase tracking-wider">{label}</span>
      <span className={`text-[13px] font-bold tabular-nums ${danger ? "text-red-400/80" : "text-white/65"}`}>
        {value}
      </span>
    </div>
  );
}
