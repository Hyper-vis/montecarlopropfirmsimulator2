"use client";

import { motion } from "framer-motion";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import StrategyCard from "@/components/discover/StrategyCard";

export type RiskLevel = "Very Low" | "Low" | "Medium" | "High";

export interface Strategy {
  id: number;
  name: string;
  instrument: string;
  style: string;
  passProbability: number;
  expectedPayout: number;
  blowRisk: number;
  avgDays: number;
  risk: RiskLevel;
  seed: number;
  tag?: string;
}

const STRATEGIES: Strategy[] = [
  {
    id: 1,
    name: "ES Scalper Pro",
    instrument: "CME ES",
    style: "Scalp",
    passProbability: 89,
    expectedPayout: 736,
    blowRisk: 6.9,
    avgDays: 22,
    risk: "Low",
    seed: 137,
    tag: "Top Pick",
  },
  {
    id: 2,
    name: "NQ Momentum",
    instrument: "CME NQ",
    style: "Momentum",
    passProbability: 74,
    expectedPayout: 512,
    blowRisk: 14.2,
    avgDays: 31,
    risk: "Medium",
    seed: 421,
  },
  {
    id: 3,
    name: "YM Trend Follower",
    instrument: "CBOT YM",
    style: "Trend",
    passProbability: 82,
    expectedPayout: 1240,
    blowRisk: 9.1,
    avgDays: 18,
    risk: "Low",
    seed: 88,
    tag: "High EV",
  },
  {
    id: 4,
    name: "CL Mean Reversion",
    instrument: "NYMEX CL",
    style: "Mean Rev.",
    passProbability: 61,
    expectedPayout: 890,
    blowRisk: 21.5,
    avgDays: 27,
    risk: "Medium",
    seed: 512,
  },
  {
    id: 5,
    name: "GC Breakout",
    instrument: "COMEX GC",
    style: "Breakout",
    passProbability: 47,
    expectedPayout: 2100,
    blowRisk: 38.4,
    avgDays: 14,
    risk: "High",
    seed: 777,
  },
  {
    id: 6,
    name: "RTY Pairs Trade",
    instrument: "CME RTY",
    style: "Pairs",
    passProbability: 78,
    expectedPayout: 430,
    blowRisk: 10.3,
    avgDays: 35,
    risk: "Low",
    seed: 303,
  },
  {
    id: 7,
    name: "MES Swing",
    instrument: "CME MES",
    style: "Swing",
    passProbability: 56,
    expectedPayout: 1680,
    blowRisk: 27.8,
    avgDays: 20,
    risk: "High",
    seed: 999,
  },
  {
    id: 8,
    name: "ZN Bonds Ladder",
    instrument: "CBOT ZN",
    style: "Ladder",
    passProbability: 91,
    expectedPayout: 280,
    blowRisk: 4.2,
    avgDays: 44,
    risk: "Very Low",
    seed: 200,
    tag: "Most Stable",
  },
  {
    id: 9,
    name: "Crude Breakout",
    instrument: "NYMEX CL",
    style: "Breakout",
    passProbability: 35,
    expectedPayout: 3200,
    blowRisk: 52.1,
    avgDays: 11,
    risk: "High",
    seed: 1111,
  },
];

const RISK_FILTERS: (RiskLevel | "All")[] = ["All", "Very Low", "Low", "Medium", "High"];

export default function DiscoverPage() {
  return (
    <div className="min-h-screen bg-[#0B0B0F]">
      <Navbar />

      <main className="pt-24 pb-28">
        {/* Header */}
        <div className="max-w-6xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
            className="mb-14"
          >
            <p className="text-xs text-accent uppercase tracking-[0.25em] font-semibold mb-4">
              Strategy Discovery
            </p>
            <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-6">
              <div>
                <h1 className="text-4xl sm:text-5xl font-bold text-white leading-tight mb-3">
                  Discovered{" "}
                  <span className="text-gradient">Strategies</span>
                </h1>
                <p className="text-white/45 text-lg max-w-xl leading-relaxed">
                  Browse strategies tested across thousands of Monte Carlo simulations.
                  Each card shows real pass probability and expected payout.
                </p>
              </div>
              {/* Live count badge */}
              <div className="flex items-center gap-2.5 rounded-full border border-accent/25 bg-accent/5 px-5 py-2.5 self-start sm:self-auto shrink-0">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent opacity-75" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-accent" />
                </span>
                <span className="text-sm font-semibold text-accent-light">
                  {STRATEGIES.length} strategies tested
                </span>
              </div>
            </div>

            {/* Divider */}
            <div className="mt-10 h-px bg-gradient-to-r from-transparent via-white/8 to-transparent" />
          </motion.div>

          {/* Stats bar */}
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
            className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-12"
          >
            {[
              { label: "Avg Pass Rate", value: "68.1%", color: "text-green-400" },
              { label: "Avg Payout", value: "$1,141", color: "text-white" },
              { label: "Avg Blow Risk", value: "20.5%", color: "text-red-400" },
              { label: "Total Simulations", value: "90,000+", color: "text-accent-light" },
            ].map(({ label, value, color }) => (
              <div
                key={label}
                className="rounded-xl border border-white/6 bg-[#111116] px-5 py-4"
              >
                <p className="text-[11px] text-white/30 uppercase tracking-wider mb-1.5">{label}</p>
                <p className={`text-2xl font-bold tabular-nums ${color}`}>{value}</p>
              </div>
            ))}
          </motion.div>

          {/* Strategy grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {STRATEGIES.map((strategy, i) => (
              <motion.div
                key={strategy.id}
                initial={{ opacity: 0, y: 28 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{
                  duration: 0.48,
                  delay: (i % 3) * 0.08,
                  ease: [0.22, 1, 0.36, 1],
                }}
              >
                <StrategyCard strategy={strategy} />
              </motion.div>
            ))}
          </div>

          {/* Bottom note */}
          <motion.p
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="mt-14 text-center text-xs text-white/20 leading-relaxed"
          >
            All strategies use placeholder data. Pass probabilities are simulated via Monte Carlo using 10,000 runs per strategy.
            <br />
            Results are not financial advice. Past simulation results do not guarantee future performance.
          </motion.p>
        </div>
      </main>

      <Footer />
    </div>
  );
}
