"use client";

import { motion } from "framer-motion";
import { UploadCloud, Activity, Target } from "lucide-react";

const steps = [
  {
    number: "01",
    icon: UploadCloud,
    title: "Upload your trades",
    body: "Export your trade history from any platform — TradingView, NinjaTrader, TradeZero. Paste or upload the CSV. We handle the rest.",
    detail: "Supports standard CSV trade exports",
  },
  {
    number: "02",
    icon: Activity,
    title: "Run Monte Carlo simulations",
    body: "Our engine samples your real trade data and runs thousands of virtual prop firm challenges, respecting every rule of the challenge exactly.",
    detail: "10,000+ simulations in under a second",
  },
  {
    number: "03",
    icon: Target,
    title: "See your real pass probability",
    body: "Get a clear pass %, blow risk, expected payout, and equity path visualization. Know before you pay the fee.",
    detail: "Strategy tier, recency drift, rescue analysis",
  },
];

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="py-28 relative">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_50%_at_50%_50%,rgba(255,122,24,0.04),transparent)] pointer-events-none" />

      <div className="relative max-w-7xl mx-auto px-6">
        {/* Header */}
        <div className="text-center max-w-xl mx-auto mb-16">
          <p className="text-xs text-accent uppercase tracking-[0.2em] font-semibold mb-4">How it works</p>
          <h2 className="text-4xl sm:text-5xl font-bold text-white leading-tight">
            Three steps to know<br />
            <span className="text-gradient">your real odds.</span>
          </h2>
        </div>

        {/* Steps */}
        <div className="relative grid grid-cols-1 md:grid-cols-3 gap-5">
          {/* Connector line - desktop only */}
          <div className="hidden md:block absolute top-10 left-[calc(16.67%+2rem)] right-[calc(16.67%+2rem)] h-px bg-gradient-to-r from-accent/30 via-accent/10 to-accent/30 pointer-events-none" />

          {steps.map(({ number, icon: Icon, title, body, detail }, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 28 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.12, ease: [0.22, 1, 0.36, 1] }}
              className="card-hover group relative rounded-2xl border border-white/6 bg-[#111116] p-8 transition-all duration-300"
            >
              {/* Top accent line */}
              <div className="absolute inset-x-0 top-0 h-px rounded-t-2xl bg-gradient-to-r from-transparent via-accent/0 to-transparent group-hover:via-accent/50 transition-all duration-500" />

              {/* Step number + icon row */}
              <div className="flex items-center justify-between mb-6">
                <span className="text-5xl font-black text-white/5 leading-none select-none tabular-nums">
                  {number}
                </span>
                <div className="icon-glow w-11 h-11 rounded-xl border border-accent/25 bg-accent/10 flex items-center justify-center text-accent transition-shadow duration-300 group-hover:shadow-[0_0_24px_rgba(255,122,24,0.4)]">
                  <Icon className="w-5 h-5" />
                </div>
              </div>

              <h3 className="text-lg font-semibold text-white mb-3 leading-snug">{title}</h3>
              <p className="text-sm text-white/45 leading-relaxed mb-5">{body}</p>

              {/* Detail pill */}
              <div className="inline-flex items-center gap-1.5 rounded-full border border-white/8 bg-white/3 px-3 py-1 text-[11px] text-white/35 font-medium">
                <span className="w-1.5 h-1.5 rounded-full bg-accent/60" />
                {detail}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
