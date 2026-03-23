"use client";

import { motion } from "framer-motion";
import { TrendingUp, TrendingDown, Clock, DollarSign, Activity, Target } from "lucide-react";

const metrics = [
    { label: "Pass Probability", value: "89%", color: "text-gradient", sub: "A-Tier strategy", icon: Target },
    { label: "Expected Payout", value: "$736", color: "text-white", sub: "per successful run", icon: DollarSign },
    { label: "Blow Risk", value: "6.9%", color: "text-red-400", sub: "trailing stop hit", icon: TrendingDown },
    { label: "Avg Days to Payout", value: "22d", color: "text-white", sub: "median path length", icon: Clock },
    { label: "Recency Trend", value: "+8.4%", color: "text-green-400", sub: "Improving vs history", icon: TrendingUp },
    { label: "Monthly EV", value: "$654", color: "text-white", sub: "net expected value", icon: Activity },
];

export default function ProductPreview() {
    return (
        <section id="product" className="py-28 relative">
            <div className="max-w-7xl mx-auto px-6">
                {/* Header */}
                <div className="text-center max-w-2xl mx-auto mb-16">
                    <p className="text-xs text-accent uppercase tracking-[0.2em] font-semibold mb-4">The product</p>
                    <h2 className="text-4xl sm:text-5xl font-bold text-white leading-tight mb-5">
                        Simulation results that actually{" "}
                        <span className="text-gradient">mean something.</span>
                    </h2>
                    <p className="text-white/45 text-lg leading-relaxed">
                        Every number comes directly from running your real trades through
                        the prop firm challenge rules. No estimates, no guessing.
                    </p>
                </div>

                {/* Preview card */}
                <div className="max-w-4xl mx-auto rounded-2xl border border-white/8 bg-[#0e0e14] overflow-hidden shadow-[0_8px_80px_rgba(0,0,0,0.6)]">
                    {/* Card top */}
                    <div className="border-b border-white/5 px-6 py-4 flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <div className="h-2 w-2 rounded-full bg-accent animate-pulse" />
                            <span className="text-xs text-white/40 font-mono uppercase tracking-widest">
                                Strategy analysis — Until Payout mode
                            </span>
                        </div>
                        <div className="hidden sm:flex items-center gap-2 rounded-full border border-accent/20 bg-accent/5 px-3 py-1">
                            <span className="text-xs text-accent font-semibold">★ A-Tier</span>
                        </div>
                    </div>

                    {/* Metrics grid */}
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-px bg-white/4">
                        {metrics.map(({ label, value, color, sub, icon: Icon }, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0, y: 16 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ duration: 0.4, delay: i * 0.07, ease: [0.22, 1, 0.36, 1] }}
                                className="bg-[#0e0e14] p-6 group hover:bg-[#111116] transition-all duration-200 hover:translate-y-[-1px]"
                            >
                                <div className="flex items-start justify-between mb-3">
                                    <p className="text-[11px] text-white/35 uppercase tracking-wider leading-none">{label}</p>
                                    <Icon className="w-3.5 h-3.5 text-white/15 group-hover:text-accent/40 transition-colors duration-200" />
                                </div>
                                <p className={`text-3xl font-bold leading-none mb-1.5 ${color}`}>{value}</p>
                                <p className="text-[11px] text-white/25">{sub}</p>
                            </motion.div>
                        ))}

                    </div>

                    {/* Equity paths bar */}
                    <div className="border-t border-white/5 px-6 py-5">
                        <div className="flex items-center justify-between mb-3">
                            <span className="text-xs text-white/30 uppercase tracking-widest">Equity path distribution</span>
                            <span className="text-xs text-white/25 font-mono">10,000 sims</span>
                        </div>
                        {/* Stacked bar mock */}
                        <div className="flex rounded-full overflow-hidden h-3 gap-0.5">
                            <div className="bg-accent/80" style={{ width: "89%" }} title="Pass" />
                            <div className="bg-red-500/70" style={{ width: "6.9%" }} title="Blow" />
                            <div className="bg-white/15" style={{ width: "4.1%" }} title="Timeout" />
                        </div>
                        <div className="flex items-center gap-5 mt-2.5">
                            <div className="flex items-center gap-1.5 text-[10px] text-white/35">
                                <div className="w-2 h-2 rounded-sm bg-accent/80" /> Pass 89%
                            </div>
                            <div className="flex items-center gap-1.5 text-[10px] text-white/35">
                                <div className="w-2 h-2 rounded-sm bg-red-500/70" /> Blow 6.9%
                            </div>
                            <div className="flex items-center gap-1.5 text-[10px] text-white/35">
                                <div className="w-2 h-2 rounded-sm bg-white/20" /> Timeout 4.1%
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}
