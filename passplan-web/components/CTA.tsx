"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight } from "lucide-react";

export default function CTA() {
    return (
        <section className="py-28 relative overflow-hidden">
            {/* Outer ambient glow */}
            <div className="absolute inset-0 bg-[radial-gradient(ellipse_60%_50%_at_50%_50%,rgba(255,122,24,0.08),transparent)] pointer-events-none" />

            <div className="relative max-w-4xl mx-auto px-6">
                <motion.div
                    initial={{ opacity: 0, y: 36 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
                    className="glow-border relative rounded-2xl bg-[#0e0e12] px-8 sm:px-16 py-16 text-center overflow-hidden"
                >
                    {/* Inner grid bg */}
                    <div
                        className="absolute inset-0 pointer-events-none opacity-[0.04]"
                        style={{
                            backgroundImage:
                                "linear-gradient(rgba(255,255,255,.4) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.4) 1px,transparent 1px)",
                            backgroundSize: "48px 48px",
                        }}
                    />
                    {/* Inner radial highlight */}
                    <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_60%_at_50%_0%,rgba(255,122,24,0.07),transparent)] pointer-events-none" />

                    <div className="relative">
                        {/* Pill */}
                        <div className="inline-flex items-center gap-2 rounded-full border border-accent/20 bg-accent/5 px-4 py-1.5 text-xs font-medium text-accent/80 mb-8">
                            Start for free · No card required
                        </div>

                        <h2 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white leading-[1.08] mb-6">
                            Stop guessing.
                            <br />
                            <span className="text-gradient">Know your pass probability.</span>
                        </h2>

                        <p className="text-lg text-white/45 max-w-lg mx-auto mb-10 leading-relaxed">
                            Upload any trade history CSV and see your real chances in seconds.
                            Free to try. No account needed.
                        </p>

                        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                            <Link
                                href="/simulator"
                                className="group inline-flex items-center gap-2.5 rounded-xl bg-gradient-to-r from-[#FF7A18] to-[#FFB347] px-8 py-4 text-base font-semibold text-white transition-all duration-300 hover:scale-[1.04] hover:shadow-[0_0_32px_rgba(255,122,24,0.5),0_0_64px_rgba(255,122,24,0.18)]"
                            >
                                Try the simulator
                                <ArrowRight className="w-4 h-4 transition-transform duration-200 group-hover:translate-x-1" />
                            </Link>
                            <a
                                href="#how-it-works"
                                className="text-sm text-white/40 hover:text-white/70 transition-colors duration-200 underline underline-offset-4"
                            >
                                Learn how it works first
                            </a>
                        </div>

                        {/* Social proof strip */}
                        <div className="mt-12 flex flex-wrap items-center justify-center gap-6 text-white/18 text-xs font-medium uppercase tracking-widest">
                            <span>Apex Trader Funding</span>
                            <span className="w-1 h-1 rounded-full bg-white/15" />
                            <span>Topstep</span>
                            <span className="w-1 h-1 rounded-full bg-white/15" />
                            <span>Take Profit Trader</span>
                            <span className="w-1 h-1 rounded-full bg-white/15" />
                            <span>TradeDay</span>
                        </div>
                    </div>
                </motion.div>
            </div>
        </section>
    );
}
