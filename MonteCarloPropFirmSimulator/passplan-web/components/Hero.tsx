"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight, Play } from "lucide-react";

const MonteCarloMock = dynamic(() => import("./MonteCarloMock"), { ssr: false });

const fadeUp = (delay = 0) => ({
    initial: { opacity: 0, y: 22 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.55, delay, ease: [0.22, 1, 0.36, 1] as [number, number, number, number] },
});

export default function Hero() {
    return (
        <section className="relative pt-32 pb-24 overflow-hidden">
            {/* Animated grid background */}
            <div
                className="absolute inset-0 pointer-events-none"
                style={{
                    backgroundImage:
                        "linear-gradient(rgba(255,255,255,0.025) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,0.025) 1px,transparent 1px)",
                    backgroundSize: "48px 48px",
                    maskImage: "radial-gradient(ellipse 90% 55% at 50% 0%, black 30%, transparent 100%)",
                    WebkitMaskImage: "radial-gradient(ellipse 90% 55% at 50% 0%, black 30%, transparent 100%)",
                }}
            />
            {/* Pulsing radial orange glow */}
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[900px] h-[600px] pointer-events-none">
                <div
                    className="absolute inset-0 animate-glow-pulse"
                    style={{
                        background:
                            "radial-gradient(ellipse at center, rgba(255,122,24,0.22) 0%, rgba(255,122,24,0.06) 42%, transparent 70%)",
                    }}
                />
            </div>
            {/* Deeper ambient glow */}
            <div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-[500px] h-[300px] rounded-full bg-accent/5 blur-[80px] pointer-events-none" />

            <div className="relative max-w-7xl mx-auto px-6">
                {/* Trust signal */}
                <motion.div {...fadeUp(0)} className="flex justify-center mb-5">
                    <span className="text-xs text-white/25 uppercase tracking-[0.3em] font-medium">
                        Built for prop trading challenges
                    </span>
                </motion.div>

                {/* Badge */}
                <motion.div {...fadeUp(0.07)} className="flex justify-center mb-8">
                    <div className="inline-flex items-center gap-2 rounded-full border border-accent/30 bg-accent/5 px-4 py-1.5 text-xs font-medium text-accent-light">
                        <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent opacity-75" />
                            <span className="relative inline-flex h-2 w-2 rounded-full bg-accent" />
                        </span>
                        Monte Carlo Simulation Engine
                    </div>
                </motion.div>

                {/* Headline */}
                <motion.div {...fadeUp(0.13)} className="text-center max-w-4xl mx-auto mb-6">
                    <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold tracking-tight text-white leading-[1.08]">
                        Know if your strategy will pass{" "}
                        <span className="text-gradient">before risking</span>{" "}
                        a prop challenge.
                    </h1>
                </motion.div>

                {/* Subheadline */}
                <motion.p
                    {...fadeUp(0.19)}
                    className="text-center text-lg sm:text-xl text-white/45 max-w-2xl mx-auto mb-10 leading-relaxed"
                >
                    Upload your trade history and run thousands of simulated prop firm
                    challenges to see your real pass probability.
                </motion.p>

                {/* Buttons */}
                <motion.div
                    {...fadeUp(0.25)}
                    className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-20"
                >
                    <Link
                        href="/simulator"
                        className="group inline-flex items-center gap-2.5 rounded-xl bg-gradient-to-r from-[#FF7A18] to-[#FFB347] px-7 py-3.5 text-base font-semibold text-white transition-all duration-300 hover:scale-[1.04] hover:shadow-[0_0_32px_rgba(255,122,24,0.5),0_0_64px_rgba(255,122,24,0.18)]"
                    >
                        Try the simulator
                        <ArrowRight className="w-4 h-4 transition-transform duration-200 group-hover:translate-x-1" />
                    </Link>
                    <a
                        href="#product"
                        className="group inline-flex items-center gap-2.5 rounded-xl border border-white/10 bg-white/4 px-7 py-3.5 text-base font-semibold text-white/75 hover:text-white hover:border-white/20 hover:bg-white/7 transition-all duration-200"
                    >
                        <Play className="w-4 h-4 text-accent" />
                        See example results
                    </a>
                </motion.div>

                {/* Hero visual */}
                <motion.div
                    initial={{ opacity: 0, y: 40 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.75, delay: 0.35, ease: [0.22, 1, 0.36, 1] }}
                    className="max-w-4xl mx-auto"
                >
                    <MonteCarloMock />
                </motion.div>
            </div>
        </section>
    );
}
