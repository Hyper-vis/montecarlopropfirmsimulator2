"use client";

import { motion } from "framer-motion";
import { TrendingDown, Dice5, ShieldAlert } from "lucide-react";

const points = [
  {
    icon: TrendingDown,
    title: "A profitable strategy can still fail most challenges.",
    body: "Even strategies with strong track records routinely blow 30–60% of prop firm attempts. Past profits don't predict pass rates.",
  },
  {
    icon: Dice5,
    title: "Without data, you're gambling on a $600 entry fee.",
    body: 'Most traders pick a challenge based on gut feel. "I\'ve been profitable this month" is not a risk model.',
  },
  {
    icon: ShieldAlert,
    title: "Prop firms are designed to be statistically hard.",
    body: "The rules — trailing drawdown, daily loss limits, profit targets — create a narrow path. One bad day can wipe a week of gains.",
  },
];

export default function Problem() {
  return (
    <section className="py-28 relative">
      {/* Subtle divider glow */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-px h-24 bg-gradient-to-b from-transparent via-accent/30 to-transparent" />

      <div className="max-w-7xl mx-auto px-6">
        {/* Header */}
        <div className="text-center max-w-2xl mx-auto mb-16">
          <p className="text-xs text-accent uppercase tracking-[0.2em] font-semibold mb-4">The problem</p>
          <h2 className="text-4xl sm:text-5xl font-bold text-white leading-tight mb-5">
            Most prop traders are{" "}
            <span className="text-gradient">guessing.</span>
          </h2>
          <p className="text-white/50 text-lg leading-relaxed">
            Prop firm challenges are difficult because traders don't know the real
            probability of success. PassPlan reveals the real odds.
          </p>
        </div>

        {/* Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {points.map(({ icon: Icon, title, body }, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 28 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.1, ease: [0.22, 1, 0.36, 1] }}
              className="card-hover group relative rounded-2xl border border-white/6 bg-[#111116] p-7 transition-all duration-300"
            >
              <div className="absolute inset-x-0 top-0 h-px rounded-t-2xl bg-gradient-to-r from-transparent via-accent/0 to-transparent group-hover:via-accent/40 transition-all duration-500" />

              <div className="mb-5 inline-flex items-center justify-center w-11 h-11 rounded-xl border border-white/8 bg-white/4 text-accent transition-colors duration-200 group-hover:border-accent/20 group-hover:bg-accent/8">
                <Icon className="w-5 h-5" />
              </div>
              <h3 className="text-base font-semibold text-white leading-snug mb-3">{title}</h3>
              <p className="text-sm text-white/45 leading-relaxed">{body}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
