import { BarChart2 } from "lucide-react";

export default function Footer() {
  return (
    <footer className="border-t border-white/5 py-12">
      <div className="max-w-7xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-6">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-accent-gradient flex items-center justify-center">
            <BarChart2 className="w-3.5 h-3.5 text-white" strokeWidth={2.5} />
          </div>
          <span className="font-bold text-white text-sm tracking-tight">
            Pass<span className="text-gradient">Plan</span>
          </span>
        </div>

        <p className="text-xs text-white/25 text-center">
          © {new Date().getFullYear()} PassPlan. Prop Firm Probability Engine.
          <span className="mx-2 text-white/10">·</span>
          Not financial advice.
        </p>

        <div className="flex items-center gap-5 text-xs text-white/30">
          <a href="#" className="hover:text-white/60 transition-colors">Privacy</a>
          <a href="#" className="hover:text-white/60 transition-colors">Terms</a>
          <a href="#" className="hover:text-white/60 transition-colors">Contact</a>
        </div>
      </div>
    </footer>
  );
}
