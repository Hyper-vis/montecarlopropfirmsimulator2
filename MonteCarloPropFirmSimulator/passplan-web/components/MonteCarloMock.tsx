"use client";

import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, ReferenceLine } from "recharts";

// Deterministic LCG — avoids hydration mismatch from Math.random()
function makeLcg(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

function generateChartData() {
  const rand = makeLcg(137);
  const START = 50000;
  const N_PATHS = 22;
  const N_DAYS = 28;
  const WINNING = Math.floor(N_PATHS * 0.68); // 15 winning, 7 losing

  const drifts = Array.from({ length: N_PATHS }, (_, i) =>
    i < WINNING ? 55 + rand() * 90 : -55 - rand() * 80
  );
  const vols = Array.from({ length: N_PATHS }, () => 260 + rand() * 380);
  const paths: number[][] = Array.from({ length: N_PATHS }, () => [START]);

  for (let day = 1; day <= N_DAYS; day++) {
    for (let p = 0; p < N_PATHS; p++) {
      const prev = paths[p][day - 1];
      const z = rand() - 0.5;
      paths[p].push(Math.max(46500, Math.round(prev + drifts[p] + z * 2 * vols[p])));
    }
  }

  const data: Record<string, number>[] = [
    {
      day: 0,
      mean: START,
      ...Object.fromEntries(Array.from({ length: N_PATHS }, (_, i) => [`p${i}`, START])),
    },
  ];
  for (let day = 1; day <= N_DAYS; day++) {
    const row: Record<string, number> = { day };
    for (let p = 0; p < N_PATHS; p++) row[`p${p}`] = paths[p][day];
    row.mean = Math.round(paths.reduce((s, pa) => s + pa[day], 0) / N_PATHS);
    data.push(row);
  }

  return { data, nPaths: N_PATHS, winning: WINNING };
}

const { data, nPaths, winning } = generateChartData();

export default function MonteCarloMock() {
  return (
    <div
      id="product"
      className="relative rounded-2xl border border-white/[0.07] bg-[#0c0c10] shadow-[0_8px_80px_rgba(0,0,0,0.7),0_0_0_1px_rgba(255,255,255,0.03)] overflow-hidden"
    >
      {/* Top bar */}
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/[0.05]">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-[#ff5f57]/70" />
          <div className="w-3 h-3 rounded-full bg-[#febc2e]/70" />
          <div className="w-3 h-3 rounded-full bg-[#28c840]/70" />
        </div>
        <div className="text-xs text-white/25 font-mono tracking-widest uppercase">
          passplan · monte carlo · 10,000 simulations
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
          <span className="text-[11px] text-green-400/70 font-semibold font-mono">LIVE</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3">
        {/* Chart area */}
        <div className="lg:col-span-2 p-6 relative border-b lg:border-b-0 lg:border-r border-white/[0.05]">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs text-white/25 uppercase tracking-widest font-mono">
              Equity paths — 10,000 sims
            </span>
            <span className="text-xs text-accent/70 font-mono">── $52,500 payout target</span>
          </div>
          <div style={{ height: 270 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
                <XAxis dataKey="day" hide />
                <YAxis domain={[46000, 54000]} hide />
                <ReferenceLine y={52500} stroke="rgba(255,122,24,0.45)" strokeDasharray="6 4" strokeWidth={1} />
                <ReferenceLine y={47000} stroke="rgba(248,113,113,0.3)" strokeDasharray="4 4" strokeWidth={1} />
                {/* Losing paths */}
                {Array.from({ length: nPaths - winning }, (_, i) => (
                  <Line
                    key={`loss-${i}`}
                    type="monotone"
                    dataKey={`p${winning + i}`}
                    stroke="rgba(248,113,113,0.13)"
                    strokeWidth={1}
                    dot={false}
                    isAnimationActive={false}
                  />
                ))}
                {/* Winning paths */}
                {Array.from({ length: winning }, (_, i) => (
                  <Line
                    key={`win-${i}`}
                    type="monotone"
                    dataKey={`p${i}`}
                    stroke="rgba(74,222,128,0.13)"
                    strokeWidth={1}
                    dot={false}
                    isAnimationActive={false}
                  />
                ))}
                {/* Mean path — animated on top */}
                <Line
                  type="monotone"
                  dataKey="mean"
                  stroke="#4ade80"
                  strokeWidth={2.5}
                  dot={false}
                  isAnimationActive
                  animationDuration={1400}
                  animationEasing="ease-out"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="flex items-center justify-between mt-1 px-1">
            <span className="text-[10px] text-white/20 font-mono">$50,000 start</span>
            <div className="flex items-center gap-4">
              <span className="text-[10px] text-accent/45 font-mono">▲ payout zone</span>
              <span className="text-[10px] text-red-400/35 font-mono">▼ drawdown limit</span>
            </div>
          </div>
        </div>

        {/* Stats panel */}
        <div className="p-6 flex flex-col justify-between">
          <div>
            <p className="text-[10px] text-white/20 uppercase tracking-[0.2em] font-mono mb-5">
              Simulation results
            </p>
            <div className="space-y-5">
              <div>
                <p className="text-[10px] text-white/35 uppercase tracking-wider mb-1.5">Pass probability</p>
                <p className="text-5xl font-bold text-gradient leading-none tabular-nums">89%</p>
              </div>
              <div className="h-px w-full bg-white/[0.05]" />
              <div>
                <p className="text-[10px] text-white/35 uppercase tracking-wider mb-1.5">Expected payout</p>
                <p className="text-2xl font-bold text-white tabular-nums">$736</p>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-[10px] text-white/35 uppercase tracking-wider mb-1.5">Blow risk</p>
                  <p className="text-xl font-bold text-red-400 tabular-nums">6.9%</p>
                </div>
                <div>
                  <p className="text-[10px] text-white/35 uppercase tracking-wider mb-1.5">Avg days</p>
                  <p className="text-xl font-bold text-white tabular-nums">22d</p>
                </div>
              </div>
            </div>
          </div>
          <div className="mt-6">
            <p className="text-[10px] text-white/20 uppercase tracking-wider mb-2 font-mono">Path distribution</p>
            <div className="flex rounded-full overflow-hidden h-2 gap-px">
              <div className="bg-green-400/65" style={{ width: "89%" }} />
              <div className="bg-red-400/55" style={{ width: "6.9%" }} />
              <div className="bg-white/12" style={{ width: "4.1%" }} />
            </div>
            <div className="flex items-center gap-4 mt-2.5">
              <span className="text-[10px] text-white/25 flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-sm bg-green-400/65 inline-block" /> Pass 89%
              </span>
              <span className="text-[10px] text-white/25 flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-sm bg-red-400/55 inline-block" /> Blow 6.9%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom bar */}
      <div className="border-t border-white/[0.04] px-5 py-2.5 flex items-center justify-between">
        <span className="text-[10px] text-white/20 font-mono">10,000 simulations · 0.4s</span>
        <span className="text-[10px] text-white/20 font-mono">Apex $50K · Until Payout mode</span>
      </div>
    </div>
  );
}