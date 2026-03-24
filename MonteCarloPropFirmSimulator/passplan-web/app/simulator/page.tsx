"use client";

import Link from "next/link";
import { LayoutDashboard, Compass, Trophy, BarChart2, ArrowLeft } from "lucide-react";
import { useState } from "react";

const NAV = [
  { label: "Dashboard", icon: LayoutDashboard, href: "/simulator" },
  { label: "Discover", icon: Compass, href: "/discover" },
  { label: "Leaderboard", icon: Trophy, href: "/simulator" },
];

const ACCENT = "linear-gradient(135deg, #FF7A18 0%, #FFB347 100%)";
const BG = "#0B0B0F";
const SURFACE = "#0e0e12";
const BORDER = "rgba(255,255,255,0.06)";
const ACTIVE_BG = "rgba(255,122,24,0.10)";
const ACTIVE_BORDER = "rgba(255,122,24,0.22)";

export default function SimulatorPage() {
  const [uploadMode, setUploadMode] = useState<"tradingview" | "mt5" | "ninjatrader">("tradingview");
  const SIM_UI_VERSION = "20260324f";

  return (
    <div style={{
      display: "flex",
      height: "100vh",
      background: BG,
      fontFamily: "'Segoe UI', system-ui, -apple-system, sans-serif",
      overflow: "hidden",
    }}>

      {/* ── Sidebar ── */}
      <aside style={{
        width: 232,
        flexShrink: 0,
        background: SURFACE,
        borderRight: `1px solid ${BORDER}`,
        display: "flex",
        flexDirection: "column",
        boxShadow: "4px 0 24px rgba(0,0,0,0.35)",
      }}>

        {/* Logo row */}
        <div style={{
          height: 60,
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "0 20px",
          borderBottom: `1px solid ${BORDER}`,
          flexShrink: 0,
        }}>
          <div style={{
            width: 30, height: 30, borderRadius: 9,
            background: ACCENT,
            display: "flex", alignItems: "center", justifyContent: "center",
            boxShadow: "0 0 14px rgba(255,122,24,0.45)",
          }}>
            <BarChart2 size={15} color="#fff" strokeWidth={2.5} />
          </div>
          <span style={{ fontWeight: 700, fontSize: 16.5, color: "#fff", letterSpacing: "-0.02em" }}>
            Pass
            <span style={{
              background: "linear-gradient(90deg,#FF7A18,#FFB347)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}>Plan</span>
          </span>
        </div>

        {/* Section label */}
        <div style={{
          padding: "20px 20px 8px",
          fontSize: 10,
          fontWeight: 600,
          letterSpacing: "0.14em",
          textTransform: "uppercase",
          color: "rgba(255,255,255,0.2)",
        }}>
          Navigation
        </div>

        {/* Nav items */}
        <nav style={{ padding: "0 10px", flex: 1 }}>
          {NAV.map(({ label, icon: Icon, href }) => {
            const active = label === "Dashboard";
            return (
              <Link key={label} href={href} style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "9px 12px",
                borderRadius: 10,
                marginBottom: 3,
                textDecoration: "none",
                fontSize: 13.5,
                fontWeight: active ? 600 : 400,
                color: active ? "#fff" : "rgba(255,255,255,0.38)",
                background: active ? ACTIVE_BG : "transparent",
                border: `1px solid ${active ? ACTIVE_BORDER : "transparent"}`,
                position: "relative",
                transition: "all 0.15s ease",
              }}>
                {/* Left accent bar */}
                {active && (
                  <div style={{
                    position: "absolute",
                    left: 0, top: "50%",
                    transform: "translateY(-50%)",
                    width: 3, height: "55%",
                    borderRadius: "0 2px 2px 0",
                    background: ACCENT,
                    boxShadow: "0 0 8px rgba(255,122,24,0.6)",
                  }} />
                )}
                <Icon
                  size={15}
                  color={active ? "#FF7A18" : "rgba(255,255,255,0.28)"}
                  strokeWidth={active ? 2.2 : 1.8}
                />
                {label}
              </Link>
            );
          })}
        </nav>

        {/* Divider + back link */}
        <div style={{
          margin: "0 10px",
          height: 1,
          background: BORDER,
        }} />
        <div style={{ padding: "10px 10px 18px" }}>
          <Link href="/" style={{
            display: "flex", alignItems: "center", gap: 8,
            padding: "8px 12px", borderRadius: 10,
            textDecoration: "none",
            fontSize: 12.5,
            color: "rgba(255,255,255,0.22)",
            border: "1px solid transparent",
            transition: "all 0.15s",
          }}>
            <ArrowLeft size={13} strokeWidth={1.8} />
            Back to site
          </Link>
        </div>
      </aside>

      {/* ── Right column ── */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>

        {/* Topbar */}
        <header style={{
          height: 60,
          flexShrink: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "0 28px",
          borderBottom: `1px solid ${BORDER}`,
          background: SURFACE,
          boxShadow: "0 1px 0 rgba(255,255,255,0.03)",
        }}>
          {/* Left: title */}
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <h1 style={{
              fontSize: 15,
              fontWeight: 600,
              color: "#fff",
              margin: 0,
              letterSpacing: "-0.01em",
            }}>
              Simulator
            </h1>
            <span style={{
              fontSize: 10,
              fontWeight: 700,
              padding: "3px 9px",
              borderRadius: 999,
              background: "rgba(255,122,24,0.10)",
              color: "#FF7A18",
              border: "1px solid rgba(255,122,24,0.22)",
              letterSpacing: "0.1em",
              textTransform: "uppercase",
            }}>
              Live
            </span>
          </div>

          {/* Right: API status indicator */}
          <div style={{
            display: "flex", alignItems: "center", gap: 6,
            fontSize: 11.5,
            color: "rgba(255,255,255,0.3)",
          }}>
            <span style={{
              width: 7, height: 7, borderRadius: "50%",
              background: "#4ade80",
              boxShadow: "0 0 6px rgba(74,222,128,0.7)",
              display: "inline-block",
            }} />
            localhost:8000
          </div>
        </header>

        {/* Main content — iframe fills all remaining space */}
        <main style={{
          flex: 1,
          minHeight: 0,
          background: BG,
          display: "flex",
          flexDirection: "column",
        }}>
          <div style={{ padding: "18px 20px 10px" }}>
            <div
              style={{
                display: "inline-flex",
                background: "#0e0e12",
                border: `1px solid ${BORDER}`,
                borderRadius: 10,
                padding: 4,
                gap: 4,
              }}
            >
              <button
                type="button"
                onClick={() => setUploadMode("tradingview")}
                style={{
                  border: `1px solid ${uploadMode === "tradingview" ? ACTIVE_BORDER : "transparent"}`,
                  background: uploadMode === "tradingview" ? ACTIVE_BG : "transparent",
                  color: uploadMode === "tradingview" ? "#fff" : "rgba(255,255,255,0.65)",
                  borderRadius: 8,
                  padding: "8px 12px",
                  fontSize: 13,
                  fontWeight: 600,
                  cursor: "pointer",
                }}
              >
                TradingView Upload
              </button>
              <button
                type="button"
                onClick={() => setUploadMode("mt5")}
                style={{
                  border: `1px solid ${uploadMode === "mt5" ? ACTIVE_BORDER : "transparent"}`,
                  background: uploadMode === "mt5" ? ACTIVE_BG : "transparent",
                  color: uploadMode === "mt5" ? "#fff" : "rgba(255,255,255,0.65)",
                  borderRadius: 8,
                  padding: "8px 12px",
                  fontSize: 13,
                  fontWeight: 600,
                  cursor: "pointer",
                }}
              >
                MT5 File Upload
              </button>
              <button
                type="button"
                onClick={() => setUploadMode("ninjatrader")}
                style={{
                  border: `1px solid ${uploadMode === "ninjatrader" ? ACTIVE_BORDER : "transparent"}`,
                  background: uploadMode === "ninjatrader" ? ACTIVE_BG : "transparent",
                  color: uploadMode === "ninjatrader" ? "#fff" : "rgba(255,255,255,0.65)",
                  borderRadius: 8,
                  padding: "8px 12px",
                  fontSize: 13,
                  fontWeight: 600,
                  cursor: "pointer",
                }}
              >
                NinjaTrader Upload
              </button>
            </div>
          </div>

          <iframe
            src={`/simulator-ui.html?v=${SIM_UI_VERSION}&uploadMode=${uploadMode}`}
            title="PassPlan Simulator"
            style={{
              display: "block",
              width: "100%",
              height: "100%",
              border: "none",
              minHeight: 360,
            }}
          />
        </main>
      </div>
    </div>
  );
}
