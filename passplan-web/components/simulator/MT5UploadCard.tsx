"use client";

import { useMemo, useState } from "react";

type Mt5UploadResult = {
    num_trades: number;
    pass_probability: number;
    blow_probability: number;
    expected_payout: number;
    trade_sample: number[];
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

function formatPct(value: number): string {
    return `${Math.round(value * 100)}%`;
}

function formatUsd(value: number): string {
    return new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 0,
    }).format(value);
}

export default function MT5UploadCard() {
    const [file, setFile] = useState<File | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string>("");
    const [result, setResult] = useState<Mt5UploadResult | null>(null);

    const loadingSteps = useMemo(
        () => [
            "Analyzing MT5 report...",
            "Extracting trades...",
            "Running challenge simulation...",
        ],
        []
    );

    async function onUpload() {
        if (!file || isLoading) return;

        setIsLoading(true);
        setError("");
        setResult(null);

        try {
            const formData = new FormData();
            formData.append("file", file);

            const response = await fetch(`${API_BASE}/upload/mt5`, {
                method: "POST",
                body: formData,
            });

            const payload = await response.json();
            if (!response.ok) {
                setError(payload?.error || "Failed to analyze MT5 report.");
                return;
            }

            setResult(payload as Mt5UploadResult);
        } catch {
            setError("Could not reach the backend. Please try again.");
        } finally {
            setIsLoading(false);
        }
    }

    return (
        <section
            style={{
                background: "#0e0e12",
                border: "1px solid rgba(255,255,255,0.07)",
                borderRadius: 12,
                padding: "18px 20px",
            }}
        >
            <div style={{ marginBottom: 14 }}>
                <h2 style={{ color: "#fff", fontSize: 18, marginBottom: 6 }}>Upload MT5 Backtest Report</h2>
                <p style={{ color: "rgba(255,255,255,0.6)", fontSize: 13 }}>
                    Supports MT5 Strategy Tester and Trade History exports (.csv, .xlsx)
                </p>
            </div>

            <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                <input
                    type="file"
                    accept=".csv,.xlsx,.xls,.xlsm"
                    onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                    style={{
                        background: "#0B0B0F",
                        color: "#e5e5e5",
                        border: "1px solid rgba(255,255,255,0.12)",
                        borderRadius: 8,
                        padding: "8px 10px",
                        fontSize: 13,
                        minWidth: 260,
                    }}
                />
                <button
                    onClick={onUpload}
                    disabled={!file || isLoading}
                    style={{
                        background: "linear-gradient(135deg, #FF7A18 0%, #FFB347 100%)",
                        color: "#fff",
                        border: "none",
                        borderRadius: 8,
                        padding: "9px 14px",
                        fontWeight: 700,
                        fontSize: 13,
                        cursor: !file || isLoading ? "not-allowed" : "pointer",
                        opacity: !file || isLoading ? 0.6 : 1,
                    }}
                >
                    {isLoading ? "Running..." : "Upload file"}
                </button>
            </div>

            {isLoading && (
                <div
                    style={{
                        marginTop: 14,
                        padding: 12,
                        borderRadius: 8,
                        background: "rgba(255,122,24,0.08)",
                        border: "1px solid rgba(255,122,24,0.25)",
                    }}
                >
                    {loadingSteps.map((step) => (
                        <div key={step} style={{ color: "#ffb57a", fontSize: 13, marginBottom: 4 }}>
                            {step}
                        </div>
                    ))}
                </div>
            )}

            {!!error && (
                <div
                    style={{
                        marginTop: 14,
                        padding: 12,
                        borderRadius: 8,
                        background: "rgba(248,113,113,0.12)",
                        border: "1px solid rgba(248,113,113,0.35)",
                        color: "#fca5a5",
                        fontSize: 13,
                    }}
                >
                    {error}
                </div>
            )}

            {result && (
                <div
                    style={{
                        marginTop: 16,
                        display: "grid",
                        gridTemplateColumns: "repeat(4, minmax(120px, 1fr))",
                        gap: 10,
                    }}
                >
                    <Metric label="Pass Chance" value={formatPct(result.pass_probability)} valueColor="#4ade80" />
                    <Metric label="Expected Payout" value={formatUsd(result.expected_payout)} valueColor="#fff" />
                    <Metric label="Blow Risk" value={formatPct(result.blow_probability)} valueColor="#f87171" />
                    <Metric label="Trades Analyzed" value={String(result.num_trades)} valueColor="#fff" />
                </div>
            )}
        </section>
    );
}

function Metric({ label, value, valueColor }: { label: string; value: string; valueColor: string }) {
    return (
        <div
            style={{
                background: "#0B0B0F",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 10,
                padding: "10px 12px",
            }}
        >
            <div
                style={{
                    fontSize: 11,
                    textTransform: "uppercase",
                    letterSpacing: "0.08em",
                    color: "rgba(255,255,255,0.5)",
                    marginBottom: 6,
                }}
            >
                {label}
            </div>
            <div style={{ fontSize: 24, fontWeight: 700, color: valueColor }}>{value}</div>
        </div>
    );
}
