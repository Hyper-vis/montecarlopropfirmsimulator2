import type { Config } from "tailwindcss";

const config: Config = {
    content: [
        "./app/**/*.{js,ts,jsx,tsx,mdx}",
        "./components/**/*.{js,ts,jsx,tsx,mdx}",
    ],
    theme: {
        extend: {
            colors: {
                background: "#0B0B0F",
                card: "#111116",
                "card-border": "#1e1e26",
                accent: "#FF7A18",
                "accent-light": "#FFB347",
            },
            fontFamily: {
                sans: ["var(--font-geist-sans)", "Inter", "system-ui", "sans-serif"],
                mono: ["var(--font-geist-mono)", "monospace"],
            },
            backgroundImage: {
                "accent-gradient": "linear-gradient(135deg, #FF7A18, #FFB347)",
                "hero-glow":
                    "radial-gradient(ellipse 80% 60% at 50% -20%, rgba(255,122,24,0.18) 0%, transparent 70%)",
                "card-gradient":
                    "linear-gradient(135deg, rgba(255,122,24,0.06), rgba(255,122,24,0.01))",
            },
            boxShadow: {
                "accent-glow": "0 0 40px rgba(255,122,24,0.25)",
                "accent-glow-sm": "0 0 20px rgba(255,122,24,0.15)",
                card: "0 1px 3px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.04)",
            },
            animation: {
                "fade-up": "fadeUp 0.6s ease-out forwards",
                pulse: "pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
                "glow-pulse": "glowPulse 4s ease-in-out infinite",
            },
            keyframes: {
                fadeUp: {
                    "0%": { opacity: "0", transform: "translateY(20px)" },
                    "100%": { opacity: "1", transform: "translateY(0)" },
                },
                glowPulse: {
                    "0%, 100%": { opacity: "0.9" },
                    "50%": { opacity: "0.4" },
                },
            },
        },
    },
    plugins: [],
};

export default config;
