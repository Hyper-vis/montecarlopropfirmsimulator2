import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PassPlan — Prop Firm Probability Engine",
  description:
    "Know your real probability of passing a prop firm challenge. Upload your trade history and run thousands of Monte Carlo simulations.",
  keywords: ["prop firm", "trading", "monte carlo", "pass probability", "challenge"],
  openGraph: {
    title: "PassPlan — Prop Firm Probability Engine",
    description:
      "Know your real probability of passing a prop firm challenge before you risk your capital.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
