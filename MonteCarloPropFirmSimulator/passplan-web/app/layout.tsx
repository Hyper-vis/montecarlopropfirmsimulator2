import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PassPlan — Account Acquisition Engine",
  description:
    "Model prop firm eval stacking, clustered wins, same-day switching, and real-world account acquisition performance.",
  keywords: ["prop firm", "trading", "monte carlo", "pass probability", "challenge", "account acquisition"],
  openGraph: {
    title: "PassPlan — Account Acquisition Engine",
    description:
      "Model prop firm eval stacking, clustered wins, same-day switching, and validation tracking.",
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
