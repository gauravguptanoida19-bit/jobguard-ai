import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "JobGuard AI - Fake Job Posting Detector",
  description: "Supervised NLP & Tabular Machine Learning System detecting fraudulent employment listings with explainable reasoning.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50 text-slate-900 antialiased">
        {children}
      </body>
    </html>
  );
}
