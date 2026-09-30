"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ShieldAlert,
  ShieldCheck,
  Lock,
  Mail,
  ArrowRight,
  ArrowLeft,
  KeyRound,
  CheckCircle2,
  AlertCircle,
  Sparkles,
  UserCheck
} from "lucide-react";
import { storeAuth, getStoredUser } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  const [apiUrl, setApiUrl] = useState("http://localhost:8000");

  useEffect(() => {
    const url = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    setApiUrl(url);

    // If already logged in, redirect to home
    const currentUser = getStoredUser();
    if (currentUser) {
      router.push("/");
    }
  }, [router]);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) {
      setError("Please enter both email and password.");
      return;
    }

    setLoading(true);
    setError(null);
    setSuccessMsg(null);

    try {
      const res = await fetch(`${apiUrl}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || "Authentication failed. Check your credentials.");
      }

      const data = await res.json();
      storeAuth(data.access_token, data.user);
      setSuccessMsg(`Welcome back, ${data.user.name}! Redirecting...`);
      setTimeout(() => {
        router.push("/");
      }, 700);
    } catch (err: any) {
      setError(err.message || "Failed to connect to authentication server.");
    } finally {
      setLoading(false);
    }
  };

  const loginDemoAccount = async (demoEmail: string, demoPass: string) => {
    setEmail(demoEmail);
    setPassword(demoPass);
    setLoading(true);
    setError(null);
    setSuccessMsg(null);

    try {
      const res = await fetch(`${apiUrl}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: demoEmail, password: demoPass }),
      });

      if (!res.ok) {
        throw new Error("Demo login service currently unavailable.");
      }

      const data = await res.json();
      storeAuth(data.access_token, data.user);
      setSuccessMsg(`Signed in as ${data.user.name} (${data.user.role}). Redirecting...`);
      setTimeout(() => {
        router.push("/");
      }, 700);
    } catch (err: any) {
      setError(err.message || "Demo login failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 flex flex-col justify-center py-12 sm:px-6 lg:px-8 relative overflow-hidden">
      {/* Background cyber accent gradients */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-7xl h-96 bg-blue-600/10 blur-3xl pointer-events-none rounded-full" />
      <div className="absolute bottom-0 right-1/4 w-80 h-80 bg-indigo-600/10 blur-3xl pointer-events-none rounded-full" />

      {/* Top security status badge */}
      <div className="sm:mx-auto sm:w-full sm:max-w-md mb-6 px-4 flex justify-center">
        <div className="inline-flex items-center text-xs font-medium text-blue-300 gap-2 bg-slate-800/80 px-4 py-1.5 rounded-full border border-slate-700/80 shadow-sm">
          <ShieldCheck className="w-3.5 h-3.5 text-blue-400" />
          <span>Security Clearance Required</span>
        </div>
      </div>

      <div className="sm:mx-auto sm:w-full sm:max-w-md px-4">
        {/* Brand header */}
        <div className="text-center">
          <div className="inline-flex items-center justify-center p-3 bg-blue-600 rounded-2xl shadow-lg shadow-blue-500/20 mb-4 border border-blue-400/30">
            <ShieldAlert className="w-8 h-8 text-white" />
          </div>
          <h2 className="text-3xl font-extrabold text-white tracking-tight">
            JobGuard <span className="text-blue-500">AI</span>
          </h2>
          <p className="mt-1 text-sm text-slate-400">
            Fraud Operations & Security Analyst Portal
          </p>
        </div>

        {/* Login Card */}
        <div className="mt-8 bg-slate-800/90 border border-slate-700/80 rounded-2xl shadow-2xl p-6 sm:p-8 backdrop-blur-xl">
          {error && (
            <div className="mb-5 p-3.5 bg-red-950/60 border border-red-800/80 rounded-xl flex items-start gap-3 text-red-300 text-sm animate-shake">
              <AlertCircle className="w-5 h-5 flex-shrink-0 text-red-400 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          {successMsg && (
            <div className="mb-5 p-3.5 bg-emerald-950/60 border border-emerald-800/80 rounded-xl flex items-start gap-3 text-emerald-300 text-sm">
              <CheckCircle2 className="w-5 h-5 flex-shrink-0 text-emerald-400 mt-0.5" />
              <span>{successMsg}</span>
            </div>
          )}

          <form onSubmit={handleLogin} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1.5">
                Analyst Work Email
              </label>
              <div className="relative rounded-lg shadow-sm">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                  <Mail className="h-4 w-4" />
                </div>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="analyst@jobguard.ai"
                  required
                  className="w-full pl-10 pr-3 py-2.5 bg-slate-900/90 border border-slate-700 rounded-lg text-white text-sm placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1.5">
                Password
              </label>
              <div className="relative rounded-lg shadow-sm">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                  <Lock className="h-4 w-4" />
                </div>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••••••"
                  required
                  className="w-full pl-10 pr-3 py-2.5 bg-slate-900/90 border border-slate-700 rounded-lg text-white text-sm placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full mt-2 flex items-center justify-center gap-2 py-2.5 px-4 border border-transparent rounded-lg shadow-sm text-sm font-semibold text-white bg-blue-600 hover:bg-blue-500 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 transition"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  <span>Authenticating...</span>
                </>
              ) : (
                <>
                  <span>Sign In to Console</span>
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </form>

          {/* Quick Demo Access Divider */}
          <div className="mt-6 pt-5 border-t border-slate-700/60">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-blue-400" />
                1-Click Demo Accounts
              </span>
              <span className="text-[11px] text-slate-500">Instant Access</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              <button
                type="button"
                onClick={() => loginDemoAccount("analyst@jobguard.ai", "guard_demo2026")}
                disabled={loading}
                className="text-left p-3 rounded-xl bg-slate-900/60 hover:bg-slate-900 border border-slate-700/80 hover:border-blue-500/50 transition group"
              >
                <div className="flex items-center gap-2 mb-1">
                  <div className="p-1 rounded bg-blue-500/10 text-blue-400">
                    <UserCheck className="w-3.5 h-3.5" />
                  </div>
                  <span className="text-xs font-semibold text-white group-hover:text-blue-400 transition">
                    Lead Analyst
                  </span>
                </div>
                <p className="text-[11px] text-slate-400">Alex Morgan</p>
                <p className="text-[10px] text-slate-500 font-mono mt-0.5">analyst@jobguard.ai</p>
              </button>

              <button
                type="button"
                onClick={() => loginDemoAccount("auditor@jobguard.ai", "guard_demo2026")}
                disabled={loading}
                className="text-left p-3 rounded-xl bg-slate-900/60 hover:bg-slate-900 border border-slate-700/80 hover:border-indigo-500/50 transition group"
              >
                <div className="flex items-center gap-2 mb-1">
                  <div className="p-1 rounded bg-indigo-500/10 text-indigo-400">
                    <ShieldCheck className="w-3.5 h-3.5" />
                  </div>
                  <span className="text-xs font-semibold text-white group-hover:text-indigo-400 transition">
                    Compliance Auditor
                  </span>
                </div>
                <p className="text-[11px] text-slate-400">Jordan Lee</p>
                <p className="text-[10px] text-slate-500 font-mono mt-0.5">auditor@jobguard.ai</p>
              </button>
            </div>
          </div>
        </div>

        {/* Security badge footer */}
        <div className="mt-6 text-center text-xs text-slate-500 flex items-center justify-center gap-1.5">
          <KeyRound className="w-3.5 h-3.5 text-slate-400" />
          <span>Role-Based Access Control • HMAC-SHA256 Signed Tokens</span>
        </div>
      </div>
    </div>
  );
}
