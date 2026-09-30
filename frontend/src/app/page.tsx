"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import {
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  CheckCircle2,
  Sparkles,
  ArrowRight,
  RefreshCw,
  Info,
  Server,
  Building,
  Briefcase,
  DollarSign,
  FileText,
  LogIn,
  LogOut,
  User,
  KeyRound
} from "lucide-react";
import { UserProfile, getStoredUser, clearAuth, AUTH_EVENT } from "@/lib/auth";

interface PredictionResult {
  fraud_probability: number;
  risk_level: string;
  is_fraudulent: boolean;
  decision_threshold: number;
  reasons: string[];
  model_name: string;
}

interface SystemHealth {
  status: string;
  model_loaded: boolean;
  version: string;
}

interface ModelInfo {
  model_name: string;
  selected_architecture: string;
  headline_metrics: {
    pr_auc: number;
    f1_score: number;
    recall: number;
    precision: number;
    roc_auc: number;
  };
}

export default function Home() {
  const [apiUrl, setApiUrl] = useState<string>("http://localhost:8000");
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [currentUser, setCurrentUser] = useState<UserProfile | null>(null);

  // Form fields
  const [title, setTitle] = useState("");
  const [companyProfile, setCompanyProfile] = useState("");
  const [description, setDescription] = useState("");
  const [requirements, setRequirements] = useState("");
  const [benefits, setBenefits] = useState("");
  const [salaryRange, setSalaryRange] = useState("");
  const [employmentType, setEmploymentType] = useState("Full-time");
  const [hasLogo, setHasLogo] = useState(true);
  const [telecommuting, setTelecommuting] = useState(false);
  const [hasQuestions, setHasQuestions] = useState(false);

  // Check health and auth on mount
  useEffect(() => {
    const url = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    setApiUrl(url);

    // Initial user profile check
    setCurrentUser(getStoredUser());

    const handleAuthChange = () => {
      setCurrentUser(getStoredUser());
    };
    window.addEventListener(AUTH_EVENT, handleAuthChange);

    const checkSystem = async () => {
      try {
        const res = await fetch(`${url}/health`);
        if (res.ok) {
          const data = await res.json();
          setHealth(data);
        }
        const infoRes = await fetch(`${url}/model-info`);
        if (infoRes.ok) {
          const infoData = await infoRes.json();
          setModelInfo(infoData);
        }
      } catch (e) {
        setHealth({ status: "offline", model_loaded: false, version: "0.1.0" });
      }
    };
    checkSystem();

    return () => {
      window.removeEventListener(AUTH_EVENT, handleAuthChange);
    };
  }, []);

  const loadScamSample = () => {
    setTitle("URGENT WORK FROM HOME DATA ENTRY ASSISTANT!!!");
    setCompanyProfile("");
    setDescription(
      "URGENT HIRE! Make $3,500 weekly working from home! No experience required! You will receive weekly checks and process payments via Western Union wire transfer. Contact our recruiter immediately on Telegram @fast_payroll_desk or email hr.jobs92@gmail.com to start today!"
    );
    setRequirements(
      "Must be 18+ and own a computer. Candidates must send a $120 registration fee and purchase the equipment starter kit via wire transfer prior to onboarding."
    );
    setBenefits("");
    setSalaryRange("150000-180000");
    setEmploymentType("Part-time");
    setHasLogo(false);
    setTelecommuting(true);
    setHasQuestions(false);
    setResult(null);
    setError(null);
  };

  const loadLegitSample = () => {
    setTitle("Senior Backend Systems Engineer");
    setCompanyProfile(
      "Acme Cloud Infrastructure builds next-generation distributed database engines for enterprise customers worldwide. Founded in 2015, we have over 400 employees and offices in New York, London, and Berlin."
    );
    setDescription(
      "We are looking for a Senior Backend Systems Engineer to architect resilient distributed data pipelines. You will lead technical design reviews, optimize latency in Go and Python services, and mentor junior engineers."
    );
    setRequirements(
      "5+ years software engineering experience. Strong proficiency in Python, relational databases (PostgreSQL), Linux internals, and Docker containerization. BS or MS in Computer Science or equivalent practical experience."
    );
    setBenefits(
      "Competitive salary ($140k - $170k), equity grants, 401(k) with 5% match, comprehensive health/dental/vision coverage, and flexible PTO."
    );
    setSalaryRange("140000-170000");
    setEmploymentType("Full-time");
    setHasLogo(true);
    setTelecommuting(true);
    setHasQuestions(true);
    setResult(null);
    setError(null);
  };

  const clearForm = () => {
    setTitle("");
    setCompanyProfile("");
    setDescription("");
    setRequirements("");
    setBenefits("");
    setSalaryRange("");
    setEmploymentType("Full-time");
    setHasLogo(true);
    setTelecommuting(false);
    setHasQuestions(false);
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!title.trim()) {
      setError("Please provide at least a Job Title.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const payload = {
      title,
      company_profile: companyProfile,
      description,
      requirements,
      benefits,
      salary_range: salaryRange || null,
      employment_type: employmentType,
      has_company_logo: hasLogo ? 1 : 0,
      telecommuting: telecommuting ? 1 : 0,
      has_questions: hasQuestions ? 1 : 0,
    };

    try {
      const res = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned ${res.status}: ${res.statusText}`);
      }

      const data: PredictionResult = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to reach JobGuard prediction API.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen pb-16">
      {/* Top Navbar */}
      <header className="border-b border-slate-200 bg-white sticky top-0 z-30 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="p-2 bg-blue-600 rounded-lg text-white shadow-sm">
              <ShieldAlert className="w-5 h-5" />
            </div>
            <div>
              <span className="font-bold text-lg tracking-tight text-slate-900">JobGuard</span>
              <span className="text-blue-600 font-semibold text-lg ml-1">AI</span>
              <span className="ml-2 text-xs font-medium px-2 py-0.5 bg-blue-50 text-blue-700 rounded-full border border-blue-200">
                Fake Job Detector
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3 text-sm">
            <div className="hidden sm:flex items-center gap-1.5 px-3 py-1 bg-slate-100 rounded-full text-slate-700 font-mono text-xs">
              <span
                className={`w-2 h-2 rounded-full ${
                  health?.status === "healthy"
                    ? "bg-emerald-500 animate-pulse"
                    : health?.status === "degraded"
                    ? "bg-amber-500"
                    : "bg-red-500"
                }`}
              />
              {health?.status === "healthy" ? "API Online" : health?.status === "degraded" ? "Model Loading" : "API Offline"}
            </div>
            {modelInfo && (
              <div className="hidden md:flex items-center gap-1 text-xs text-slate-500 mr-1">
                <Briefcase className="w-3.5 h-3.5" />
                <span>PR-AUC: {(modelInfo.headline_metrics.pr_auc * 100).toFixed(1)}%</span>
              </div>
            )}

            {currentUser ? (
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-2 pl-2 pr-3 py-1 bg-slate-100/80 rounded-full border border-slate-200">
                  <div className="w-6 h-6 rounded-full bg-blue-600 text-white text-[11px] font-bold flex items-center justify-center shadow-xs">
                    {currentUser.name.split(" ").map((n) => n[0]).join("")}
                  </div>
                  <div className="text-left hidden sm:block">
                    <p className="text-xs font-semibold text-slate-800 leading-tight">{currentUser.name}</p>
                    <p className="text-[10px] text-slate-500 font-medium leading-none">{currentUser.role}</p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => clearAuth()}
                  className="p-1.5 text-slate-400 hover:text-rose-600 hover:bg-rose-50 rounded-lg transition"
                  title="Sign Out"
                >
                  <LogOut className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <Link
                href="/login"
                className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 hover:bg-blue-600 text-white rounded-lg text-xs font-semibold shadow-sm transition"
              >
                <LogIn className="w-3.5 h-3.5" />
                <span>Analyst Sign In</span>
              </Link>
            )}
          </div>
        </div>
      </header>

      {/* Main Container */}
      <div className="max-w-6xl mx-auto px-4 mt-8">
        {/* Hero Section */}
        <div className="mb-8">
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
            Fake Job Posting Detector & Risk Analyzer
          </h1>
          <p className="text-slate-600 mt-2 max-w-3xl leading-relaxed">
            A production supervised ML system combining sentence-transformer embeddings with domain heuristic feature engineering. 
            Audited for zero train/test leakage and imbalanced precision-recall trade-offs.
          </p>

          {/* Quick Preset Buttons */}
          <div className="mt-4 flex flex-wrap items-center gap-2.5">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider mr-1">Quick Demos:</span>
            <button
              type="button"
              onClick={loadScamSample}
              className="text-xs font-medium px-3 py-1.5 bg-rose-50 text-rose-700 border border-rose-200 rounded-md hover:bg-rose-100 transition-colors flex items-center gap-1.5"
            >
              <AlertTriangle className="w-3.5 h-3.5 text-rose-600" />
              Load Scam Example
            </button>
            <button
              type="button"
              onClick={loadLegitSample}
              className="text-xs font-medium px-3 py-1.5 bg-emerald-50 text-emerald-700 border border-emerald-200 rounded-md hover:bg-emerald-100 transition-colors flex items-center gap-1.5"
            >
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              Load Legitimate Tech Job
            </button>
            <button
              type="button"
              onClick={clearForm}
              className="text-xs font-medium px-3 py-1.5 bg-slate-100 text-slate-600 rounded-md hover:bg-slate-200 transition-colors"
            >
              Clear
            </button>
          </div>
        </div>

        {/* Two Column Layout: Form and Results */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Form: 7 cols */}
          <div className="lg:col-span-7 bg-white rounded-xl border border-slate-200 shadow-sm p-6">
            <form onSubmit={handleAnalyze} className="space-y-4">
              <div>
                <label className="block text-sm font-semibold text-slate-800 mb-1">
                  Job Title <span className="text-rose-500">*</span>
                </label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Senior Software Engineer"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  className="w-full px-3.5 py-2 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-semibold text-slate-800 mb-1">
                  Company Profile / About Us
                  <span className="text-xs font-normal text-slate-500 ml-2">(Missing in ~67% of scam postings)</span>
                </label>
                <textarea
                  rows={2}
                  placeholder="Brief description of the hiring company or leave empty..."
                  value={companyProfile}
                  onChange={(e) => setCompanyProfile(e.target.value)}
                  className="w-full px-3.5 py-2 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-semibold text-slate-800 mb-1">
                  Job Description <span className="text-rose-500">*</span>
                </label>
                <textarea
                  rows={5}
                  required
                  placeholder="Paste the full job posting description here..."
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  className="w-full px-3.5 py-2 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-semibold text-slate-800 mb-1">Candidate Requirements</label>
                  <textarea
                    rows={3}
                    placeholder="Required qualifications, skills, experience..."
                    value={requirements}
                    onChange={(e) => setRequirements(e.target.value)}
                    className="w-full px-3.5 py-2 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-semibold text-slate-800 mb-1">Benefits Package</label>
                  <textarea
                    rows={3}
                    placeholder="401k, health coverage, PTO, equity..."
                    value={benefits}
                    onChange={(e) => setBenefits(e.target.value)}
                    className="w-full px-3.5 py-2 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-1">
                <div>
                  <label className="block text-sm font-semibold text-slate-800 mb-1">Salary Range</label>
                  <input
                    type="text"
                    placeholder="e.g. 80000-120000"
                    value={salaryRange}
                    onChange={(e) => setSalaryRange(e.target.value)}
                    className="w-full px-3.5 py-2 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-semibold text-slate-800 mb-1">Employment Type</label>
                  <select
                    value={employmentType}
                    onChange={(e) => setEmploymentType(e.target.value)}
                    className="w-full px-3.5 py-2 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white"
                  >
                    <option value="Full-time">Full-time</option>
                    <option value="Part-time">Part-time</option>
                    <option value="Contract">Contract</option>
                    <option value="Temporary">Temporary</option>
                    <option value="Unknown">Unknown / Other</option>
                  </select>
                </div>
              </div>

              {/* Checkboxes */}
              <div className="pt-2 flex flex-wrap items-center gap-6 text-sm">
                <label className="flex items-center gap-2 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={hasLogo}
                    onChange={(e) => setHasLogo(e.target.checked)}
                    className="rounded text-blue-600 focus:ring-blue-500 w-4 h-4"
                  />
                  <span className="text-slate-700">Has Company Logo</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={telecommuting}
                    onChange={(e) => setTelecommuting(e.target.checked)}
                    className="rounded text-blue-600 focus:ring-blue-500 w-4 h-4"
                  />
                  <span className="text-slate-700">Remote / Telecommuting</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={hasQuestions}
                    onChange={(e) => setHasQuestions(e.target.checked)}
                    className="rounded text-blue-600 focus:ring-blue-500 w-4 h-4"
                  />
                  <span className="text-slate-700">Screening Questions</span>
                </label>
              </div>

              {/* Submit Button */}
              <div className="pt-4">
                <button
                  type="submit"
                  disabled={loading}
                  className="w-full py-2.5 px-4 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg shadow-sm transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
                >
                  {loading ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      Analyzing Features & Dense Embeddings...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      Analyze Job Posting Risk
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>

          {/* Right Results Panel: 5 cols */}
          <div className="lg:col-span-5 space-y-6">
            {error && (
              <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl text-rose-800 text-sm flex gap-3">
                <AlertTriangle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-rose-900">Analysis Error</h4>
                  <p className="mt-1 text-xs text-rose-700">{error}</p>
                </div>
              </div>
            )}

            {/* Results Card */}
            {result ? (
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-6">
                <div>
                  <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                    Prediction Result
                  </span>
                  <div className="mt-2 flex items-center justify-between">
                    <span
                      className={`text-sm font-bold px-3 py-1 rounded-full uppercase tracking-wide border ${
                        result.risk_level === "High Risk"
                          ? "bg-rose-100 text-rose-800 border-rose-300"
                          : result.risk_level === "Moderate Risk"
                          ? "bg-amber-100 text-amber-800 border-amber-300"
                          : "bg-emerald-100 text-emerald-800 border-emerald-300"
                      }`}
                    >
                      {result.risk_level}
                    </span>
                    <span className="text-xs font-mono text-slate-500">
                      Threshold: {result.decision_threshold.toFixed(2)}
                    </span>
                  </div>
                </div>

                {/* Risk Gauge Bar */}
                <div>
                  <div className="flex justify-between items-baseline mb-1.5">
                    <span className="text-sm font-medium text-slate-700">Fraud Probability</span>
                    <span className="text-2xl font-extrabold text-slate-900 font-mono">
                      {(result.fraud_probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-100 h-3 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all duration-500 ${
                        result.fraud_probability > 0.6
                          ? "bg-rose-600"
                          : result.fraud_probability > 0.3
                          ? "bg-amber-500"
                          : "bg-emerald-500"
                      }`}
                      style={{ width: `${Math.max(4, result.fraud_probability * 100)}%` }}
                    />
                  </div>
                </div>

                {/* Plain-English Reasons */}
                <div>
                  <h4 className="text-sm font-semibold text-slate-900 flex items-center gap-1.5 mb-3">
                    <Info className="w-4 h-4 text-blue-600" />
                    Key Decision Drivers:
                  </h4>
                  <ul className="space-y-2.5">
                    {result.reasons.map((reason, idx) => (
                      <li
                        key={idx}
                        className={`text-xs p-2.5 rounded-lg border flex gap-2.5 items-start ${
                          result.is_fraudulent
                            ? "bg-rose-50/50 border-rose-100 text-slate-800"
                            : "bg-emerald-50/50 border-emerald-100 text-slate-800"
                        }`}
                      >
                        {result.is_fraudulent ? (
                          <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
                        ) : (
                          <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
                        )}
                        <span className="leading-relaxed">{reason}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Model Attribution Footer */}
                <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-xs text-slate-400">
                  <span>Model: {result.model_name}</span>
                  <span>Zero-Leakage Verified</span>
                </div>

                {currentUser && (
                  <div className="p-2.5 bg-blue-50/70 border border-blue-100 rounded-lg flex items-center justify-between text-[11px] text-blue-900">
                    <span className="flex items-center gap-1.5 font-medium">
                      <ShieldCheck className="w-3.5 h-3.5 text-blue-600" />
                      Analyst Session: {currentUser.name}
                    </span>
                    <span className="text-[10px] bg-blue-200/60 px-1.5 py-0.5 rounded text-blue-800 font-mono">
                      {currentUser.role}
                    </span>
                  </div>
                )}
              </div>
            ) : loading ? (
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-8 text-center space-y-4">
                <RefreshCw className="w-8 h-8 text-blue-600 animate-spin mx-auto" />
                <h4 className="font-semibold text-slate-800 text-sm">Evaluating Risk Signals</h4>
                <p className="text-xs text-slate-500 max-w-xs mx-auto">
                  Computing 384-dimensional dense semantic vectors and domain regex heuristic patterns...
                </p>
              </div>
            ) : (
              <div className="bg-white rounded-xl border border-dashed border-slate-300 p-8 text-center text-slate-500 space-y-3">
                <ShieldCheck className="w-10 h-10 text-slate-300 mx-auto" />
                <h4 className="font-semibold text-slate-700 text-sm">Ready to Inspect</h4>
                <p className="text-xs text-slate-500 max-w-xs mx-auto">
                  Paste or fill in a job posting on the left, or click a demo sample above to see the fraud probability and plain-English explanation.
                </p>
              </div>
            )}

            {/* Architecture Card */}
            {modelInfo && (
              <div className="bg-slate-900 text-white rounded-xl p-5 shadow-sm space-y-3 text-xs">
                <div className="flex items-center justify-between">
                  <span className="font-bold text-slate-200 flex items-center gap-1.5">
                    <Server className="w-4 h-4 text-blue-400" />
                    Model Telemetry (Test Holdout)
                  </span>
                  <span className="text-[10px] bg-slate-800 text-slate-400 px-2 py-0.5 rounded">
                    5-Fold Stratified
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 pt-1 font-mono">
                  <div className="bg-slate-800/80 p-2 rounded">
                    <span className="text-slate-400 block text-[10px]">PR-AUC</span>
                    <span className="text-sm font-bold text-blue-400">
                      {(modelInfo.headline_metrics.pr_auc * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="bg-slate-800/80 p-2 rounded">
                    <span className="text-slate-400 block text-[10px]">F1 Score</span>
                    <span className="text-sm font-bold text-emerald-400">
                      {(modelInfo.headline_metrics.f1_score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="bg-slate-800/80 p-2 rounded">
                    <span className="text-slate-400 block text-[10px]">Recall</span>
                    <span className="text-sm font-bold text-amber-400">
                      {(modelInfo.headline_metrics.recall * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="bg-slate-800/80 p-2 rounded">
                    <span className="text-slate-400 block text-[10px]">Precision</span>
                    <span className="text-sm font-bold text-purple-400">
                      {(modelInfo.headline_metrics.precision * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
