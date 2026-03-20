import { useState, useEffect } from "react";
import "./App.css";
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  AreaChart, Area,
} from "recharts";

// Empty string = relative URLs — works for both local and Vercel
const API = process.env.REACT_APP_API_URL || "";

function Card({ title, children }) {
  return (
    <div className="card">
      {title && <h3 className="card-title">{title}</h3>}
      {children}
    </div>
  );
}

function StatCard({ label, value, sub, color }) {
  return (
    <div className="stat-card" style={{ borderTop: `3px solid ${color}` }}>
      <div className="stat-value" style={{ color }}>{value}</div>
      <div className="stat-label">{label}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

function Spinner() {
  return <div className="spinner" />;
}

export default function App() {
  const [portfolio,    setPortfolio]    = useState(null);
  const [expectedLoss, setExpectedLoss] = useState(null);
  const [varData,      setVarData]      = useState(null);
  const [lossDist,     setLossDist]     = useState(null);
  const [featureImp,   setFeatureImp]   = useState(null);
  const [performance,  setPerformance]  = useState(null);
  const [summary,      setSummary]      = useState("");
  const [loadingSummary, setLoadingSummary] = useState(false);
  const [loading,      setLoading]      = useState(true);
  const [error,        setError]        = useState(null);

  useEffect(() => {
    async function fetchAll() {
      try {
        const [p, el, v, ld, fi, perf] = await Promise.all([
          fetch(`${API}/api/portfolio`).then(r => r.json()),
          fetch(`${API}/api/expected-loss`).then(r => r.json()),
          fetch(`${API}/api/var`).then(r => r.json()),
          fetch(`${API}/api/loss-distribution`).then(r => r.json()),
          fetch(`${API}/api/feature-importance`).then(r => r.json()),
          fetch(`${API}/api/performance`).then(r => r.json()),
        ]);
        setPortfolio(p);
        setExpectedLoss(el);
        setVarData(v);
        setLossDist(ld);
        setFeatureImp(fi);
        setPerformance(perf);
      } catch (e) {
        setError("Failed to load data. Make sure the backend is running.");
      } finally {
        setLoading(false);
      }
    }
    fetchAll();
  }, []);

  async function generateSummary() {
    setLoadingSummary(true);
    setSummary("");
    try {
      const res  = await fetch(`${API}/api/summary`);
      const data = await res.json();
      setSummary(data.summary);
    } catch {
      setSummary("Failed to generate summary. Check your GEMINI_API_KEY.");
    } finally {
      setLoadingSummary(false);
    }
  }

  if (loading) return (
    <div className="loading-screen">
      <Spinner />
      <p>Running ML pipeline...</p>
    </div>
  );

  if (error) return <div className="error-screen">{error}</div>;

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div>
            <h1 className="header-title">💼 Credit Risk Intelligence Dashboard</h1>
            <p className="header-sub">
              Portfolio of <strong>{portfolio?.total}</strong> borrowers ·
              German Credit Dataset · Random Forest AUC&nbsp;
              <strong>{performance?.bestAuc}</strong>
            </p>
          </div>
          <div className="model-badges">
            {performance?.models.map(m => (
              <span key={m.model} className="badge">
                {m.model}: <strong>{m.auc}</strong>
              </span>
            ))}
          </div>
        </div>
      </header>

      <main className="main">
        <div className="stats-row">
          <StatCard label="Expected Loss / Borrower" value={`$${varData?.expectedLoss?.toLocaleString()}`} sub="Average per borrower" color="#6366f1" />
          <StatCard label="Value-at-Risk (95%)" value={`$${varData?.var95?.toLocaleString()}`} sub="Max loss at 95% confidence" color="#f59e0b" />
          <StatCard label="Value-at-Risk (99%)" value={`$${varData?.var99?.toLocaleString()}`} sub="Max loss at 99% confidence" color="#ef4444" />
          <StatCard label="High Risk Borrowers" value={`${portfolio?.riskBreakdown?.find(t => t.tier === "High Risk")?.percentage}%`} sub="Require immediate attention" color="#ef4444" />
        </div>

        <div className="grid-2">
          <Card title="Portfolio Risk Breakdown">
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie data={portfolio?.riskBreakdown} dataKey="count" nameKey="tier" cx="50%" cy="50%" outerRadius={100} label={({ tier, percentage }) => `${tier}: ${percentage}%`}>
                  {portfolio?.riskBreakdown.map(entry => <Cell key={entry.tier} fill={entry.color} />)}
                </Pie>
                <Tooltip formatter={(v, name) => [`${v} borrowers`, name]} />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Expected Loss by Risk Tier (USD)">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={expectedLoss} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="tier" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip formatter={v => [`$${v.toLocaleString()}`, "Avg EL"]} />
                <Bar dataKey="expectedLoss" radius={[4, 4, 0, 0]}>
                  {expectedLoss?.map(entry => <Cell key={entry.tier} fill={entry.color} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>

        <div className="grid-2">
          <Card title="Monte Carlo Loss Distribution (1,000 Scenarios)">
            <p className="chart-sub">
              VaR 95%: <strong style={{color:"#f59e0b"}}>${lossDist?.var95?.toLocaleString()}</strong>
              &nbsp;·&nbsp;
              VaR 99%: <strong style={{color:"#ef4444"}}>${lossDist?.var99?.toLocaleString()}</strong>
            </p>
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={lossDist?.data} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                <defs>
                  <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#6366f1" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0}   />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="simulation" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={v => [`$${v.toLocaleString()}`, "Portfolio Loss"]} />
                <Area type="monotone" dataKey="loss" stroke="#6366f1" fill="url(#lossGrad)" strokeWidth={2} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Top 10 Risk Drivers (Feature Importance)">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={featureImp} layout="vertical" margin={{ top: 5, right: 20, left: 90, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="feature" type="category" tick={{ fontSize: 11 }} width={90} />
                <Tooltip formatter={v => [v.toFixed(4), "Importance"]} />
                <Bar dataKey="importance" fill="#6366f1" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>

        <Card title="🤖 AI Executive Summary (Gemini)">
          <p className="card-sub">Generate a board-level credit risk summary from the portfolio data above.</p>
          <button className="btn-primary" onClick={generateSummary} disabled={loadingSummary}>
            {loadingSummary ? "Generating..." : "Generate Executive Summary"}
          </button>
          {loadingSummary && <div className="summary-loading"><Spinner /><span>Analysing portfolio with Gemini...</span></div>}
          {summary && (
            <div className="summary-box">
              {summary.split("\n").map((para, i) => para.trim() ? <p key={i}>{para}</p> : null)}
            </div>
          )}
        </Card>
      </main>

      <footer className="footer">
        Built by Sanvi Udhan · BITS Pilani &#x27;27 ·&nbsp;
        <a href="https://github.com/Sanu700/AI-Credit-Risk-Intelligence" target="_blank" rel="noreferrer">GitHub</a>
      </footer>
    </div>
  );
}
