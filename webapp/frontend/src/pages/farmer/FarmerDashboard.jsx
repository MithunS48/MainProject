import { useEffect, useState, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Link } from "react-router-dom";
import {
  ScanLine, History, BookOpen, ArrowRight, TrendingUp, Fish,
  CheckCircle2, AlertTriangle, Sparkles, Droplets, Thermometer,
  Wifi, WifiOff, ShieldAlert, Lightbulb, Activity, Clock, ShieldCheck,
} from "lucide-react";
import { useAuth } from "../../context/AuthContext";
import { getMyPredictions, getBlynkLatest } from "../../api/endpoints";
import { getDiseaseInfo } from "../../utils/diseaseInfo";
import { getWaterQualityAdvice } from "../../utils/waterQualityAdvisor";
import { formatDateTime } from "../../utils/format";
import StatCard from "../../components/ui/StatCard";
import OceanBackground from "../../components/OceanBackground";

export default function FarmerDashboard() {
  const { user } = useAuth();
  const [recent, setRecent] = useState([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({ total: 0, healthy: 0, disease: 0 });

  // ── IoT water quality state ─────────────────────────────────────────────
  const [iotStatus, setIotStatus] = useState(null);
  const [iotLoading, setIotLoading] = useState(false);
  const [iotError, setIotError] = useState("");
  const iotPollRef = useRef(null);
  const ACTIVE_DEVICE_KEY = "iot_device_id";

  const fetchIoTData = useCallback(async () => {
    setIotLoading(true);
    try {
      const res = await getBlynkLatest();
      // Map Blynk response into the shape WaterQualityWidget expects
      setIotStatus({
        online: true,
        latest: {
          ph:          res.data.ph,
          temperature: res.data.temperature,
          ph_status:   res.data.ph_status,
          temp_status: res.data.temp_status,
          alerts:      res.data.alerts,
          received_at: new Date().toISOString(),
        },
        ph_min:   6.5,
        ph_max:   8.5,
        temp_min: 20.0,
        temp_max: 32.0,
      });
      setIotError("");
    } catch (err) {
      if (err.response?.status === 502) {
        setIotError("ESP32 is offline. Check your device is powered on and connected to Wi-Fi.");
      } else {
        setIotError("Could not fetch sensor data from Blynk Cloud.");
      }
      setIotStatus(null);
    } finally {
      setIotLoading(false);
    }
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const res = await getMyPredictions({ page: 1, page_size: 5, sort: "date_desc" });
        const items = res.data.items || [];
        setRecent(items);

        const allRes = await getMyPredictions({ page: 1, page_size: 100 });
        const all = allRes.data.items || [];
        const healthy = all.filter((p) => p.predicted_class === "healthy" && p.status === "success").length;
        setStats({
          total: allRes.data.total || all.length,
          healthy,
          disease: all.filter((p) => p.status === "success").length - healthy,
        });
      } catch {
        // Non-fatal — dashboard still renders with empty state.
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  // ── Poll IoT data every 30s ──────────────────────────────────────────────
  useEffect(() => {
    fetchIoTData();
    iotPollRef.current = setInterval(fetchIoTData, 30_000);
    return () => clearInterval(iotPollRef.current);
  }, [fetchIoTData]);

  return (
    <div className="space-y-8">
      {/* Welcome hero */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative overflow-hidden rounded-3xl glass-card p-8 sm:p-10 bg-gradient-to-br from-ocean-50 via-white to-seaweed-50/60"
      >
        <OceanBackground fishCount={3} bubbleCount={10} variant="light" />
        <div className="relative z-10">
          <span className="pill bg-ocean-100 text-ocean-700 border border-ocean-200 mb-4">
            <Sparkles className="w-3.5 h-3.5" /> Welcome back
          </span>
          <h1 className="font-display text-2xl sm:text-3xl font-bold text-slate-900">
            Hello, {user?.full_name?.split(" ")[0] || "Farmer"} 👋
          </h1>
          <p className="text-slate-500 mt-2 max-w-xl">
            Keep your fish stock healthy with instant, AI-powered disease detection.
            Upload a photo any time you spot something unusual.
          </p>
          <Link to="/dashboard/detect" className="btn-primary mt-6 inline-flex">
            <ScanLine className="w-4 h-4" /> Analyze a Fish Now <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </motion.div>

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <StatCard icon={Fish} label="Total Analyses" value={stats.total} accent="ocean" />
        <StatCard icon={CheckCircle2} label="Healthy Results" value={stats.healthy} accent="seaweed" delay={0.05} />
        <StatCard icon={AlertTriangle} label="Disease Detected" value={stats.disease} accent="coral" delay={0.1} />
      </div>

      {/* ── Water Quality Live Monitor ────────────────────────────────────── */}
      <WaterQualityWidget
        status={iotStatus}
        loading={iotLoading}
        error={iotError}
      />

      {/* Quick actions */}
      <div className="grid sm:grid-cols-3 gap-4">
        {[
          { to: "/dashboard/detect", icon: ScanLine, title: "Detect Disease", desc: "Upload a new photo for AI analysis", accent: "ocean" },
          { to: "/dashboard/history", icon: History, title: "Prediction History", desc: "Review your past analyses", accent: "seaweed" },
          { to: "/dashboard/diseases", icon: BookOpen, title: "Disease Information", desc: "Learn about each condition", accent: "coral" },
        ].map((a, i) => (
          <motion.div
            key={a.to}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 + i * 0.08 }}
            whileHover={{ y: -4 }}
          >
            <Link to={a.to} className="glass-card hover:shadow-md transition-shadow rounded-2xl p-5 flex items-center gap-4 h-full">
              <div
                className={`w-11 h-11 rounded-xl flex items-center justify-center shrink-0 bg-gradient-to-br ${
                  a.accent === "ocean"
                    ? "from-ocean-400 to-ocean-600"
                    : a.accent === "seaweed"
                    ? "from-seaweed-400 to-seaweed-600"
                    : "from-coral-400 to-coral-600"
                }`}
              >
                <a.icon className="w-5 h-5 text-white" />
              </div>
              <div className="flex-1">
                <div className="font-semibold text-slate-900 text-sm">{a.title}</div>
                <div className="text-xs text-slate-500 mt-0.5">{a.desc}</div>
              </div>
              <ArrowRight className="w-4 h-4 text-slate-300" />
            </Link>
          </motion.div>
        ))}
      </div>

      {/* Recent predictions */}
      <div className="glass-card rounded-2xl p-6">
        <div className="flex items-center justify-between mb-5">
          <h2 className="font-display font-semibold text-slate-900 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-ocean-500" /> Recent Predictions
          </h2>
          <Link to="/dashboard/history" className="text-sm text-ocean-600 hover:text-ocean-700 font-medium">
            View all
          </Link>
        </div>

        {loading ? (
          <div className="text-slate-400 text-sm py-8 text-center">Loading...</div>
        ) : recent.length === 0 ? (
          <div className="text-center py-10">
            <Fish className="w-10 h-10 text-slate-200 mx-auto mb-3" />
            <p className="text-slate-500 text-sm">No predictions yet. Analyze your first fish image!</p>
            <Link to="/dashboard/detect" className="btn-secondary mt-4 inline-flex text-sm">
              <ScanLine className="w-4 h-4" /> Get Started
            </Link>
          </div>
        ) : (
          <div className="space-y-2">
            {recent.map((p) => {
              const info = getDiseaseInfo(p.predicted_class);
              return (
                <div
                  key={p.id}
                  className="flex items-center gap-4 rounded-xl px-3 py-3 hover:bg-slate-50 transition-colors"
                >
                  <img
                    src={p.image_url}
                    alt=""
                    className="w-12 h-12 rounded-lg object-cover shrink-0 bg-slate-100"
                  />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-slate-900 truncate">{info.name}</div>
                    <div className="text-xs text-slate-400">{formatDateTime(p.created_at)}</div>
                  </div>
                  <div
                    className={`text-sm font-semibold shrink-0 ${
                      info.isHealthy ? "text-seaweed-600" : "text-coral-600"
                    }`}
                  >
                    {p.confidence_pct}%
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// WATER QUALITY WIDGET
// ════════════════════════════════════════════════════════════════════════════

const LEVEL_STYLES = {
  normal:  { bg: "bg-green-50",   border: "border-green-200",   text: "text-green-700",   badge: "bg-green-100 text-green-700" },
  warning: { bg: "bg-amber-50",   border: "border-amber-200",   text: "text-amber-700",   badge: "bg-amber-100 text-amber-700" },
  danger:  { bg: "bg-red-50",     border: "border-red-200",     text: "text-red-700",     badge: "bg-red-100 text-red-700" },
};

const LEVEL_ICONS = {
  normal:  ShieldCheck,
  warning: AlertTriangle,
  danger:  ShieldAlert,
};

function WaterQualityWidget({ status, loading, error }) {
  const r = status?.latest;
  const advice = r ? getWaterQualityAdvice(r.ph, r.temperature) : null;

  // ── No device registered ──────────────────────────────────────────────
  if (error && !status) {
    return (
      <div className="glass-card rounded-2xl p-6">
        <div className="flex items-center gap-2 mb-2">
          <Droplets className="w-5 h-5 text-ocean-500" />
          <h2 className="font-display font-semibold text-slate-900">Water Quality Monitor</h2>
        </div>
        <div className="flex items-center gap-3 text-sm text-slate-500">
          <AlertTriangle className="w-4 h-4 text-amber-400" />
          {error}
          <Link to="/dashboard/water-quality" className="text-ocean-600 hover:text-ocean-700 font-medium ml-1">
            Set up →
          </Link>
        </div>
      </div>
    );
  }

  // ── Loading state ─────────────────────────────────────────────────────
  if (loading && !status) {
    return (
      <div className="glass-card rounded-2xl p-6">
        <div className="flex items-center gap-2 mb-4">
          <Droplets className="w-5 h-5 text-ocean-500" />
          <h2 className="font-display font-semibold text-slate-900">Water Quality Monitor</h2>
        </div>
        <div className="grid sm:grid-cols-2 gap-4">
          {[0, 1].map((i) => (
            <div key={i} className="h-28 rounded-xl bg-slate-100 animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  // ── No data yet ────────────────────────────────────────────────────────
  if (!status || !r) {
    return (
      <div className="glass-card rounded-2xl p-6">
        <div className="flex items-center gap-2 mb-2">
          <Droplets className="w-5 h-5 text-ocean-500" />
          <h2 className="font-display font-semibold text-slate-900">Water Quality Monitor</h2>
        </div>
        <p className="text-sm text-slate-500">
          No sensor data yet. Register your ESP32 device on the{" "}
          <Link to="/dashboard/water-quality" className="text-ocean-600 hover:text-ocean-700 font-medium">
            Water Quality
          </Link>{" "}
          page.
        </p>
      </div>
    );
  }

  const overallStyle = LEVEL_STYLES[advice.overallLevel];
  const OverallIcon  = LEVEL_ICONS[advice.overallLevel];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card rounded-2xl p-6 space-y-5"
    >
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-2">
          <Droplets className="w-5 h-5 text-ocean-500" />
          <h2 className="font-display font-semibold text-slate-900">Water Quality Monitor</h2>
        </div>
        <div className="flex items-center gap-3">
          {status.online ? (
            <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-green-600">
              <Wifi className="w-3.5 h-3.5" /> Online
            </span>
          ) : (
            <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-slate-400">
              <WifiOff className="w-3.5 h-3.5" /> Offline
            </span>
          )}
          <span className="text-xs text-slate-400 flex items-center gap-1">
            <Clock className="w-3 h-3" />
            {r.device_ts ? formatDateTime(r.device_ts) : formatDateTime(r.received_at)}
          </span>
        </div>
      </div>

      {/* ── Overall Status Banner ───────────────────────────────────────── */}
      <div className={`rounded-xl px-4 py-3 flex items-center gap-3 ${overallStyle.bg} ${overallStyle.border} border`}>
        <OverallIcon className={`w-5 h-5 ${overallStyle.text}`} />
        <div>
          <span className={`text-sm font-semibold ${overallStyle.text}`}>
            {advice.overallLevel === "normal"
              ? "Water conditions are optimal — fish are safe."
              : advice.overallLevel === "warning"
              ? "Water conditions need attention — fish are under stress."
              : "Critical water conditions — immediate action required!"}
          </span>
        </div>
        <span className={`ml-auto text-xs px-2.5 py-1 rounded-full font-semibold ${overallStyle.badge}`}>
          {advice.overallLevel.toUpperCase()}
        </span>
      </div>

      {/* ── pH + Temp Cards ─────────────────────────────────────────────── */}
      <div className="grid sm:grid-cols-2 gap-4">
        <SensorCard
          icon={Droplets}
          title="pH Level"
          value={r.ph?.toFixed(2)}
          advice={advice.ph}
        />
        <SensorCard
          icon={Thermometer}
          title="Temperature"
          value={`${r.temperature?.toFixed(1)}°C`}
          advice={advice.temp}
        />
      </div>

      {/* ── Fish Impact ────────────────────────────────────────────────── */}
      {advice.hasIssues && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-2">
            <Fish className="w-4 h-4 text-slate-400" /> What's happening to your fish
          </h3>
          {advice.ph?.fishImpact?.length > 0 && advice.ph.level !== "normal" && (
            <div className="rounded-xl bg-slate-50 border border-slate-100 p-3 space-y-1">
              <span className="text-xs font-semibold text-slate-500">pH — {advice.ph.label}</span>
              <ul className="space-y-1">
                {advice.ph.fishImpact.map((impact, i) => (
                  <li key={i} className="text-sm text-slate-600 flex items-start gap-2">
                    <span className="text-slate-300 mt-0.5">•</span> {impact}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {advice.temp?.fishImpact?.length > 0 && advice.temp.level !== "normal" && (
            <div className="rounded-xl bg-slate-50 border border-slate-100 p-3 space-y-1">
              <span className="text-xs font-semibold text-slate-500">Temperature — {advice.temp.label}</span>
              <ul className="space-y-1">
                {advice.temp.fishImpact.map((impact, i) => (
                  <li key={i} className="text-sm text-slate-600 flex items-start gap-2">
                    <span className="text-slate-300 mt-0.5">•</span> {impact}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* ── Recommendations ─────────────────────────────────────────────── */}
      {advice.hasIssues && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-2">
            <Lightbulb className="w-4 h-4 text-amber-400" /> Recommended Actions
          </h3>
          {advice.ph?.recommendations?.length > 0 && advice.ph.level !== "normal" && (
            <div className="rounded-xl bg-ocean-50 border border-ocean-100 p-3 space-y-1">
              <span className="text-xs font-semibold text-ocean-600">Fix pH — {advice.ph.label}</span>
              <ul className="space-y-1">
                {advice.ph.recommendations.map((rec, i) => (
                  <li key={i} className="text-sm text-slate-600 flex items-start gap-2">
                    <span className="text-ocean-300 mt-0.5">•</span> {rec}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {advice.temp?.recommendations?.length > 0 && advice.temp.level !== "normal" && (
            <div className="rounded-xl bg-orange-50 border border-orange-100 p-3 space-y-1">
              <span className="text-xs font-semibold text-orange-600">Fix Temperature — {advice.temp.label}</span>
              <ul className="space-y-1">
                {advice.temp.recommendations.map((rec, i) => (
                  <li key={i} className="text-sm text-slate-600 flex items-start gap-2">
                    <span className="text-orange-300 mt-0.5">•</span> {rec}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* ── Disease Risks ──────────────────────────────────────────────── */}
      {advice.diseaseRisks.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-coral-400" /> Likely Diseases in These Conditions
          </h3>
          <div className="flex flex-wrap gap-2">
            {advice.diseaseRisks.map((disease, i) => (
              <span
                key={i}
                className="text-xs font-medium px-3 py-1.5 rounded-full bg-coral-50 text-coral-700 border border-coral-100"
              >
                {disease}
              </span>
            ))}
          </div>
          <p className="text-xs text-slate-400">
            These diseases are more likely to appear if current conditions persist. Monitor your fish closely for symptoms and consider using the disease detection tool.
          </p>
        </div>
      )}

      {/* ── Footer link ────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between pt-2 border-t border-slate-100">
        <span className="text-xs text-slate-400 flex items-center gap-1">
          <Activity className="w-3 h-3" /> Auto-refresh every 30s
        </span>
        <Link to="/dashboard/water-quality" className="text-sm text-ocean-600 hover:text-ocean-700 font-medium">
          Detailed view →
        </Link>
      </div>
    </motion.div>
  );
}

// ─── Individual Sensor Card ───────────────────────────────────────────────────
function SensorCard({ icon: Icon, title, value, advice }) {
  if (!advice) return null;
  const style = LEVEL_STYLES[advice.level];

  return (
    <div className={`rounded-xl p-4 border ${style.border} ${style.bg}`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2 text-slate-500 text-sm font-medium">
          <Icon className="w-4 h-4" /> {title}
        </div>
        <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${style.badge}`}>
          {advice.label}
        </span>
      </div>
      <div className="text-3xl font-extrabold text-slate-800">
        {value}
      </div>
    </div>
  );
}
