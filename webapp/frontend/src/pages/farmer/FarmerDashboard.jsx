import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import {
  ScanLine, History, BookOpen, ArrowRight, TrendingUp, Fish,
  CheckCircle2, AlertTriangle, Sparkles,
} from "lucide-react";
import { useAuth } from "../../context/AuthContext";
import { getMyPredictions } from "../../api/endpoints";
import { getDiseaseInfo } from "../../utils/diseaseInfo";
import { formatDateTime } from "../../utils/format";
import StatCard from "../../components/ui/StatCard";
import OceanBackground from "../../components/OceanBackground";

export default function FarmerDashboard() {
  const { user } = useAuth();
  const [recent, setRecent] = useState([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({ total: 0, healthy: 0, disease: 0 });

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

  return (
    <div className="space-y-8">
      {/* Welcome hero */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative overflow-hidden rounded-3xl glass-card p-8 sm:p-10"
      >
        <OceanBackground fishCount={3} bubbleCount={10} />
        <div className="relative z-10">
          <span className="pill bg-white/10 text-ocean-200 border border-white/10 mb-4">
            <Sparkles className="w-3.5 h-3.5" /> Welcome back
          </span>
          <h1 className="font-display text-2xl sm:text-3xl font-bold text-white">
            Hello, {user?.full_name?.split(" ")[0] || "Farmer"} 👋
          </h1>
          <p className="text-white/60 mt-2 max-w-xl">
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
            <Link to={a.to} className="glass-card rounded-2xl p-5 flex items-center gap-4 h-full">
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
                <div className="font-semibold text-white text-sm">{a.title}</div>
                <div className="text-xs text-white/50 mt-0.5">{a.desc}</div>
              </div>
              <ArrowRight className="w-4 h-4 text-white/30" />
            </Link>
          </motion.div>
        ))}
      </div>

      {/* Recent predictions */}
      <div className="glass-card rounded-2xl p-6">
        <div className="flex items-center justify-between mb-5">
          <h2 className="font-display font-semibold text-white flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-ocean-300" /> Recent Predictions
          </h2>
          <Link to="/dashboard/history" className="text-sm text-ocean-300 hover:text-ocean-200 font-medium">
            View all
          </Link>
        </div>

        {loading ? (
          <div className="text-white/40 text-sm py-8 text-center">Loading...</div>
        ) : recent.length === 0 ? (
          <div className="text-center py-10">
            <Fish className="w-10 h-10 text-white/20 mx-auto mb-3" />
            <p className="text-white/50 text-sm">No predictions yet. Analyze your first fish image!</p>
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
                  className="flex items-center gap-4 rounded-xl px-3 py-3 hover:bg-white/5 transition-colors"
                >
                  <img
                    src={p.image_url}
                    alt=""
                    className="w-12 h-12 rounded-lg object-cover shrink-0 bg-white/5"
                  />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-white truncate">{info.name}</div>
                    <div className="text-xs text-white/40">{formatDateTime(p.created_at)}</div>
                  </div>
                  <div
                    className={`text-sm font-semibold shrink-0 ${
                      info.isHealthy ? "text-seaweed-400" : "text-coral-400"
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
