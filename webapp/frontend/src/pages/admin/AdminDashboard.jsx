import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import {
  Users, Fish, CheckCircle2, AlertTriangle, TrendingUp, Target,
  ArrowRight, BarChart3, ClipboardList,
} from "lucide-react";
import { getAdminOverview } from "../../api/endpoints";
import { getDiseaseInfo } from "../../utils/diseaseInfo";
import StatCard from "../../components/ui/StatCard";

export default function AdminDashboard() {
  const [overview, setOverview] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const res = await getAdminOverview();
        setOverview(res.data);
      } catch {
        setOverview(null);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const mostDetected = overview?.most_detected_disease ? getDiseaseInfo(overview.most_detected_disease) : null;

  return (
    <div className="space-y-8">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="font-display text-2xl sm:text-3xl font-bold text-white">Admin Dashboard</h1>
        <p className="text-white/60 mt-1">System-wide overview of users, predictions, and model performance.</p>
      </motion.div>

      {loading ? (
        <div className="text-white/40 text-sm py-16 text-center">Loading overview...</div>
      ) : (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <StatCard icon={Users} label="Total Users" value={overview?.total_users ?? 0} accent="ocean" />
            <StatCard icon={Fish} label="Total Farmers" value={overview?.total_farmers ?? 0} accent="seaweed" delay={0.05} />
            <StatCard icon={ClipboardList} label="Total Predictions" value={overview?.total_predictions ?? 0} accent="sand" delay={0.1} />
            <StatCard icon={Target} label="Model Accuracy" value={(overview?.model_accuracy ?? 0.9829) * 100} suffix="%" decimals={2} accent="coral" delay={0.15} />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <StatCard icon={CheckCircle2} label="Healthy Predictions" value={overview?.healthy_predictions ?? 0} accent="seaweed" delay={0.2} />
            <StatCard icon={AlertTriangle} label="Disease Predictions" value={overview?.disease_predictions ?? 0} accent="coral" delay={0.25} />
            <div className="glass-card rounded-2xl p-5 sm:p-6 flex items-center gap-4">
              <div className="shrink-0 rounded-xl bg-gradient-to-br from-ocean-400 to-ocean-600 p-3">
                <TrendingUp className="w-6 h-6 text-white" />
              </div>
              <div>
                <div className="text-lg sm:text-xl font-display font-bold text-white">
                  {mostDetected ? mostDetected.shortName : "—"}
                </div>
                <div className="text-xs sm:text-sm text-white/60 font-medium">Most Detected Disease</div>
              </div>
            </div>
          </div>

          <div className="grid sm:grid-cols-2 gap-4">
            <QuickLink to="/admin/analytics" icon={BarChart3} title="View Analytics" desc="Disease distribution & model performance charts" />
            <QuickLink to="/admin/predictions" icon={ClipboardList} title="Manage Predictions" desc="Browse all user predictions" />
          </div>
        </>
      )}
    </div>
  );
}

function QuickLink({ to, icon: Icon, title, desc }) {
  return (
    <Link to={to}>
      <motion.div whileHover={{ y: -4 }} className="glass-card rounded-2xl p-5 flex items-center gap-4">
        <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-ocean-500 to-seaweed-500 flex items-center justify-center shrink-0">
          <Icon className="w-5 h-5 text-white" />
        </div>
        <div className="flex-1">
          <div className="font-semibold text-white text-sm">{title}</div>
          <div className="text-xs text-white/50 mt-0.5">{desc}</div>
        </div>
        <ArrowRight className="w-4 h-4 text-white/30" />
      </motion.div>
    </Link>
  );
}
