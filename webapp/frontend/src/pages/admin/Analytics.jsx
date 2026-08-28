import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
} from "recharts";
import { BarChart3, PieChart as PieIcon, Target } from "lucide-react";
import { getAdminAnalytics } from "../../api/endpoints";
import { getDiseaseInfo } from "../../utils/diseaseInfo";

const COLOR_HEX = {
  coral: "#ff6f4f",
  seaweed: "#40d287",
  sand: "#e3b455",
  ocean: "#46c9ff",
};

export default function Analytics() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const res = await getAdminAnalytics();
        setData(res.data);
      } catch {
        setData(null);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const distribution = data?.live_disease_distribution || {};
  const pieData = Object.entries(distribution).map(([key, count]) => {
    const info = getDiseaseInfo(key);
    return { name: info.shortName, value: count, color: COLOR_HEX[info.color] || "#46c9ff" };
  });

  const perf = data?.model_performance || { accuracy: 0.9829, f1_score: 0.9829, auc: 0.9989 };
  const perfData = [
    { name: "Accuracy", value: perf.accuracy * 100 },
    { name: "F1-Score", value: perf.f1_score * 100 },
    { name: "AUC", value: perf.auc * 100 },
  ];

  const totalPredictions = pieData.reduce((s, d) => s + d.value, 0);

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="font-display text-2xl sm:text-3xl font-bold text-slate-900">Analytics</h1>
        <p className="text-slate-500 mt-1">Disease distribution across all predictions and overall model performance.</p>
      </motion.div>

      {loading ? (
        <div className="text-slate-400 text-sm py-16 text-center">Loading analytics...</div>
      ) : (
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Disease distribution pie */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass-card rounded-2xl p-6">
            <h2 className="font-display font-semibold text-slate-900 mb-1 flex items-center gap-2">
              <PieIcon className="w-4 h-4 text-ocean-500" /> Disease Distribution
            </h2>
            <p className="text-xs text-slate-400 mb-4">Live counts from all predictions made on the platform ({totalPredictions} total)</p>
            {totalPredictions === 0 ? (
              <p className="text-slate-400 text-sm text-center py-16">No predictions recorded yet.</p>
            ) : (
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie
                    data={pieData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={95}
                    paddingAngle={3}
                    animationDuration={800}
                  >
                    {pieData.map((entry, i) => (
                      <Cell key={i} fill={entry.color} stroke="none" />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid #e2e8f0", borderRadius: 12, color: "#1b2b3a", boxShadow: "0 4px 16px rgba(15,60,90,0.1)" }} />
                  <Legend wrapperStyle={{ color: "#475569", fontSize: 12 }} />
                </PieChart>
              </ResponsiveContainer>
            )}
          </motion.div>

          {/* Model performance bar */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="glass-card rounded-2xl p-6">
            <h2 className="font-display font-semibold text-slate-900 mb-1 flex items-center gap-2">
              <Target className="w-4 h-4 text-seaweed-600" /> Model Performance
            </h2>
            <p className="text-xs text-slate-400 mb-4">Final model (MobileNetV2 + ConvNeXt + Polynomial SVM) — test set results</p>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={perfData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} />
                <YAxis domain={[90, 100]} stroke="#94a3b8" fontSize={12} unit="%" />
                <Tooltip
                  contentStyle={{ background: "#ffffff", border: "1px solid #e2e8f0", borderRadius: 12, color: "#1b2b3a", boxShadow: "0 4px 16px rgba(15,60,90,0.1)" }}
                  formatter={(v) => `${v.toFixed(2)}%`}
                />
                <Bar dataKey="value" radius={[8, 8, 0, 0]} animationDuration={800}>
                  <Cell fill="#1cabf2" />
                  <Cell fill="#1ab469" />
                  <Cell fill="#ff6f4f" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </motion.div>
        </div>
      )}

      {/* Summary stat strip */}
      <div className="grid grid-cols-3 gap-4">
        {perfData.map((p) => (
          <div key={p.name} className="glass-card rounded-2xl p-5 text-center">
            <div className="text-2xl font-display font-bold text-slate-900">{p.value.toFixed(2)}%</div>
            <div className="text-xs text-slate-500 mt-1">{p.name}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
