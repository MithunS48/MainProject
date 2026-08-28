import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  FlaskConical, Award, Layers, GitCompare, Sigma, Grid3x3, LineChart as LineChartIcon,
} from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  LineChart, Line, Legend,
} from "recharts";
import {
  getCnnComparison, getFusionResults, getPcaComparison, getKernelComparison,
  getConfusionMatrix, getRocAuc,
} from "../../api/endpoints";
import { getDiseaseInfo } from "../../utils/diseaseInfo";

const TABS = [
  { key: "cnn", label: "Individual CNNs", icon: Award },
  { key: "fusion", label: "Fusion Results", icon: Layers },
  { key: "pca", label: "PCA Comparison", icon: GitCompare },
  { key: "kernel", label: "SVM Kernel Comparison", icon: Sigma },
  { key: "confusion", label: "Confusion Matrix", icon: Grid3x3 },
  { key: "roc", label: "ROC / AUC", icon: LineChartIcon },
];

export default function ResearchResults() {
  const [tab, setTab] = useState("cnn");
  const [cnn, setCnn] = useState([]);
  const [fusion, setFusion] = useState([]);
  const [pca, setPca] = useState([]);
  const [kernel, setKernel] = useState([]);
  const [confusion, setConfusion] = useState(null);
  const [roc, setRoc] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const [c, f, p, k, cm, r] = await Promise.all([
          getCnnComparison(), getFusionResults(), getPcaComparison(),
          getKernelComparison(), getConfusionMatrix(), getRocAuc(),
        ]);
        setCnn(c.data);
        setFusion(f.data);
        setPca(p.data);
        setKernel(k.data);
        setConfusion(cm.data);
        setRoc(r.data);
      } catch {
        // leave defaults
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="font-display text-2xl sm:text-3xl font-bold text-white">Research Results</h1>
        <p className="text-white/60 mt-1">
          Full experimental comparison behind the final deployed model — individual CNNs, feature
          fusion, PCA, and SVM kernel selection.
        </p>
      </motion.div>

      {/* Tabs */}
      <div className="flex gap-2 overflow-x-auto pb-1">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium whitespace-nowrap transition-all ${
              tab === t.key
                ? "bg-gradient-to-r from-ocean-500 to-seaweed-500 text-white shadow-glow-ocean"
                : "bg-white/5 text-white/60 hover:bg-white/10"
            }`}
          >
            <t.icon className="w-4 h-4" /> {t.label}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="text-white/40 text-sm py-16 text-center">Loading research data...</div>
      ) : (
        <AnimatePresence mode="wait">
          <motion.div key={tab} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
            {tab === "cnn" && <CnnTab data={cnn} />}
            {tab === "fusion" && <FusionTab data={fusion} />}
            {tab === "pca" && <PcaTab data={pca} />}
            {tab === "kernel" && <KernelTab data={kernel} />}
            {tab === "confusion" && <ConfusionTab data={confusion} />}
            {tab === "roc" && <RocTab data={roc} />}
          </motion.div>
        </AnimatePresence>
      )}
    </div>
  );
}

function pct(v) {
  return `${(v * 100).toFixed(2)}%`;
}

function CnnTab({ data }) {
  return (
    <div className="glass-card rounded-2xl overflow-hidden">
      <div className="p-6 pb-0">
        <h3 className="font-display font-semibold text-white mb-1">Individual CNN Performance</h3>
        <p className="text-xs text-white/40 mb-4">Test-set results for each CNN backbone, trained independently, before fusion.</p>
      </div>
      <table className="w-full">
        <thead>
          <tr className="border-b border-white/10 text-left">
            <th className="px-6 py-3 text-xs font-semibold text-white/40 uppercase">Model</th>
            <th className="px-6 py-3 text-xs font-semibold text-white/40 uppercase">Accuracy</th>
            <th className="px-6 py-3 text-xs font-semibold text-white/40 uppercase hidden sm:table-cell">Precision</th>
            <th className="px-6 py-3 text-xs font-semibold text-white/40 uppercase hidden sm:table-cell">Recall</th>
            <th className="px-6 py-3 text-xs font-semibold text-white/40 uppercase hidden md:table-cell">F1-Score</th>
          </tr>
        </thead>
        <tbody>
          {data.map((m) => (
            <tr key={m.model} className={`border-b border-white/5 ${m.is_final ? "bg-seaweed-500/10" : ""}`}>
              <td className="px-6 py-3.5 text-sm font-medium text-white flex items-center gap-2">
                {m.model}
                {m.is_final && <span className="pill bg-seaweed-500/20 text-seaweed-300 text-[10px]">FINAL MODEL</span>}
              </td>
              <td className="px-6 py-3.5 text-sm font-semibold text-ocean-300">{pct(m.test_accuracy)}</td>
              <td className="px-6 py-3.5 text-sm text-white/70 hidden sm:table-cell">{pct(m.test_precision)}</td>
              <td className="px-6 py-3.5 text-sm text-white/70 hidden sm:table-cell">{pct(m.test_recall)}</td>
              <td className="px-6 py-3.5 text-sm text-white/70 hidden md:table-cell">{pct(m.test_f1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FusionTab({ data }) {
  const chartData = data.map((d) => ({ name: d.combination.replace(/_plus_/g, "+"), accuracy: d.accuracy * 100, highlighted: d.is_highlighted }));
  return (
    <div className="space-y-6">
      <div className="glass-card rounded-2xl p-6">
        <h3 className="font-display font-semibold text-white mb-1">All Feature-Fusion Combinations</h3>
        <p className="text-xs text-white/40 mb-4">
          Every combination of CNN feature vectors tested with an SVM classifier (before final kernel tuning).
          <span className="text-seaweed-300 font-medium"> MobileNetV2 + ConvNeXt</span> gave the best fusion result at 97.77%.
        </p>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" vertical={false} />
            <XAxis dataKey="name" stroke="rgba(255,255,255,0.4)" fontSize={11} angle={-25} textAnchor="end" interval={0} height={60} />
            <YAxis domain={[90, 100]} stroke="rgba(255,255,255,0.4)" fontSize={12} unit="%" />
            <Tooltip contentStyle={{ background: "#0b3049", border: "1px solid rgba(255,255,255,0.15)", borderRadius: 12, color: "#fff" }} formatter={(v) => `${v.toFixed(2)}%`} />
            <Bar dataKey="accuracy" radius={[8, 8, 0, 0]}>
              {chartData.map((entry, i) => (
                <Cell key={i} fill={entry.highlighted ? "#1ab469" : "#1cabf2"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-seaweed-500/10 border border-seaweed-500/25 rounded-xl px-5 py-4 text-sm text-white/70">
        <strong className="text-seaweed-300">MobileNetV2 + ConvNeXt (97.77%)</strong> was selected as the final
        feature-fusion combination. Applying a Polynomial-kernel SVM on top of this fusion further improved
        accuracy to <strong className="text-white">98.29%</strong> — the final deployed model.
      </div>
    </div>
  );
}

function PcaTab({ data }) {
  // Group by combination
  const groups = {};
  data.forEach((d) => {
    groups[d.combination] = groups[d.combination] || {};
    groups[d.combination][d.experiment] = d;
  });

  return (
    <div className="space-y-6">
      <div className="bg-ocean-500/10 border border-ocean-500/25 rounded-xl px-5 py-4 text-sm text-white/70">
        Across every tested combination, models trained <strong className="text-white">WITHOUT PCA</strong>{" "}
        consistently outperformed their PCA-reduced counterparts. The final deployed system therefore uses
        the <strong className="text-white">full, un-reduced 2,048-d fused feature vector</strong> — no PCA step.
      </div>
      <div className="glass-card rounded-2xl overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="border-b border-white/10 text-left">
              <th className="px-6 py-3 text-xs font-semibold text-white/40 uppercase">Combination</th>
              <th className="px-6 py-3 text-xs font-semibold text-white/40 uppercase">No PCA Accuracy</th>
              <th className="px-6 py-3 text-xs font-semibold text-white/40 uppercase">With PCA Accuracy</th>
              <th className="px-6 py-3 text-xs font-semibold text-white/40 uppercase hidden sm:table-cell">PCA Dim</th>
              <th className="px-6 py-3 text-xs font-semibold text-white/40 uppercase hidden md:table-cell">Variance Retained</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(groups).map(([combo, exp]) => {
              const noPca = exp.NO_PCA;
              const withPca = exp.WITH_PCA;
              return (
                <tr key={combo} className="border-b border-white/5">
                  <td className="px-6 py-3.5 text-sm font-medium text-white">{combo.replace(/_plus_/g, "+")}</td>
                  <td className="px-6 py-3.5 text-sm font-semibold text-seaweed-400">{noPca ? pct(noPca.accuracy) : "—"}</td>
                  <td className="px-6 py-3.5 text-sm text-white/60">{withPca ? pct(withPca.accuracy) : "—"}</td>
                  <td className="px-6 py-3.5 text-sm text-white/60 hidden sm:table-cell">{withPca ? `${withPca.pca_dim}-d` : "—"}</td>
                  <td className="px-6 py-3.5 text-sm text-white/60 hidden md:table-cell">{withPca ? `${withPca.variance_retained_pct.toFixed(1)}%` : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function KernelTab({ data }) {
  const groups = {};
  data.forEach((d) => {
    groups[d.combination] = groups[d.combination] || {};
    groups[d.combination][d.kernel] = d;
  });

  return (
    <div className="space-y-6">
      <div className="glass-card rounded-2xl overflow-hidden">
        <div className="p-6 pb-0">
          <h3 className="font-display font-semibold text-white mb-1">SVM Kernel Comparison</h3>
          <p className="text-xs text-white/40 mb-4">Per-combination accuracy across Linear, RBF, and Polynomial kernels.</p>
        </div>
        <table className="w-full">
          <thead>
            <tr className="border-b border-white/10 text-left">
              <th className="px-6 py-3 text-xs font-semibold text-white/40 uppercase">Combination</th>
              <th className="px-6 py-3 text-xs font-semibold text-white/40 uppercase">Linear</th>
              <th className="px-6 py-3 text-xs font-semibold text-white/40 uppercase">RBF</th>
              <th className="px-6 py-3 text-xs font-semibold text-white/40 uppercase">Polynomial</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(groups).map(([combo, kernels]) => {
              const isFinal = combo === "MobileNetV2_plus_ConvNeXt";
              return (
                <tr key={combo} className={`border-b border-white/5 ${isFinal ? "bg-seaweed-500/10" : ""}`}>
                  <td className="px-6 py-3.5 text-sm font-medium text-white">
                    {combo.replace(/_plus_/g, "+")}
                    {isFinal && <span className="pill bg-seaweed-500/20 text-seaweed-300 text-[10px] ml-2">FINAL</span>}
                  </td>
                  <td className="px-6 py-3.5 text-sm text-white/70">{kernels.Linear ? pct(kernels.Linear.accuracy) : "—"}</td>
                  <td className="px-6 py-3.5 text-sm text-white/70">{kernels.RBF ? pct(kernels.RBF.accuracy) : "—"}</td>
                  <td className={`px-6 py-3.5 text-sm font-semibold ${isFinal ? "text-seaweed-400" : "text-white/70"}`}>
                    {kernels.Polynomial ? pct(kernels.Polynomial.accuracy) : "—"}
                    {isFinal && " ✓"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="bg-seaweed-500/10 border border-seaweed-500/25 rounded-xl px-5 py-4 text-sm text-white/70">
        For the final <strong className="text-white">MobileNetV2 + ConvNeXt</strong> fusion, the{" "}
        <strong className="text-seaweed-300">Polynomial kernel (98.29%)</strong> outperformed both Linear
        (97.96%) and RBF (97.81%), and was selected for the deployed model (C=1, degree=3, gamma=scale).
      </div>
    </div>
  );
}

function ConfusionTab({ data }) {
  if (!data) return <div className="text-white/40 text-sm text-center py-16">No confusion matrix data available.</div>;
  const labels = data.labels;
  const matrix = data.matrix;
  const max = Math.max(...matrix.flat());

  return (
    <div className="glass-card rounded-2xl p-6 sm:p-8">
      <h3 className="font-display font-semibold text-white mb-1">Final Model — Confusion Matrix</h3>
      <p className="text-xs text-white/40 mb-6">Rows = true class, columns = predicted class. Darker cells = higher counts.</p>
      <div className="overflow-x-auto">
        <table className="border-collapse mx-auto">
          <thead>
            <tr>
              <th className="p-2"></th>
              {labels.map((l) => (
                <th key={l} className="p-2 text-xs font-semibold text-white/50">{getDiseaseInfo(l).shortName}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={i}>
                <td className="p-2 text-xs font-semibold text-white/50 text-right pr-3">{getDiseaseInfo(labels[i]).shortName}</td>
                {row.map((val, j) => {
                  const intensity = val / max;
                  const isDiagonal = i === j;
                  return (
                    <td key={j} className="p-1">
                      <motion.div
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ delay: (i * labels.length + j) * 0.02 }}
                        className={`w-16 h-16 sm:w-20 sm:h-20 rounded-xl flex items-center justify-center text-sm font-bold ${
                          isDiagonal ? "text-white" : "text-white/70"
                        }`}
                        style={{
                          background: isDiagonal
                            ? `rgba(26, 180, 105, ${0.25 + intensity * 0.65})`
                            : val > 0
                            ? `rgba(255, 111, 79, ${0.15 + intensity * 0.55})`
                            : "rgba(255,255,255,0.04)",
                        }}
                      >
                        {val}
                      </motion.div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex items-center justify-center gap-6 mt-6 text-xs text-white/50">
        <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-seaweed-500/60"></span> Correct (diagonal)</span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-coral-500/60"></span> Misclassified</span>
      </div>
    </div>
  );
}

function RocTab({ data }) {
  if (!data) return <div className="text-white/40 text-sm text-center py-16">No ROC/AUC data available.</div>;
  const entries = Object.entries(data).filter(([k]) => !["Micro-average", "Macro-average"].includes(k));
  const macro = data["Macro-average"];
  const micro = data["Micro-average"];

  return (
    <div className="space-y-6">
      <div className="glass-card rounded-2xl p-6 sm:p-8">
        <h3 className="font-display font-semibold text-white mb-1">Per-Class AUC Scores</h3>
        <p className="text-xs text-white/40 mb-6">Area Under the ROC Curve — closer to 1.0 indicates near-perfect class separability.</p>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {entries.map(([cls, auc], i) => {
            const info = getDiseaseInfo(cls);
            return (
              <motion.div
                key={cls}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.08 }}
                className="bg-white/5 rounded-2xl p-5 text-center"
              >
                <div className="relative w-20 h-20 mx-auto mb-3">
                  <svg viewBox="0 0 36 36" className="w-full h-full -rotate-90">
                    <circle cx="18" cy="18" r="16" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="3" />
                    <motion.circle
                      cx="18" cy="18" r="16" fill="none" stroke="#40d287" strokeWidth="3" strokeLinecap="round"
                      strokeDasharray={`${auc * 100.5} 100.5`}
                      initial={{ strokeDasharray: "0 100.5" }}
                      animate={{ strokeDasharray: `${auc * 100.5} 100.5` }}
                      transition={{ duration: 1, delay: 0.2 }}
                    />
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center text-sm font-bold text-white">
                    {auc.toFixed(3)}
                  </div>
                </div>
                <div className="text-sm font-semibold text-white">{info.shortName}</div>
              </motion.div>
            );
          })}
        </div>
        <div className="grid sm:grid-cols-2 gap-4 mt-6">
          <div className="bg-ocean-500/10 rounded-xl p-4 text-center">
            <div className="text-xl font-display font-bold text-ocean-300">{micro?.toFixed(4)}</div>
            <div className="text-xs text-white/50 mt-1">Micro-average AUC</div>
          </div>
          <div className="bg-seaweed-500/10 rounded-xl p-4 text-center">
            <div className="text-xl font-display font-bold text-seaweed-400">{macro?.toFixed(4)}</div>
            <div className="text-xs text-white/50 mt-1">Macro-average AUC</div>
          </div>
        </div>
      </div>
    </div>
  );
}
