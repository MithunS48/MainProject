import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, ArrowRight, ListTree } from "lucide-react";
import { getErrorAnalysis, getClassificationReport } from "../../api/endpoints";
import { getDiseaseInfo } from "../../utils/diseaseInfo";

export default function ErrorAnalysis() {
  const [data, setData] = useState(null);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const [e, r] = await Promise.all([getErrorAnalysis(), getClassificationReport()]);
        setData(e.data);
        setReport(r.data);
      } catch {
        setData(null);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  if (loading) {
    return <div className="text-white/40 text-sm py-16 text-center">Loading error analysis...</div>;
  }

  const correctPct = data ? ((data.correct_predictions / data.test_samples) * 100).toFixed(2) : 0;
  const incorrectPct = data ? ((data.incorrect_predictions / data.test_samples) * 100).toFixed(2) : 0;

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="font-display text-2xl sm:text-3xl font-bold text-white">Error Analysis</h1>
        <p className="text-white/60 mt-1">
          Breakdown of correct vs. incorrect predictions on the held-out test set, and where
          misclassifications occurred.
        </p>
      </motion.div>

      {/* Correct/incorrect summary */}
      <div className="grid sm:grid-cols-2 gap-4">
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="glass-card rounded-2xl p-6 border-l-4 border-l-seaweed-400">
          <div className="flex items-center gap-3 mb-2">
            <CheckCircle2 className="w-6 h-6 text-seaweed-400" />
            <span className="text-sm font-semibold text-white/60">Correct Predictions</span>
          </div>
          <div className="text-3xl font-display font-bold text-white">
            {data?.correct_predictions} <span className="text-lg text-white/40">/ {data?.test_samples}</span>
          </div>
          <div className="text-sm text-seaweed-400 font-medium mt-1">{correctPct}% of test set</div>
        </motion.div>

        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="glass-card rounded-2xl p-6 border-l-4 border-l-coral-400">
          <div className="flex items-center gap-3 mb-2">
            <AlertTriangle className="w-6 h-6 text-coral-400" />
            <span className="text-sm font-semibold text-white/60">Incorrect Predictions</span>
          </div>
          <div className="text-3xl font-display font-bold text-white">
            {data?.incorrect_predictions} <span className="text-lg text-white/40">/ {data?.test_samples}</span>
          </div>
          <div className="text-sm text-coral-400 font-medium mt-1">{incorrectPct}% of test set</div>
        </motion.div>
      </div>

      {/* Error breakdown */}
      <div className="glass-card rounded-2xl p-6">
        <h3 className="font-display font-semibold text-white mb-1 flex items-center gap-2">
          <ListTree className="w-4 h-4 text-ocean-300" /> Misclassification Breakdown
        </h3>
        <p className="text-xs text-white/40 mb-4">Which true classes were confused with which predicted classes, and how often.</p>
        <div className="space-y-2">
          {(data?.error_breakdown || []).map((e, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.05 }}
              className="flex items-center gap-3 bg-white/5 rounded-xl px-4 py-3"
            >
              <span className="text-sm font-medium text-coral-300">{getDiseaseInfo(e.true_class).shortName}</span>
              <ArrowRight className="w-4 h-4 text-white/30" />
              <span className="text-sm font-medium text-white/70">{getDiseaseInfo(e.predicted_class).shortName}</span>
              <span className="ml-auto text-sm font-bold text-white bg-white/10 rounded-full px-2.5 py-0.5">{e.count}×</span>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Per-class report */}
      {report && (
        <div className="glass-card rounded-2xl overflow-hidden">
          <div className="p-6 pb-0">
            <h3 className="font-display font-semibold text-white mb-1">Per-Class Classification Report</h3>
            <p className="text-xs text-white/40 mb-4">Precision, recall, and F1-score for each disease class on the test set.</p>
          </div>
          <table className="w-full">
            <thead>
              <tr className="border-b border-white/10 text-left">
                <th className="px-6 py-3 text-xs font-semibold text-white/40 uppercase">Class</th>
                <th className="px-6 py-3 text-xs font-semibold text-white/40 uppercase">Precision</th>
                <th className="px-6 py-3 text-xs font-semibold text-white/40 uppercase">Recall</th>
                <th className="px-6 py-3 text-xs font-semibold text-white/40 uppercase">F1-Score</th>
                <th className="px-6 py-3 text-xs font-semibold text-white/40 uppercase hidden sm:table-cell">Support</th>
              </tr>
            </thead>
            <tbody>
              {(report.per_class_metrics || []).map((m) => (
                <tr key={m.class} className="border-b border-white/5">
                  <td className="px-6 py-3.5 text-sm font-medium text-white">{getDiseaseInfo(m.class).shortName}</td>
                  <td className="px-6 py-3.5 text-sm text-white/70">{(m.precision * 100).toFixed(2)}%</td>
                  <td className="px-6 py-3.5 text-sm text-white/70">{(m.recall * 100).toFixed(2)}%</td>
                  <td className="px-6 py-3.5 text-sm text-white/70">{(m.f1_score * 100).toFixed(2)}%</td>
                  <td className="px-6 py-3.5 text-sm text-white/50 hidden sm:table-cell">{m.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Sample errors */}
      {data?.sample_errors?.length > 0 && (
        <div className="glass-card rounded-2xl p-6">
          <h3 className="font-display font-semibold text-white mb-1">Sample Misclassified Predictions</h3>
          <p className="text-xs text-white/40 mb-4">Test-set examples where the model's predicted class differed from the true label.</p>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/10 text-left">
                  <th className="px-4 py-2 text-xs font-semibold text-white/40 uppercase">True</th>
                  <th className="px-4 py-2 text-xs font-semibold text-white/40 uppercase">Predicted</th>
                  <th className="px-4 py-2 text-xs font-semibold text-white/40 uppercase">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {data.sample_errors.slice(0, 15).map((e, i) => (
                  <tr key={i} className="border-b border-white/5">
                    <td className="px-4 py-2.5 text-sm text-white/70">{getDiseaseInfo(e.true_class).shortName}</td>
                    <td className="px-4 py-2.5 text-sm text-coral-300 font-medium">{getDiseaseInfo(e.predicted_class).shortName}</td>
                    <td className="px-4 py-2.5 text-sm text-white/50">{(e.confidence * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
