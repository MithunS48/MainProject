import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  Fish, Cpu, Layers, Sparkles, BrainCircuit, ClipboardCheck, ArrowRight,
  Target, TrendingUp, Activity, Award,
} from "lucide-react";
import { getModelInfo } from "../../api/endpoints";

const STAGE_ICONS = {
  "Input Fish Image": Fish,
  MobileNetV2: Cpu,
  ConvNeXt: Layers,
  "Feature Fusion": Sparkles,
  "Polynomial SVM": BrainCircuit,
  "Disease Prediction": ClipboardCheck,
};

export default function ModelPerformance() {
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const res = await getModelInfo();
        setInfo(res.data);
      } catch {
        setInfo(null);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  return (
    <div className="space-y-8">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="font-display text-2xl sm:text-3xl font-bold text-slate-900">Model Information</h1>
        <p className="text-slate-500 mt-1">
          Architecture, performance metrics, and rationale behind the deployed fish-disease classifier.
        </p>
      </motion.div>

      {loading ? (
        <div className="text-slate-400 text-sm py-16 text-center">Loading model information...</div>
      ) : (
        <>
          {/* Pipeline diagram */}
          <div className="glass-card rounded-3xl p-6 sm:p-8">
            <h2 className="font-display font-semibold text-slate-900 mb-6">Pipeline Architecture</h2>
            <div className="flex flex-wrap items-center justify-center gap-3">
              {(info?.pipeline || []).map((step, i) => {
                const Icon = STAGE_ICONS[step.step] || Sparkles;
                return (
                  <div key={step.step} className="flex items-center gap-3">
                    <motion.div
                      initial={{ opacity: 0, scale: 0.85 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: i * 0.08 }}
                      whileHover={{ y: -4 }}
                      className="glass-card rounded-2xl px-5 py-5 flex flex-col items-center gap-2 min-w-[150px] text-center hover:shadow-glow-ocean transition-shadow"
                    >
                      <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-ocean-500 to-seaweed-500 flex items-center justify-center">
                        <Icon className="w-5 h-5 text-white" />
                      </div>
                      <div className="text-sm font-semibold text-slate-900">{step.step}</div>
                      <div className="text-xs text-slate-500">{step.detail}</div>
                    </motion.div>
                    {i < (info?.pipeline?.length || 0) - 1 && (
                      <ArrowRight className="w-5 h-5 text-slate-300 hidden sm:block" />
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Headline metrics */}
          <div className="grid sm:grid-cols-3 gap-4">
            <MetricCard icon={Target} label="Test Accuracy" value={`${(info?.accuracy * 100).toFixed(2)}%`} accent="seaweed" />
            <MetricCard icon={TrendingUp} label="F1-Score" value={info?.f1_score?.toFixed(4)} accent="ocean" />
            <MetricCard icon={Activity} label="Macro AUC" value={info?.auc?.toFixed(4)} accent="coral" />
          </div>

          {/* Correct/Incorrect breakdown */}
          <div className="glass-card rounded-2xl p-6 sm:p-8">
            <h2 className="font-display font-semibold text-slate-900 mb-4 flex items-center gap-2">
              <Award className="w-4 h-4 text-ocean-500" /> Test Set Results
            </h2>
            <div className="grid sm:grid-cols-3 gap-4">
              <div className="bg-slate-50 rounded-xl p-4 text-center">
                <div className="text-2xl font-display font-bold text-slate-900">{info?.test_samples}</div>
                <div className="text-xs text-slate-500 mt-1">Total Test Samples</div>
              </div>
              <div className="bg-seaweed-50 rounded-xl p-4 text-center">
                <div className="text-2xl font-display font-bold text-seaweed-600">{info?.correct_predictions}</div>
                <div className="text-xs text-slate-500 mt-1">Correct Predictions</div>
              </div>
              <div className="bg-coral-50 rounded-xl p-4 text-center">
                <div className="text-2xl font-display font-bold text-coral-600">{info?.incorrect_predictions}</div>
                <div className="text-xs text-slate-500 mt-1">Incorrect Predictions</div>
              </div>
            </div>
          </div>

          {/* Feature dims + classifier config */}
          <div className="grid sm:grid-cols-2 gap-6">
            <div className="glass-card rounded-2xl p-6">
              <h3 className="font-display font-semibold text-slate-900 mb-4">Feature Dimensions</h3>
              <div className="space-y-3">
                {Object.entries(info?.feature_dimensions || {}).map(([k, v]) => (
                  <div key={k} className="flex justify-between text-sm">
                    <span className="text-slate-500">{k}</span>
                    <span className="text-slate-900 font-semibold">{v}-d</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="glass-card rounded-2xl p-6">
              <h3 className="font-display font-semibold text-slate-900 mb-4">Classifier Configuration</h3>
              <p className="text-sm text-slate-600 leading-relaxed">{info?.classifier}</p>
            </div>
          </div>

          {/* Rationale */}
          <div className="glass-card rounded-2xl p-6 sm:p-8">
            <h2 className="font-display font-semibold text-slate-900 mb-3">Why Feature Fusion?</h2>
            <p className="text-sm text-slate-500 leading-relaxed">
              MobileNetV2 and ConvNeXt learn complementary visual representations — MobileNetV2's
              efficient depthwise-separable convolutions capture fine-grained local texture patterns
              (useful for detecting lesions and discoloration), while ConvNeXt's modern architecture
              captures broader structural and contextual patterns. Concatenating their 1,280-d and
              768-d feature vectors into a single 2,048-d representation gives the downstream
              classifier access to both perspectives simultaneously. Empirically, this fused
              representation (97.77% with a linear/RBF SVM) outperformed either individual CNN, and
              tuning the classifier to a Polynomial-kernel SVM pushed the final accuracy to 98.29% —
              the best result across every architecture and fusion combination tested during this
              research.
            </p>
          </div>
        </>
      )}
    </div>
  );
}

function MetricCard({ icon: Icon, label, value, accent }) {
  const accentMap = {
    ocean: "from-ocean-400 to-ocean-600",
    seaweed: "from-seaweed-400 to-seaweed-600",
    coral: "from-coral-400 to-coral-600",
  };
  return (
    <div className="glass-card rounded-2xl p-6 flex items-center gap-4">
      <div className={`w-11 h-11 rounded-xl bg-gradient-to-br ${accentMap[accent]} flex items-center justify-center shrink-0`}>
        <Icon className="w-5 h-5 text-white" />
      </div>
      <div>
        <div className="text-2xl font-display font-bold text-slate-900">{value}</div>
        <div className="text-xs text-slate-500">{label}</div>
      </div>
    </div>
  );
}
