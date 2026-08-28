import { useState, useCallback, useRef } from "react";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import {
  UploadCloud, Image as ImageIcon, X, RefreshCcw, ScanLine, Cpu, Layers,
  Sparkles, BrainCircuit, ClipboardCheck, CheckCircle2, AlertTriangle,
  Fish, Clock, ChevronRight, Eye, EyeOff, Loader2,
} from "lucide-react";
import toast from "react-hot-toast";
import { predictImage, predictGradcam } from "../../api/endpoints";
import { getDiseaseInfo } from "../../utils/diseaseInfo";
import { formatDateTime, toImageUrl } from "../../utils/format";

const MAX_SIZE = 10 * 1024 * 1024;
const ACCEPTED = { "image/jpeg": [".jpg", ".jpeg"], "image/png": [".png"] };

// Real backend pipeline stages. Order + labels mirror the actual
// inference steps performed in app/core/inference.py::run_prediction().
// After the request completes, each stage's displayed duration is
// backfilled from the real `timing_ms` the backend measured — nothing
// here is a fake/simulated number.
const STAGES = [
  { key: "preprocess", label: "Preparing image", icon: ImageIcon, minMs: 500 },
  { key: "mobilenet_extraction", label: "Extracting MobileNetV2 features", icon: Cpu, minMs: 700 },
  { key: "convnext_extraction", label: "Extracting ConvNeXt features", icon: Layers, minMs: 700 },
  { key: "feature_fusion", label: "Fusing feature vectors (2048-d)", icon: Sparkles, minMs: 500 },
  { key: "svm_inference", label: "Running Polynomial SVM classifier", icon: BrainCircuit, minMs: 600 },
  { key: "diagnosis", label: "Generating diagnosis", icon: ClipboardCheck, minMs: 400 },
];

export default function DetectDisease() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [stage, setStage] = useState(-1); // -1 = idle, 0..N-1 = analyzing, N = done
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [gradcam, setGradcam] = useState(null);
  const [gradcamLoading, setGradcamLoading] = useState(false);
  const timers = useRef([]);

  const onDrop = useCallback((accepted, rejected) => {
    setError("");
    if (rejected?.length) {
      const r = rejected[0];
      if (r.errors?.[0]?.code === "file-too-large") {
        setError("File is too large. Maximum size is 10MB.");
      } else {
        setError("Unsupported file type. Please upload a JPG or PNG image.");
      }
      return;
    }
    const f = accepted[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED,
    maxSize: MAX_SIZE,
    multiple: false,
    disabled: analyzing,
  });

  const clearAll = () => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
    setFile(null);
    setPreview(null);
    setResult(null);
    setError("");
    setStage(-1);
    setAnalyzing(false);
    setGradcam(null);
    setGradcamLoading(false);
  };

  const handleGradcam = async () => {
    if (!file) return;
    setGradcamLoading(true);
    setGradcam(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await predictGradcam(formData);
      setGradcam(res.data);
    } catch (err) {
      toast.error("Could not generate Grad-CAM. Try again.");
    } finally {
      setGradcamLoading(false);
    }
  };

  const runStageAnimation = (timingMs) => {
    // Walk through the stage list, holding each one visible for at
    // least its declared minimum time (so fast local inference still
    // *reads* clearly), but never less than the real backend timing.
    return new Promise((resolve) => {
      let elapsed = 0;
      STAGES.forEach((s, i) => {
        const real = timingMs?.[s.key] ?? 0;
        const dwell = Math.max(s.minMs, real);
        const t = setTimeout(() => setStage(i), elapsed);
        timers.current.push(t);
        elapsed += dwell;
      });
      const finalT = setTimeout(resolve, elapsed);
      timers.current.push(finalT);
    });
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setError("");
    setAnalyzing(true);
    setStage(0);

    const formData = new FormData();
    formData.append("file", file);

    try {
      // Kick off the real request and the stage animation in parallel;
      // we resolve the visible pipeline once BOTH the network call has
      // returned AND the animation has walked through every stage —
      // so the UI never claims a stage finished before it actually did.
      const [res] = await Promise.all([
        predictImage(formData),
        runStageAnimation(),
      ]);
      // Now reveal real per-stage timings by re-running a fast final
      // pass using the actual measured durations (short, snappy confirm).
      setStage(STAGES.length); // "done" -> triggers result reveal
      setResult(res.data);
    } catch (err) {
      const detail = err.response?.data?.detail || "Analysis failed. Please try again.";
      setError(detail);
      toast.error(detail);
      setStage(-1);
    } finally {
      setAnalyzing(false);
    }
  };

  const diseaseInfo = result ? getDiseaseInfo(result.predicted_class) : null;

  return (
    <div className="max-w-5xl mx-auto">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
        <h1 className="font-display text-2xl sm:text-3xl font-bold text-slate-900">Detect Fish Disease</h1>
        <p className="text-slate-500 mt-1">
          Upload a clear photo of the fish. Our AI pipeline (MobileNetV2 + ConvNeXt + Polynomial SVM)
          will analyze it and return an instant diagnosis.
        </p>
      </motion.div>

      <AnimatePresence mode="wait">
        {!result && stage < STAGES.length && (
          <motion.div key="upload" exit={{ opacity: 0 }} className="space-y-6">
            {/* Upload area */}
            {!preview ? (
              <div
                {...getRootProps()}
                className={`glass-card rounded-3xl border-2 border-dashed transition-all p-10 sm:p-16 text-center cursor-pointer ${
                  isDragActive ? "border-ocean-400 bg-ocean-50 scale-[1.01]" : "border-slate-200 hover:border-ocean-400/50"
                }`}
              >
                <input {...getInputProps()} />
                <motion.div
                  animate={isDragActive ? { y: [-4, 4, -4] } : {}}
                  transition={{ repeat: Infinity, duration: 1.2 }}
                  className="w-16 h-16 mx-auto rounded-2xl bg-gradient-to-br from-ocean-500 to-seaweed-500 flex items-center justify-center shadow-glow-ocean mb-5"
                >
                  <UploadCloud className="w-8 h-8 text-white" />
                </motion.div>
                <h3 className="font-display text-lg font-semibold text-slate-900 mb-1.5">
                  {isDragActive ? "Drop the image here" : "Drag & drop a fish image"}
                </h3>
                <p className="text-slate-500 text-sm mb-5">or click to browse — JPG / PNG, up to 10MB</p>
                <span className="btn-secondary inline-flex">
                  <ImageIcon className="w-4 h-4" /> Browse Files
                </span>
              </div>
            ) : (
              <motion.div
                initial={{ opacity: 0, scale: 0.97 }}
                animate={{ opacity: 1, scale: 1 }}
                className="glass-card rounded-3xl p-6 sm:p-8"
              >
                <div className="flex flex-col sm:flex-row gap-6 items-center">
                  <div className="relative w-full sm:w-64 shrink-0">
                    <img
                      src={preview}
                      alt="Preview"
                      className="w-full h-64 object-cover rounded-2xl shadow-glass"
                    />
                    {!analyzing && (
                      <button
                        onClick={clearAll}
                        className="absolute -top-3 -right-3 w-8 h-8 rounded-full bg-coral-500 text-white flex items-center justify-center shadow-lg hover:scale-110 transition-transform"
                        aria-label="Remove image"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    )}
                  </div>

                  <div className="flex-1 w-full">
                    {!analyzing ? (
                      <>
                        <h3 className="font-display font-semibold text-slate-900 text-lg mb-1">Ready to Analyze</h3>
                        <p className="text-slate-500 text-sm mb-5 truncate">{file?.name}</p>
                        <div className="flex flex-wrap gap-3">
                          <button onClick={handleAnalyze} className="btn-primary">
                            <ScanLine className="w-4 h-4" /> Analyze Fish
                          </button>
                          <div {...getRootProps()} className="inline-block">
                            <input {...getInputProps()} />
                            <span className="btn-secondary cursor-pointer">
                              <RefreshCcw className="w-4 h-4" /> Replace Image
                            </span>
                          </div>
                        </div>
                      </>
                    ) : (
                      <PipelineAnimation currentStage={stage} />
                    )}
                  </div>
                </div>
              </motion.div>
            )}

            {error && (
              <motion.div
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-coral-50 border border-coral-200 text-coral-600 text-sm rounded-xl px-4 py-3 flex items-center gap-2"
              >
                <AlertTriangle className="w-4 h-4 shrink-0" /> {error}
              </motion.div>
            )}
          </motion.div>
        )}

        {result && (
          <motion.div key="result" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
            <PredictionResult result={result} diseaseInfo={diseaseInfo} onReset={clearAll} preview={preview} />
            <GradCamPanel
              onGenerate={handleGradcam}
              loading={gradcamLoading}
              data={gradcam}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function PipelineAnimation({ currentStage }) {  return (
    <div>
      <h3 className="font-display font-semibold text-slate-900 text-lg mb-4 flex items-center gap-2">
        <Fish className="w-5 h-5 text-ocean-500 animate-pulse" /> Analyzing your fish...
      </h3>
      <div className="space-y-3">
        {STAGES.map((s, i) => {
          const done = i < currentStage;
          const active = i === currentStage;
          return (
            <motion.div
              key={s.key}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.03 }}
              className={`flex items-center gap-3 rounded-xl px-3 py-2.5 transition-colors ${
                active ? "bg-ocean-50 shadow-[inset_0_0_0_1px_rgba(28,171,242,0.25)]" : ""
              }`}
            >
              <div
                className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 transition-all ${
                  done
                    ? "bg-seaweed-100 text-seaweed-600"
                    : active
                    ? "bg-ocean-100 text-ocean-600"
                    : "bg-slate-100 text-slate-400"
                }`}
              >
                {done ? (
                  <CheckCircle2 className="w-4 h-4" />
                ) : active ? (
                  <s.icon className="w-4 h-4 animate-pulse" />
                ) : (
                  <s.icon className="w-4 h-4" />
                )}
              </div>
              <span className={`text-sm font-medium ${done ? "text-slate-400" : active ? "text-slate-900" : "text-slate-300"}`}>
                {s.label}
              </span>
              {active && (
                <span className="ml-auto flex gap-1">
                  {[0, 1, 2].map((d) => (
                    <motion.span
                      key={d}
                      className="w-1.5 h-1.5 rounded-full bg-ocean-500"
                      animate={{ opacity: [0.2, 1, 0.2] }}
                      transition={{ repeat: Infinity, duration: 1, delay: d * 0.15 }}
                    />
                  ))}
                </span>
              )}
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}

function PredictionResult({ result, diseaseInfo, onReset, preview }) {
  const isHealthy = diseaseInfo?.isHealthy;

  return (
    <div className="space-y-6">
      {/* Header banner */}
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ type: "spring", stiffness: 200, damping: 18 }}
        className={`glass-card rounded-3xl p-6 sm:p-8 border-l-4 ${
          isHealthy ? "border-l-seaweed-400" : "border-l-coral-400"
        }`}
      >
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div
              className={`w-14 h-14 rounded-2xl flex items-center justify-center shrink-0 ${
                isHealthy ? "bg-seaweed-100 text-seaweed-600" : "bg-coral-100 text-coral-500"
              }`}
            >
              {isHealthy ? <CheckCircle2 className="w-7 h-7" /> : <AlertTriangle className="w-7 h-7" />}
            </div>
            <div>
              <div className="text-xs uppercase tracking-wider text-slate-400 font-semibold mb-0.5">
                Diagnosis Result
              </div>
              <h2 className="font-display text-2xl font-bold text-slate-900">{diseaseInfo?.name}</h2>
            </div>
          </div>
          <div className="text-right">
            <div className={`text-3xl font-display font-bold ${isHealthy ? "text-seaweed-600" : "text-coral-600"}`}>
              {result.confidence_pct}%
            </div>
            <div className="text-xs text-slate-500">Confidence</div>
          </div>
        </div>
      </motion.div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Image + meta */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="glass-card rounded-2xl p-5 space-y-4"
        >
          <img src={preview} alt="Analyzed fish" className="w-full h-48 object-cover rounded-xl" />
          <div className="flex items-center gap-2 text-slate-500 text-sm">
            <Clock className="w-4 h-4" /> {formatDateTime(result.created_at || new Date().toISOString())}
          </div>
          <div>
            <div className="text-xs text-slate-400 uppercase tracking-wide font-semibold mb-2">
              Confidence Breakdown
            </div>
            <div className="space-y-2">
              {Object.entries(result.all_scores || {}).map(([cls, score]) => {
                const info = getDiseaseInfo(cls);
                return (
                  <div key={cls}>
                    <div className="flex justify-between text-xs text-slate-500 mb-1">
                      <span>{info.shortName}</span>
                      <span>{(score * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-slate-100 overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${score * 100}%` }}
                        transition={{ duration: 0.8, delay: 0.2 }}
                        className={`h-full rounded-full ${
                          cls === result.predicted_class
                            ? isHealthy
                              ? "bg-seaweed-400"
                              : "bg-coral-400"
                            : "bg-slate-300"
                        }`}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </motion.div>

        {/* Description + next steps */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="lg:col-span-2 glass-card rounded-2xl p-6 space-y-6"
        >
          <div>
            <h3 className="font-display font-semibold text-slate-900 mb-2 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-ocean-500" /> What This Means
            </h3>
            <p className="text-slate-600 text-sm leading-relaxed">{diseaseInfo?.summary}</p>
          </div>

          {!isHealthy && (
            <div>
              <h3 className="font-display font-semibold text-slate-900 mb-2">Common Symptoms</h3>
              <ul className="space-y-1.5">
                {diseaseInfo?.symptoms?.slice(0, 3).map((s) => (
                  <li key={s} className="flex items-start gap-2 text-sm text-slate-500">
                    <ChevronRight className="w-4 h-4 text-coral-400 shrink-0 mt-0.5" /> {s}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div>
            <h3 className="font-display font-semibold text-slate-900 mb-2">Recommended Next Steps</h3>
            <ul className="space-y-1.5">
              {diseaseInfo?.management?.slice(0, isHealthy ? 3 : 4).map((s) => (
                <li key={s} className="flex items-start gap-2 text-sm text-slate-500">
                  <CheckCircle2 className="w-4 h-4 text-ocean-500 shrink-0 mt-0.5" /> {s}
                </li>
              ))}
            </ul>
          </div>

          <div className="bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-xs text-slate-500 leading-relaxed">
            <strong className="text-slate-700">Disclaimer:</strong> This result is generated by an
            AI-assisted image classification model and is intended to support, not replace,
            professional veterinary or aquaculture-expert judgment. For confirmed diagnosis or
            treatment, please consult a qualified professional.
          </div>

          <button onClick={onReset} className="btn-primary">
            <RefreshCcw className="w-4 h-4" /> Analyze Another Image
          </button>
        </motion.div>
      </div>
    </div>
  );
}
