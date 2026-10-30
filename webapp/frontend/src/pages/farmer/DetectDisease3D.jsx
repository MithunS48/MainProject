import { useState, useCallback, useRef, Suspense } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Float, Stars, Html } from "@react-three/drei";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import * as THREE from "three";
import {
  UploadCloud, Image as ImageIcon, X, RefreshCcw, ScanLine, Cpu, Layers,
  Sparkles, BrainCircuit, ClipboardCheck, CheckCircle2, AlertTriangle,
  Fish, Clock, ChevronRight, Camera, CameraOff,
} from "lucide-react";
import toast from "react-hot-toast";
import { predictImage } from "../../api/endpoints";
import { getDiseaseInfo } from "../../utils/diseaseInfo";
import { formatDateTime } from "../../utils/format";

// ─── Constants ────────────────────────────────────────────────────────────────
const MAX_SIZE = 10 * 1024 * 1024;
const ACCEPTED = { "image/jpeg": [".jpg", ".jpeg"], "image/png": [".png"] };

const STAGES = [
  { key: "preprocess",           label: "Preparing image",                  icon: ImageIcon,     minMs: 500, color: "#00d4ff" },
  { key: "mobilenet_extraction", label: "Extracting MobileNetV2 features",  icon: Cpu,           minMs: 700, color: "#00ff88" },
  { key: "convnext_extraction",  label: "Extracting ConvNeXt features",     icon: Layers,        minMs: 700, color: "#a855f7" },
  { key: "feature_fusion",       label: "Fusing feature vectors (2048-d)",  icon: Sparkles,      minMs: 500, color: "#ff6b35" },
  { key: "svm_inference",        label: "Running Polynomial SVM classifier",icon: BrainCircuit,  minMs: 600, color: "#ffd60a" },
  { key: "diagnosis",            label: "Generating diagnosis",             icon: ClipboardCheck,minMs: 400, color: "#00d4ff" },
];

// ─── 3D Background: Floating Pipeline Nodes ───────────────────────────────────
function PipelineNode3D({ position, color, active, done, index }) {
  const mesh = useRef();
  useFrame((state) => {
    const t = state.clock.elapsedTime;
    mesh.current.rotation.y = t * 0.5 + index;
    mesh.current.rotation.x = Math.sin(t * 0.3 + index) * 0.2;
    mesh.current.scale.setScalar(
      active ? 1.4 + Math.sin(t * 3) * 0.1 : done ? 1.1 : 0.8
    );
  });
  return (
    <Float speed={1.5} floatIntensity={0.4}>
      <mesh ref={mesh} position={position}>
        <octahedronGeometry args={[0.25, 0]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={active ? 1.2 : done ? 0.6 : 0.15}
          roughness={0.1}
          metalness={0.9}
        />
      </mesh>
    </Float>
  );
}

// ─── Connector line between nodes ────────────────────────────────────────────
function NodeConnector({ from, to, active }) {
  const points = [new THREE.Vector3(...from), new THREE.Vector3(...to)];
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  return (
    <line geometry={geometry}>
      <lineBasicMaterial
        color={active ? "#00d4ff" : "#0a2a4a"}
        opacity={active ? 0.8 : 0.3}
        transparent
      />
    </line>
  );
}

// ─── Particle ring ────────────────────────────────────────────────────────────
function ParticleRing({ active }) {
  const ref = useRef();
  const positions = useRef(
    new Float32Array(
      Array.from({ length: 120 }, (_, i) => {
        const angle = (i / 120) * Math.PI * 2;
        const r = 3.5 + Math.random() * 0.5;
        return [Math.cos(angle) * r, Math.sin(angle) * r, (Math.random() - 0.5) * 0.5];
      }).flat()
    )
  ).current;

  useFrame((state) => {
    if (ref.current) {
      ref.current.rotation.z = state.clock.elapsedTime * 0.1;
      ref.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.05) * 0.2;
    }
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial
        color={active ? "#00d4ff" : "#0a3a5a"}
        size={0.05}
        transparent
        opacity={active ? 0.8 : 0.3}
        sizeAttenuation
      />
    </points>
  );
}

// ─── 3D background canvas ─────────────────────────────────────────────────────
function DetectCanvas({ currentStage }) {
  const nodePositions = [
    [-5, 1.5, -2], [-2.5, -1, -2], [0, 1.5, -2],
    [2.5, -1, -2], [5, 1.5, -2], [0, -3, -2],
  ];

  return (
    <Canvas camera={{ position: [0, 0, 8], fov: 55 }} dpr={[1, 1.5]} gl={{ alpha: true }} style={{ pointerEvents: "none" }}>
      <Suspense fallback={null}>
        <ambientLight intensity={0.3} color="#0077b6" />
        <pointLight position={[0, 5, 3]} intensity={1.5} color="#00d4ff" />
        <Stars radius={60} depth={30} count={400} factor={2} fade />
        <ParticleRing active={currentStage >= 0} />

        {nodePositions.map((pos, i) => (
          <PipelineNode3D
            key={i}
            position={pos}
            color={STAGES[i]?.color || "#00d4ff"}
            active={i === currentStage}
            done={i < currentStage}
            index={i}
          />
        ))}

        {nodePositions.slice(0, -1).map((pos, i) => (
          <NodeConnector
            key={i}
            from={pos}
            to={nodePositions[i + 1]}
            active={i < currentStage}
          />
        ))}
      </Suspense>
    </Canvas>
  );
}

// ─── Camera Capture Component ─────────────────────────────────────────────────
function CameraCapture({ onCapture, onClose }) {
  const videoRef  = useRef(null);
  const streamRef = useRef(null);
  const [ready, setReady]       = useState(false);
  const [facing, setFacing]     = useState("environment"); // rear camera on mobile
  const [cameraErr, setCameraErr] = useState("");

  const startCamera = useCallback(async (facingMode) => {
    // Stop any previous stream
    streamRef.current?.getTracks().forEach((t) => t.stop());
    setCameraErr("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode, width: { ideal: 1280 }, height: { ideal: 960 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => setReady(true);
      }
    } catch (err) {
      setCameraErr(
        err.name === "NotAllowedError"
          ? "Camera permission denied. Please allow camera access in your browser settings."
          : err.name === "NotFoundError"
          ? "No camera found on this device."
          : `Camera error: ${err.message}`
      );
    }
  }, []);

  // Start on mount
  useCallback(() => { startCamera(facing); }, [])();

  // Actually call startCamera on mount via useEffect-like pattern
  const startedRef = useRef(false);
  if (!startedRef.current) {
    startedRef.current = true;
    startCamera(facing);
  }

  const flipCamera = () => {
    const next = facing === "environment" ? "user" : "environment";
    setFacing(next);
    setReady(false);
    startCamera(next);
  };

  const capture = () => {
    if (!videoRef.current || !ready) return;
    const canvas = document.createElement("canvas");
    canvas.width  = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    canvas.getContext("2d").drawImage(videoRef.current, 0, 0);
    canvas.toBlob((blob) => {
      if (!blob) return;
      const file = new File([blob], `capture_${Date.now()}.jpg`, { type: "image/jpeg" });
      streamRef.current?.getTracks().forEach((t) => t.stop());
      onCapture(file, URL.createObjectURL(blob));
    }, "image/jpeg", 0.92);
  };

  const close = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    onClose();
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.96 }}
      className="rounded-3xl border border-sky-200 bg-white/95 overflow-hidden shadow-xl"
    >
      {cameraErr ? (
        <div className="p-8 text-center space-y-3">
          <CameraOff className="w-12 h-12 text-slate-300 mx-auto" />
          <p className="text-sm text-red-600">{cameraErr}</p>
          <button onClick={close} className="text-sm text-sky-600 underline">Close</button>
        </div>
      ) : (
        <>
          {/* Video preview */}
          <div className="relative bg-black">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full max-h-72 object-cover"
            />
            {!ready && (
              <div className="absolute inset-0 flex items-center justify-center bg-slate-900/60">
                <div className="w-8 h-8 border-2 border-sky-400/40 border-t-sky-400 rounded-full animate-spin" />
              </div>
            )}
            {/* Crosshair guide */}
            <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
              <div className="w-48 h-48 border-2 border-white/40 rounded-2xl" />
            </div>
            <button onClick={close} className="absolute top-3 right-3 w-8 h-8 rounded-full bg-black/50 text-white flex items-center justify-center hover:bg-black/70">
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="p-4 flex items-center gap-3">
            <button
              onClick={flipCamera}
              title="Flip camera"
              className="p-2.5 rounded-xl border border-slate-200 text-slate-500 hover:bg-sky-50 transition-colors"
            >
              <RefreshCcw className="w-4 h-4" />
            </button>
            <button
              onClick={capture}
              disabled={!ready}
              className="flex-1 py-3 rounded-xl bg-gradient-to-r from-sky-500 to-blue-600 text-white font-semibold text-sm hover:from-sky-400 hover:to-blue-500 disabled:opacity-40 transition-all flex items-center justify-center gap-2"
            >
              <Camera className="w-4 h-4" /> Capture Photo
            </button>
          </div>
          <p className="text-center text-xs text-slate-400 pb-3 px-4">
            Position the fish inside the guide box. Use the rear camera for best results.
          </p>
        </>
      )}
    </motion.div>
  );
}


export default function DetectDisease3D() {
  const [file, setFile]       = useState(null);
  const [preview, setPreview] = useState(null);
  const [stage, setStage]     = useState(-1);
  const [result, setResult]   = useState(null);
  const [error, setError]     = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [showCamera, setShowCamera] = useState(false);
  const timers = useRef([]);

  const onDrop = useCallback((accepted, rejected) => {
    setError("");
    if (rejected?.length) {
      setError(
        rejected[0].errors?.[0]?.code === "file-too-large"
          ? "File is too large. Max 10MB."
          : "Unsupported file type. Use JPG or PNG."
      );
      return;
    }
    const f = accepted[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: ACCEPTED, maxSize: MAX_SIZE, multiple: false, disabled: analyzing,
  });

  const clearAll = () => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
    setFile(null); setPreview(null); setResult(null);
    setError(""); setStage(-1); setAnalyzing(false);
    setShowCamera(false);
  };

  const handleCameraCapture = (capturedFile, blobUrl) => {
    setFile(capturedFile);
    setPreview(blobUrl);
    setResult(null);
    setShowCamera(false);
  };

  const runStageAnimation = (timingMs) =>
    new Promise((resolve) => {
      let elapsed = 0;
      STAGES.forEach((s, i) => {
        const dwell = Math.max(s.minMs, timingMs?.[s.key] ?? 0);
        const t = setTimeout(() => setStage(i), elapsed);
        timers.current.push(t);
        elapsed += dwell;
      });
      timers.current.push(setTimeout(resolve, elapsed));
    });

  const handleAnalyze = async () => {
    if (!file) return;
    setError(""); setAnalyzing(true); setStage(0);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const [res] = await Promise.all([predictImage(formData), runStageAnimation()]);
      setStage(STAGES.length);
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
    <div className="relative min-h-[calc(100vh-4rem)] bg-[#e8f4fd]">

      {/* 3D Background Canvas — pointer-events none so dashboard nav works */}
      <div className="fixed inset-0 pointer-events-none z-0 opacity-70">
        <DetectCanvas currentStage={stage} />
      </div>

      {/* ── Content ── */}
      <div className="relative z-10 max-w-5xl mx-auto px-4 py-6">

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-800 flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-sky-500 to-blue-600 flex items-center justify-center shadow-lg shadow-sky-500/30">
              <Fish className="w-5 h-5 text-slate-800" />
            </div>
            Detect Fish Disease
          </h1>
          <p className="text-slate-500 mt-2 text-sm">
            Upload a clear photo. Our AI pipeline (MobileNetV2 + ConvNeXt + Polynomial SVM) returns an instant diagnosis.
          </p>
        </motion.div>

        <AnimatePresence mode="wait">

          {/* ── Upload / Analyzing ── */}
          {!result && stage < STAGES.length && (
            <motion.div key="upload" exit={{ opacity: 0 }} className="space-y-6">

              {!preview ? (
                /* Drop Zone */
                <div className="space-y-4">
                  <div
                    {...getRootProps()}
                    className={`rounded-3xl border-2 border-dashed transition-all p-10 sm:p-16 text-center cursor-pointer backdrop-blur-sm
                      ${isDragActive
                        ? "border-cyan-400 bg-white/70 scale-[1.01]"
                        : "border-cyan-900/50 bg-white/70 hover:border-cyan-500/50 hover:bg-slate-900/80"
                      }`}
                  >
                    <input {...getInputProps()} />
                    <motion.div
                      animate={isDragActive ? { y: [-4, 4, -4] } : {}}
                      transition={{ repeat: Infinity, duration: 1.2 }}
                      className="w-16 h-16 mx-auto rounded-2xl bg-gradient-to-br from-sky-500 to-blue-600 flex items-center justify-center shadow-lg shadow-sky-500/30 mb-5"
                    >
                      <UploadCloud className="w-8 h-8 text-slate-800" />
                    </motion.div>
                    <h3 className="text-lg font-semibold text-slate-800 mb-1.5">
                      {isDragActive ? "Drop the image here" : "Drag & drop a fish image"}
                    </h3>
                    <p className="text-slate-500 text-sm mb-5">or click to browse — JPG / PNG, up to 10MB</p>
                    <span className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl border border-sky-200 text-sky-700 text-sm font-medium bg-white/70 hover:bg-sky-100/70 transition-colors">
                      <ImageIcon className="w-4 h-4" /> Browse Files
                    </span>
                  </div>

                  {/* Camera button */}
                  <div className="text-center">
                    <span className="text-slate-400 text-sm mr-3">or</span>
                    <button
                      onClick={() => setShowCamera(true)}
                      className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl border border-sky-200 bg-white text-sky-700 text-sm font-medium hover:bg-sky-50 transition-colors shadow-sm"
                    >
                      <Camera className="w-4 h-4" /> Use Camera
                    </button>
                  </div>

                  <AnimatePresence>
                    {showCamera && (
                      <CameraCapture
                        onCapture={handleCameraCapture}
                        onClose={() => setShowCamera(false)}
                      />
                    )}
                  </AnimatePresence>
                </div>
              ) : (
                /* Preview + Analyze */
                <motion.div
                  initial={{ opacity: 0, scale: 0.97 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="rounded-3xl border border-sky-200 bg-white/90 backdrop-blur-sm p-6 sm:p-8"
                >
                  <div className="flex flex-col sm:flex-row gap-6 items-center">
                    <div className="relative w-full sm:w-64 shrink-0">
                      <img src={preview} alt="Preview" className="w-full h-64 object-cover rounded-2xl shadow-lg shadow-cyan-500/10" />
                      {!analyzing && (
                        <button
                          onClick={clearAll}
                          className="absolute -top-3 -right-3 w-8 h-8 rounded-full bg-red-500 text-slate-800 flex items-center justify-center shadow-lg hover:scale-110 transition-transform"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                    <div className="flex-1 w-full">
                      {!analyzing ? (
                        <>
                          <h3 className="font-semibold text-slate-800 text-lg mb-1">Ready to Analyze</h3>
                          <p className="text-slate-500 text-sm mb-5 truncate">{file?.name}</p>
                          <div className="flex flex-wrap gap-3">
                            <button
                              onClick={handleAnalyze}
                              className="inline-flex items-center gap-2 px-6 py-3 rounded-xl font-semibold text-sm bg-gradient-to-r from-sky-500 to-blue-600 text-slate-800 hover:from-cyan-400 hover:to-blue-500 transition-all shadow-lg shadow-cyan-500/25 hover:scale-105 active:scale-95"
                            >
                              <ScanLine className="w-4 h-4" /> Analyze Fish
                            </button>
                            <div {...getRootProps()} className="inline-block">
                              <input {...getInputProps()} />
                              <span className="inline-flex items-center gap-2 px-5 py-3 rounded-xl border border-cyan-800/50 text-cyan-300 text-sm font-medium bg-sky-100/40 hover:bg-cyan-950/50 transition-colors cursor-pointer">
                                <RefreshCcw className="w-4 h-4" /> Replace
                              </span>
                            </div>
                          </div>
                        </>
                      ) : (
                        <Pipeline3DAnimation currentStage={stage} />
                      )}
                    </div>
                  </div>
                </motion.div>
              )}

              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -6 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-red-50 border border-red-200 text-red-600 text-sm rounded-xl px-4 py-3 flex items-center gap-2 backdrop-blur-sm"
                >
                  <AlertTriangle className="w-4 h-4 shrink-0" /> {error}
                </motion.div>
              )}
            </motion.div>
          )}

          {/* ── Result ── */}
          {result && (
            <motion.div
              key="result"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <PredictionResult3D
                result={result}
                diseaseInfo={diseaseInfo}
                onReset={clearAll}
                preview={preview}
              />
            </motion.div>
          )}

        </AnimatePresence>
      </div>
    </div>
  );
}

// ─── 3D Pipeline Animation ────────────────────────────────────────────────────
function Pipeline3DAnimation({ currentStage }) {
  return (
    <div>
      <h3 className="font-semibold text-slate-800 text-lg mb-4 flex items-center gap-2">
        <Fish className="w-5 h-5 text-cyan-400 animate-pulse" />
        Analyzing your fish...
      </h3>
      <div className="space-y-2">
        {STAGES.map((s, i) => {
          const done   = i < currentStage;
          const active = i === currentStage;
          return (
            <motion.div
              key={s.key}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.03 }}
              className={`flex items-center gap-3 rounded-xl px-3 py-2.5 transition-all ${
                active ? "bg-sky-100/70 border border-sky-200" : ""
              }`}
            >
              <div
                className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 transition-all"
                style={{
                  background: done
                    ? "rgba(0,255,136,0.15)"
                    : active
                    ? `rgba(${s.color}, 0.15)`
                    : "rgba(255,255,255,0.05)",
                  color: done ? "#00ff88" : active ? s.color : "#334155",
                  boxShadow: active ? `0 0 12px ${s.color}40` : "none",
                }}
              >
                {done ? <CheckCircle2 className="w-4 h-4" /> :
                 active ? <s.icon className="w-4 h-4 animate-pulse" /> :
                 <s.icon className="w-4 h-4" />}
              </div>
              <span className={`text-sm font-medium ${
                done ? "text-green-600/80" : active ? "text-slate-800" : "text-slate-600"
              }`}>
                {s.label}
              </span>
              {active && (
                <span className="ml-auto flex gap-1">
                  {[0, 1, 2].map((d) => (
                    <motion.span
                      key={d}
                      className="w-1.5 h-1.5 rounded-full"
                      style={{ background: s.color }}
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

// ─── 3D Prediction Result ─────────────────────────────────────────────────────
function PredictionResult3D({ result, diseaseInfo, onReset, preview }) {
  const isHealthy = diseaseInfo?.isHealthy;
  const accentColor = isHealthy ? "#00ff88" : "#ff6b35";
  const accentBg    = isHealthy ? "rgba(0,255,136,0.1)" : "rgba(255,107,53,0.1)";
  const accentBorder= isHealthy ? "border-green-500/30" : "border-orange-500/30";

  return (
    <div className="space-y-6">

      {/* Banner */}
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ type: "spring", stiffness: 200, damping: 18 }}
        className={`rounded-3xl p-6 sm:p-8 border ${accentBorder} backdrop-blur-sm`}
        style={{ background: accentBg, boxShadow: `0 0 40px ${accentColor}15` }}
      >
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div
              className="w-14 h-14 rounded-2xl flex items-center justify-center shrink-0"
              style={{ background: `${accentColor}20`, color: accentColor }}
            >
              {isHealthy
                ? <CheckCircle2 className="w-7 h-7" />
                : <AlertTriangle className="w-7 h-7" />
              }
            </div>
            <div>
              <div className="text-xs uppercase tracking-wider text-slate-500 font-semibold mb-0.5">
                Diagnosis Result
              </div>
              <h2 className="text-2xl font-extrabold text-slate-800">{diseaseInfo?.name}</h2>
            </div>
          </div>
          <div className="text-right">
            <div className="text-3xl font-extrabold" style={{ color: accentColor }}>
              {result.confidence_pct}%
            </div>
            <div className="text-xs text-slate-500">Confidence</div>
          </div>
        </div>
      </motion.div>

      <div className="grid lg:grid-cols-3 gap-6">

        {/* Left: image + scores */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className={`rounded-2xl p-5 space-y-4 border ${accentBorder} backdrop-blur-sm`}
          style={{ background: "rgba(255,255,255,0.92)" }}
        >
          <img src={preview} alt="Fish" className="w-full h-48 object-cover rounded-xl shadow-lg" />
          <div className="flex items-center gap-2 text-slate-400 text-sm">
            <Clock className="w-4 h-4" />
            {formatDateTime(result.created_at || new Date().toISOString())}
          </div>
          <div>
            <div className="text-xs text-slate-400 uppercase tracking-wide font-semibold mb-2">
              Confidence Breakdown
            </div>
            <div className="space-y-2">
              {Object.entries(result.all_scores || {}).map(([cls, score]) => {
                const info = getDiseaseInfo(cls);
                const isSelected = cls === result.predicted_class;
                return (
                  <div key={cls}>
                    <div className="flex justify-between text-xs text-slate-500 mb-1">
                      <span>{info.shortName}</span>
                      <span>{(score * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${score * 100}%` }}
                        transition={{ duration: 0.8, delay: 0.2 }}
                        className="h-full rounded-full"
                        style={{ background: isSelected ? accentColor : "#1e3a5f" }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </motion.div>

        {/* Right: details */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className={`lg:col-span-2 rounded-2xl p-6 space-y-5 border ${accentBorder} backdrop-blur-sm`}
          style={{ background: "rgba(255,255,255,0.92)" }}
        >
          <div>
            <h3 className="font-semibold text-slate-800 mb-2 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-cyan-400" /> What This Means
            </h3>
            <p className="text-slate-500 text-sm leading-relaxed">{diseaseInfo?.summary}</p>
          </div>

          {!isHealthy && diseaseInfo?.symptoms?.length > 0 && (
            <div>
              <h3 className="font-semibold text-slate-800 mb-2">Common Symptoms</h3>
              <ul className="space-y-1.5">
                {diseaseInfo.symptoms.slice(0, 3).map((s) => (
                  <li key={s} className="flex items-start gap-2 text-sm text-blue-200/50">
                    <ChevronRight className="w-4 h-4 shrink-0 mt-0.5" style={{ color: accentColor }} />
                    {s}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div>
            <h3 className="font-semibold text-slate-800 mb-2">Recommended Next Steps</h3>
            <ul className="space-y-1.5">
              {diseaseInfo?.management?.slice(0, isHealthy ? 3 : 4).map((s) => (
                <li key={s} className="flex items-start gap-2 text-sm text-blue-200/50">
                  <CheckCircle2 className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" /> {s}
                </li>
              ))}
            </ul>
          </div>

          <div
            className="rounded-xl px-4 py-3 text-xs text-slate-400 leading-relaxed border border-sky-200"
            style={{ background: "rgba(0,20,40,0.5)" }}
          >
            <strong className="text-blue-300/60">Disclaimer:</strong> This result is generated by an
            AI-assisted model and is intended to support, not replace, professional judgment.
          </div>

          <button
            onClick={onReset}
            className="inline-flex items-center gap-2 px-6 py-3 rounded-xl font-semibold text-sm bg-gradient-to-r from-sky-500 to-blue-600 text-slate-800 hover:from-cyan-400 hover:to-blue-500 transition-all shadow-lg shadow-cyan-500/25 hover:scale-105 active:scale-95"
          >
            <RefreshCcw className="w-4 h-4" /> Analyze Another Image
          </button>
        </motion.div>
      </div>
    </div>
  );
}

