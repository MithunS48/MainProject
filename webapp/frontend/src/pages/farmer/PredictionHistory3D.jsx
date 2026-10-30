import { useEffect, useState, useCallback, useRef, Suspense } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Float, Stars } from "@react-three/drei";
import { motion, AnimatePresence } from "framer-motion";
import * as THREE from "three";
import {
  Search, Filter, ChevronLeft, ChevronRight, Eye, X, Fish,
  ArrowUpDown, CheckCircle2, AlertTriangle, Clock, TrendingUp,
  Activity, Layers, BarChart3, Download,
} from "lucide-react";
import { getMyPredictions } from "../../api/endpoints";
import { getDiseaseInfo, DISEASE_ORDER } from "../../utils/diseaseInfo";
import { formatDateTime } from "../../utils/format";
import { exportPredictionReport } from "../../utils/exportPdf";
import { useAuth } from "../../context/AuthContext";

const PAGE_SIZE = 10;
const SORT_OPTIONS = [
  { value: "date_desc",       label: "Newest First"        },
  { value: "date_asc",        label: "Oldest First"        },
  { value: "confidence_desc", label: "Highest Confidence"  },
  { value: "confidence_asc",  label: "Lowest Confidence"   },
];

// ─── 3D Floating Stat Card ────────────────────────────────────────────────────
function StatOrb({ position, color, index }) {
  const mesh = useRef();
  useFrame((state) => {
    const t = state.clock.elapsedTime;
    mesh.current.rotation.x = t * 0.3 + index;
    mesh.current.rotation.y = t * 0.5 + index;
    mesh.current.position.y = position[1] + Math.sin(t * 0.6 + index) * 0.2;
  });
  return (
    <Float speed={1.2} floatIntensity={0.3}>
      <mesh ref={mesh} position={position}>
        <icosahedronGeometry args={[0.3, 1]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.4}
          roughness={0.1}
          metalness={0.9}
          wireframe
        />
      </mesh>
    </Float>
  );
}

// ─── 3D Bar chart bars (history visualization) ────────────────────────────────
function Bar3D({ position, height, color }) {
  const mesh = useRef();
  useFrame((state) => {
    const t = state.clock.elapsedTime;
    mesh.current.scale.y = height * (0.9 + Math.sin(t * 0.5 + position[0]) * 0.05);
  });
  return (
    <mesh ref={mesh} position={[position[0], position[1] + height / 2, position[2]]}>
      <boxGeometry args={[0.35, height, 0.35]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={0.3}
        roughness={0.2}
        metalness={0.7}
        transparent
        opacity={0.8}
      />
    </mesh>
  );
}

// ─── Flowing data streams ─────────────────────────────────────────────────────
function DataStream({ startX, color }) {
  const points = useRef([]);
  const lineRef = useRef();

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    if (lineRef.current) {
      lineRef.current.position.x = ((t * 0.5 + startX / 5) % 1) * 20 - 10;
      lineRef.current.material.opacity = 0.15 + Math.sin(t * 2 + startX) * 0.1;
    }
  });

  const pts = Array.from({ length: 20 }, (_, i) => new THREE.Vector3(i * 0.3, 0, 0));
  const geo = new THREE.BufferGeometry().setFromPoints(pts);

  return (
    <line ref={lineRef} geometry={geo} position={[startX, (Math.random() - 0.5) * 4, -3]}>
      <lineBasicMaterial color={color} transparent opacity={0.2} />
    </line>
  );
}

// ─── Background Canvas ────────────────────────────────────────────────────────
function HistoryCanvas({ items }) {
  const bars = [
    { pos: [-4, -1.5, -4], h: 1.2, color: "#00d4ff" },
    { pos: [-2.5, -1.5, -4], h: 2.0, color: "#00ff88" },
    { pos: [-1, -1.5, -4], h: 0.8, color: "#a855f7" },
    { pos: [0.5, -1.5, -4], h: 1.6, color: "#ff6b35" },
    { pos: [2, -1.5, -4], h: 2.4, color: "#00d4ff" },
    { pos: [3.5, -1.5, -4], h: 1.0, color: "#00ff88" },
    { pos: [5, -1.5, -4], h: 1.8, color: "#ffd60a" },
  ];

  const orbs = [
    { pos: [-6, 2, -3], color: "#00d4ff" },
    { pos: [6, -2, -3], color: "#00ff88" },
    { pos: [-5, -2, -2], color: "#a855f7" },
    { pos: [5, 2, -2], color: "#ff6b35" },
  ];

  return (
    <Canvas camera={{ position: [0, 0, 8], fov: 55 }} dpr={[1, 1.5]} gl={{ alpha: true }} style={{ pointerEvents: "none" }}>
      <Suspense fallback={null}>
        <ambientLight intensity={0.3} color="#0077b6" />
        <pointLight position={[0, 5, 5]} intensity={1.2} color="#00d4ff" />
        <Stars radius={60} depth={30} count={300} factor={2} fade />

        {bars.map((b, i) => (
          <Bar3D key={i} position={b.pos} height={b.h} color={b.color} />
        ))}

        {orbs.map((o, i) => (
          <StatOrb key={i} position={o.pos} color={o.color} index={i} />
        ))}

        {[-8, -5, -2, 2, 5, 8].map((x, i) => (
          <DataStream key={i} startX={x} color={["#00d4ff", "#00ff88", "#a855f7"][i % 3]} />
        ))}

        {/* Ground plane with glow */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -2.5, -3]}>
          <planeGeometry args={[20, 10]} />
          <meshStandardMaterial color="#e8f4fd" roughness={1} />
        </mesh>
      </Suspense>
    </Canvas>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────
export default function PredictionHistory3D() {
  const [items, setItems]     = useState([]);
  const [total, setTotal]     = useState(0);
  const [page, setPage]       = useState(1);
  const [search, setSearch]   = useState("");
  const [disease, setDisease] = useState("all");
  const [sort, setSort]       = useState("date_desc");
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState(null);
  const [exporting, setExporting] = useState(false);
  const { user } = useAuth();

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await getMyPredictions({
        page, page_size: PAGE_SIZE,
        search: search || undefined,
        disease: disease !== "all" ? disease : undefined,
        sort,
      });
      setItems(res.data.items || []);
      setTotal(res.data.total || 0);
    } catch {
      setItems([]); setTotal(0);
    } finally {
      setLoading(false);
    }
  }, [page, search, disease, sort]);

  useEffect(() => { load(); }, [load]);

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  // Stats derived from items
  const successCount  = items.filter((i) => i.status === "success").length;
  const healthyCount  = items.filter((i) => getDiseaseInfo(i.predicted_class)?.isHealthy).length;
  const avgConfidence = items.length
    ? (items.reduce((s, i) => s + (i.confidence_pct || 0), 0) / items.length).toFixed(1)
    : "—";

  return (
    <div className="relative min-h-[calc(100vh-4rem)] bg-[#e8f4fd]">

      {/* 3D Background — pointer-events none so all nav/clicks work normally */}
      <div className="fixed inset-0 pointer-events-none z-0 opacity-60">
        <HistoryCanvas items={items} />
      </div>

      {/* Content */}
      <div className="relative z-10 space-y-6 px-4 py-6 max-w-6xl mx-auto">

        {/* Header */}
        <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-800 flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/30">
                  <BarChart3 className="w-5 h-5 text-slate-800" />
                </div>
                Prediction History
              </h1>
              <p className="text-slate-500 mt-1 text-sm">
                Review, search, and filter all your past fish disease analyses.
              </p>
            </div>
            <button
              onClick={async () => {
                setExporting(true);
                try {
                  const res = await getMyPredictions({ page: 1, page_size: 500, sort });
                  await exportPredictionReport(res.data.items || [], user);
                } finally { setExporting(false); }
              }}
              disabled={exporting || total === 0}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl border border-sky-200 bg-white text-sky-700 text-sm font-medium hover:bg-sky-50 disabled:opacity-40 transition-colors shadow-sm"
            >
              <Download className="w-4 h-4" />
              {exporting ? "Generating…" : "Export PDF"}
            </button>
          </div>
        </motion.div>

        {/* Quick stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {[
            { icon: Activity,   label: "Total",      value: total,          color: "#00d4ff" },
            { icon: CheckCircle2, label: "Successful", value: successCount, color: "#00ff88" },
            { icon: Fish,       label: "Healthy",    value: healthyCount,   color: "#a855f7" },
            { icon: TrendingUp, label: "Avg. Conf.",  value: `${avgConfidence}%`, color: "#ff6b35" },
          ].map((s, i) => (
            <motion.div
              key={s.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.08 }}
              whileHover={{ scale: 1.04, y: -4 }}
              className="rounded-2xl p-4 border border-sky-200 backdrop-blur-sm"
              style={{
                background: "rgba(255,255,255,0.92)",
                boxShadow: `0 0 20px ${s.color}10`,
              }}
            >
              <div
                className="w-9 h-9 rounded-xl flex items-center justify-center mb-2"
                style={{ background: `${s.color}20`, color: s.color }}
              >
                <s.icon className="w-4 h-4" />
              </div>
              <div className="text-2xl font-extrabold text-slate-800">{s.value}</div>
              <div className="text-xs text-slate-500 mt-0.5">{s.label}</div>
            </motion.div>
          ))}
        </div>

        {/* Filters */}
        <div
          className="rounded-2xl p-4 flex flex-col sm:flex-row gap-3 border border-sky-200 backdrop-blur-sm"
          style={{ background: "rgba(255,255,255,0.92)" }}
        >
          <div className="relative flex-1">
            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-cyan-500/50" />
            <input
              value={search}
              onChange={(e) => { setPage(1); setSearch(e.target.value); }}
              placeholder="Search by diagnosis..."
              className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-sky-100/60 border border-sky-200 text-slate-800 placeholder-slate-400 text-sm focus:outline-none focus:border-cyan-500/50 transition-colors"
            />
          </div>
          <div className="relative">
            <Filter className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-cyan-500/50 pointer-events-none" />
            <select
              value={disease}
              onChange={(e) => { setPage(1); setDisease(e.target.value); }}
              className="pl-10 pr-4 py-2.5 rounded-xl bg-sky-100/60 border border-sky-200 text-slate-800 text-sm focus:outline-none focus:border-cyan-500/50 appearance-none cursor-pointer sm:w-44"
            >
              <option value="all">All Diagnoses</option>
              {DISEASE_ORDER.map((k) => (
                <option key={k} value={k}>{getDiseaseInfo(k).shortName}</option>
              ))}
            </select>
          </div>
          <div className="relative">
            <ArrowUpDown className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-cyan-500/50 pointer-events-none" />
            <select
              value={sort}
              onChange={(e) => setSort(e.target.value)}
              className="pl-10 pr-4 py-2.5 rounded-xl bg-sky-100/60 border border-sky-200 text-slate-800 text-sm focus:outline-none focus:border-cyan-500/50 appearance-none cursor-pointer sm:w-44"
            >
              {SORT_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Table */}
        <div
          className="rounded-2xl overflow-hidden border border-sky-200 backdrop-blur-sm"
          style={{ background: "rgba(255,255,255,0.92)" }}
        >
          {loading ? (
            <div className="text-center py-16 text-slate-400 text-sm">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ repeat: Infinity, duration: 1.5, ease: "linear" }}
                className="w-8 h-8 border-2 border-cyan-500/40 border-t-cyan-400 rounded-full mx-auto mb-3"
              />
              Loading predictions...
            </div>
          ) : items.length === 0 ? (
            <div className="text-center py-16">
              <Fish className="w-10 h-10 text-cyan-900 mx-auto mb-3" />
              <p className="text-slate-400 text-sm">No predictions match your filters.</p>
            </div>
          ) : (
            <>
              {/* Desktop table */}
              <table className="w-full hidden md:table">
                <thead>
                  <tr className="border-b border-sky-200">
                    {["Image", "Prediction", "Confidence", "Date", "Status", ""].map((h) => (
                      <th key={h} className="px-5 py-3 text-left text-xs font-semibold text-slate-400 uppercase tracking-wide">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {items.map((p, idx) => {
                    const info = getDiseaseInfo(p.predicted_class);
                    return (
                      <motion.tr
                        key={p.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.04 }}
                        className="border-b border-cyan-900/20 hover:bg-cyan-950/30 transition-colors cursor-pointer"
                        onClick={() => setSelected(p)}
                      >
                        <td className="px-5 py-3">
                          <img
                            src={p.image_url}
                            alt=""
                            className="w-10 h-10 rounded-lg object-cover bg-sky-100 border border-sky-200"
                          />
                        </td>
                        <td className="px-5 py-3 text-sm font-medium text-slate-800">{info.name}</td>
                        <td className="px-5 py-3">
                          <span
                            className="text-sm font-semibold"
                            style={{ color: info.isHealthy ? "#00ff88" : "#ff6b35" }}
                          >
                            {p.confidence_pct}%
                          </span>
                        </td>
                        <td className="px-5 py-3 text-sm text-slate-500">{formatDateTime(p.created_at)}</td>
                        <td className="px-5 py-3">
                          <StatusBadge3D status={p.status} />
                        </td>
                        <td className="px-5 py-3 text-right">
                          <button
                            onClick={(e) => { e.stopPropagation(); setSelected(p); }}
                            className="text-cyan-400 hover:text-cyan-300 text-sm font-medium inline-flex items-center gap-1 transition-colors"
                          >
                            <Eye className="w-4 h-4" /> View
                          </button>
                        </td>
                      </motion.tr>
                    );
                  })}
                </tbody>
              </table>

              {/* Mobile cards */}
              <div className="md:hidden divide-y divide-cyan-900/20">
                {items.map((p, idx) => {
                  const info = getDiseaseInfo(p.predicted_class);
                  return (
                    <motion.div
                      key={p.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: idx * 0.05 }}
                      className="p-4 flex items-center gap-3 hover:bg-cyan-950/30 transition-colors"
                      onClick={() => setSelected(p)}
                    >
                      <img
                        src={p.image_url}
                        alt=""
                        className="w-14 h-14 rounded-xl object-cover bg-sky-100 border border-sky-200 shrink-0"
                      />
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium text-slate-800 truncate">{info.name}</div>
                        <div className="text-xs text-slate-400">{formatDateTime(p.created_at)}</div>
                        <StatusBadge3D status={p.status} className="mt-1" />
                      </div>
                      <div
                        className="text-sm font-semibold shrink-0"
                        style={{ color: info.isHealthy ? "#00ff88" : "#ff6b35" }}
                      >
                        {p.confidence_pct}%
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </>
          )}
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between">
            <span className="text-sm text-slate-400">
              Page {page} of {totalPages} · {total} total
            </span>
            <div className="flex gap-2">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
                className="p-2 rounded-xl border border-sky-200 text-cyan-400 hover:bg-sky-100/60 disabled:opacity-30 transition-colors"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <button
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page === totalPages}
                className="p-2 rounded-xl border border-sky-200 text-cyan-400 hover:bg-sky-100/60 disabled:opacity-30 transition-colors"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Detail modal */}
      <AnimatePresence>
        {selected && (
          <DetailModal3D prediction={selected} onClose={() => setSelected(null)} />
        )}
      </AnimatePresence>
    </div>
  );
}

// ─── Status Badge ─────────────────────────────────────────────────────────────
function StatusBadge3D({ status, className = "" }) {
  if (status === "success") {
    return (
      <span className={`inline-flex items-center gap-1 text-xs font-semibold text-green-400 ${className}`}>
        <CheckCircle2 className="w-3.5 h-3.5" /> Success
      </span>
    );
  }
  return (
    <span className={`inline-flex items-center gap-1 text-xs font-semibold text-orange-400 ${className}`}>
      <AlertTriangle className="w-3.5 h-3.5" /> Error
    </span>
  );
}

// ─── Detail Modal ─────────────────────────────────────────────────────────────
function DetailModal3D({ prediction, onClose }) {
  const info = getDiseaseInfo(prediction.predicted_class);
  const accentColor = info.isHealthy ? "#00ff88" : "#ff6b35";

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0, y: 20 }}
        animate={{ scale: 1, opacity: 1, y: 0 }}
        exit={{ scale: 0.9, opacity: 0, y: 20 }}
        transition={{ type: "spring", stiffness: 260, damping: 22 }}
        onClick={(e) => e.stopPropagation()}
        className="rounded-3xl p-6 sm:p-8 max-w-lg w-full max-h-[85vh] overflow-y-auto border border-cyan-900/50 backdrop-blur-xl"
        style={{
          background: "rgba(255,255,255,0.97)",
          boxShadow: `0 0 60px ${accentColor}20`,
        }}
      >
        <div className="flex items-center justify-between mb-5">
          <h3 className="text-xl font-extrabold text-slate-800">Prediction Detail</h3>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-lg bg-sky-100/60 border border-sky-200 flex items-center justify-center text-slate-500 hover:text-slate-800 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <img
          src={prediction.image_url}
          alt=""
          className="w-full h-52 object-cover rounded-2xl mb-5 border border-sky-200"
        />

        {/* Confidence arc */}
        <div
          className="rounded-2xl p-4 mb-4 border flex items-center justify-between"
          style={{ borderColor: `${accentColor}30`, background: `${accentColor}08` }}
        >
          <div>
            <div className="text-xs text-slate-400 uppercase tracking-wide font-semibold">Diagnosis</div>
            <div className="text-slate-800 font-bold mt-0.5">{info.name}</div>
          </div>
          <div className="text-right">
            <div className="text-3xl font-extrabold" style={{ color: accentColor }}>
              {prediction.confidence_pct}%
            </div>
            <div className="text-xs text-slate-400">Confidence</div>
          </div>
        </div>

        <div className="space-y-3">
          <Row3D label="Date"   value={formatDateTime(prediction.created_at)} />
          <Row3D label="Status" value={<StatusBadge3D status={prediction.status} />} />
          {prediction.error_message && (
            <Row3D label="Error" value={prediction.error_message} valueClass="text-orange-400" />
          )}
        </div>

        {prediction.all_scores && (
          <div className="mt-5">
            <div className="text-xs text-slate-400 uppercase tracking-wide font-semibold mb-3">
              All Class Scores
            </div>
            <div className="space-y-2">
              {Object.entries(prediction.all_scores).map(([cls, score]) => {
                const d = getDiseaseInfo(cls);
                const isSelected = cls === prediction.predicted_class;
                return (
                  <div key={cls}>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-500">{d.shortName}</span>
                      <span className="text-slate-800 font-medium">{(score * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-sky-100 overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${score * 100}%` }}
                        transition={{ duration: 0.8 }}
                        className="h-full rounded-full"
                        style={{ background: isSelected ? accentColor : "#1e3a5f" }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </motion.div>
    </motion.div>
  );
}

function Row3D({ label, value, valueClass = "" }) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-slate-400">{label}</span>
      <span className={`font-medium text-slate-800 ${valueClass}`}>{value}</span>
    </div>
  );
}

