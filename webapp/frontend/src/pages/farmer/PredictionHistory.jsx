import { useEffect, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search, Filter, ChevronLeft, ChevronRight, Eye, X, Fish,
  ArrowUpDown, CheckCircle2, AlertTriangle, Clock,
} from "lucide-react";
import { getMyPredictions } from "../../api/endpoints";
import { getDiseaseInfo, DISEASE_ORDER } from "../../utils/diseaseInfo";
import { formatDateTime } from "../../utils/format";

const PAGE_SIZE = 10;

const SORT_OPTIONS = [
  { value: "date_desc", label: "Newest First" },
  { value: "date_asc", label: "Oldest First" },
  { value: "confidence_desc", label: "Highest Confidence" },
  { value: "confidence_asc", label: "Lowest Confidence" },
];

export default function PredictionHistory() {
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [disease, setDisease] = useState("all");
  const [sort, setSort] = useState("date_desc");
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await getMyPredictions({
        page,
        page_size: PAGE_SIZE,
        search: search || undefined,
        disease: disease !== "all" ? disease : undefined,
        sort,
      });
      setItems(res.data.items || []);
      setTotal(res.data.total || 0);
    } catch {
      setItems([]);
      setTotal(0);
    } finally {
      setLoading(false);
    }
  }, [page, search, disease, sort]);

  useEffect(() => {
    load();
  }, [load]);

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="font-display text-2xl sm:text-3xl font-bold text-white">Prediction History</h1>
        <p className="text-white/60 mt-1">Review, search, and filter all your past fish disease analyses.</p>
      </motion.div>

      {/* Filters */}
      <div className="glass-card rounded-2xl p-4 flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-white/40" />
          <input
            value={search}
            onChange={(e) => {
              setPage(1);
              setSearch(e.target.value);
            }}
            placeholder="Search by diagnosis..."
            className="input-field pl-10 !py-2.5"
          />
        </div>
        <div className="relative">
          <Filter className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-white/40 pointer-events-none" />
          <select
            value={disease}
            onChange={(e) => {
              setPage(1);
              setDisease(e.target.value);
            }}
            className="input-field pl-10 !py-2.5 appearance-none cursor-pointer sm:w-48"
          >
            <option value="all">All Diagnoses</option>
            {DISEASE_ORDER.map((k) => (
              <option key={k} value={k}>
                {getDiseaseInfo(k).shortName}
              </option>
            ))}
          </select>
        </div>
        <div className="relative">
          <ArrowUpDown className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-white/40 pointer-events-none" />
          <select
            value={sort}
            onChange={(e) => setSort(e.target.value)}
            className="input-field pl-10 !py-2.5 appearance-none cursor-pointer sm:w-48"
          >
            {SORT_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Table / cards */}
      <div className="glass-card rounded-2xl overflow-hidden">
        {loading ? (
          <div className="text-center py-16 text-white/40 text-sm">Loading predictions...</div>
        ) : items.length === 0 ? (
          <div className="text-center py-16">
            <Fish className="w-10 h-10 text-white/20 mx-auto mb-3" />
            <p className="text-white/50 text-sm">No predictions match your filters.</p>
          </div>
        ) : (
          <>
            {/* Desktop table */}
            <table className="w-full hidden md:table">
              <thead>
                <tr className="border-b border-white/10 text-left">
                  <th className="px-5 py-3 text-xs font-semibold text-white/40 uppercase tracking-wide">Image</th>
                  <th className="px-5 py-3 text-xs font-semibold text-white/40 uppercase tracking-wide">Prediction</th>
                  <th className="px-5 py-3 text-xs font-semibold text-white/40 uppercase tracking-wide">Confidence</th>
                  <th className="px-5 py-3 text-xs font-semibold text-white/40 uppercase tracking-wide">Date</th>
                  <th className="px-5 py-3 text-xs font-semibold text-white/40 uppercase tracking-wide">Status</th>
                  <th className="px-5 py-3 text-xs font-semibold text-white/40 uppercase tracking-wide"></th>
                </tr>
              </thead>
              <tbody>
                {items.map((p) => {
                  const info = getDiseaseInfo(p.predicted_class);
                  return (
                    <tr key={p.id} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                      <td className="px-5 py-3">
                        <img src={p.image_url} alt="" className="w-10 h-10 rounded-lg object-cover bg-white/5" />
                      </td>
                      <td className="px-5 py-3 text-sm font-medium text-white">{info.name}</td>
                      <td className="px-5 py-3">
                        <span className={`text-sm font-semibold ${info.isHealthy ? "text-seaweed-400" : "text-coral-400"}`}>
                          {p.confidence_pct}%
                        </span>
                      </td>
                      <td className="px-5 py-3 text-sm text-white/50">{formatDateTime(p.created_at)}</td>
                      <td className="px-5 py-3">
                        <StatusBadge status={p.status} />
                      </td>
                      <td className="px-5 py-3 text-right">
                        <button
                          onClick={() => setSelected(p)}
                          className="text-ocean-300 hover:text-ocean-200 text-sm font-medium inline-flex items-center gap-1"
                        >
                          <Eye className="w-4 h-4" /> View
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>

            {/* Mobile cards */}
            <div className="md:hidden divide-y divide-white/5">
              {items.map((p) => {
                const info = getDiseaseInfo(p.predicted_class);
                return (
                  <div key={p.id} className="p-4 flex items-center gap-3" onClick={() => setSelected(p)}>
                    <img src={p.image_url} alt="" className="w-14 h-14 rounded-xl object-cover bg-white/5 shrink-0" />
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-white truncate">{info.name}</div>
                      <div className="text-xs text-white/40">{formatDateTime(p.created_at)}</div>
                      <StatusBadge status={p.status} className="mt-1" />
                    </div>
                    <div className={`text-sm font-semibold shrink-0 ${info.isHealthy ? "text-seaweed-400" : "text-coral-400"}`}>
                      {p.confidence_pct}%
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <span className="text-sm text-white/40">
            Page {page} of {totalPages} &bull; {total} total
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
              className="btn-secondary !px-3 !py-2 disabled:opacity-30"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
              className="btn-secondary !px-3 !py-2 disabled:opacity-30"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Detail modal */}
      <AnimatePresence>
        {selected && <DetailModal prediction={selected} onClose={() => setSelected(null)} />}
      </AnimatePresence>
    </div>
  );
}

function StatusBadge({ status, className = "" }) {
  if (status === "success") {
    return (
      <span className={`inline-flex items-center gap-1 text-xs font-semibold text-seaweed-400 ${className}`}>
        <CheckCircle2 className="w-3.5 h-3.5" /> Success
      </span>
    );
  }
  return (
    <span className={`inline-flex items-center gap-1 text-xs font-semibold text-coral-400 ${className}`}>
      <AlertTriangle className="w-3.5 h-3.5" /> Error
    </span>
  );
}

function DetailModal({ prediction, onClose }) {
  const info = getDiseaseInfo(prediction.predicted_class);
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        transition={{ type: "spring", stiffness: 260, damping: 22 }}
        onClick={(e) => e.stopPropagation()}
        className="glass-card rounded-3xl p-6 sm:p-8 max-w-lg w-full max-h-[85vh] overflow-y-auto"
      >
        <div className="flex items-center justify-between mb-5">
          <h3 className="font-display text-xl font-bold text-white">Prediction Detail</h3>
          <button onClick={onClose} className="text-white/40 hover:text-white">
            <X className="w-5 h-5" />
          </button>
        </div>

        <img src={prediction.image_url} alt="" className="w-full h-52 object-cover rounded-2xl mb-5" />

        <div className="space-y-3">
          <Row label="Diagnosis" value={info.name} />
          <Row
            label="Confidence"
            value={`${prediction.confidence_pct}%`}
            valueClass={info.isHealthy ? "text-seaweed-400" : "text-coral-400"}
          />
          <Row label="Date" value={formatDateTime(prediction.created_at)} />
          <Row label="Status" value={<StatusBadge status={prediction.status} />} />
          {prediction.error_message && <Row label="Error" value={prediction.error_message} valueClass="text-coral-300" />}
        </div>

        {prediction.all_scores && (
          <div className="mt-5">
            <div className="text-xs text-white/40 uppercase tracking-wide font-semibold mb-2">All Scores</div>
            <div className="space-y-2">
              {Object.entries(prediction.all_scores).map(([cls, score]) => (
                <div key={cls} className="flex justify-between text-sm">
                  <span className="text-white/60">{getDiseaseInfo(cls).shortName}</span>
                  <span className="text-white font-medium">{(score * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </motion.div>
    </motion.div>
  );
}

function Row({ label, value, valueClass = "" }) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-white/40">{label}</span>
      <span className={`font-medium text-white ${valueClass}`}>{value}</span>
    </div>
  );
}
