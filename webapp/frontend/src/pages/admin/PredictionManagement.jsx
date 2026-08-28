import { useEffect, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search, Filter, ChevronLeft, ChevronRight, Eye, X, ClipboardList,
  ArrowUpDown, CheckCircle2, AlertTriangle,
} from "lucide-react";
import { getAdminPredictions } from "../../api/endpoints";
import { getDiseaseInfo, DISEASE_ORDER } from "../../utils/diseaseInfo";
import { formatDateTime } from "../../utils/format";

const PAGE_SIZE = 15;

const SORT_OPTIONS = [
  { value: "date_desc", label: "Newest First" },
  { value: "date_asc", label: "Oldest First" },
  { value: "confidence_desc", label: "Highest Confidence" },
  { value: "confidence_asc", label: "Lowest Confidence" },
];

export default function PredictionManagement() {
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [disease, setDisease] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [sort, setSort] = useState("date_desc");
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await getAdminPredictions({
        page,
        page_size: PAGE_SIZE,
        search: search || undefined,
        disease: disease !== "all" ? disease : undefined,
        status: statusFilter !== "all" ? statusFilter : undefined,
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
  }, [page, search, disease, statusFilter, sort]);

  useEffect(() => {
    load();
  }, [load]);

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="font-display text-2xl sm:text-3xl font-bold text-slate-900">Prediction Management</h1>
        <p className="text-slate-500 mt-1">Browse and filter every prediction made across the platform.</p>
      </motion.div>

      <div className="glass-card rounded-2xl p-4 flex flex-col sm:flex-row gap-3 flex-wrap">
        <div className="relative flex-1 min-w-[180px]">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
          <input
            value={search}
            onChange={(e) => {
              setPage(1);
              setSearch(e.target.value);
            }}
            placeholder="Search by user name or email..."
            className="input-field pl-10 !py-2.5"
          />
        </div>
        <select value={disease} onChange={(e) => { setPage(1); setDisease(e.target.value); }} className="input-field !py-2.5 sm:w-44 cursor-pointer">
          <option value="all">All Diagnoses</option>
          {DISEASE_ORDER.map((k) => (
            <option key={k} value={k}>{getDiseaseInfo(k).shortName}</option>
          ))}
        </select>
        <select value={statusFilter} onChange={(e) => { setPage(1); setStatusFilter(e.target.value); }} className="input-field !py-2.5 sm:w-36 cursor-pointer">
          <option value="all">All Status</option>
          <option value="success">Success</option>
          <option value="error">Error</option>
        </select>
        <select value={sort} onChange={(e) => setSort(e.target.value)} className="input-field !py-2.5 sm:w-44 cursor-pointer">
          {SORT_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
      </div>

      <div className="glass-card rounded-2xl overflow-hidden">
        {loading ? (
          <div className="text-center py-16 text-slate-400 text-sm">Loading predictions...</div>
        ) : items.length === 0 ? (
          <div className="text-center py-16">
            <ClipboardList className="w-10 h-10 text-slate-200 mx-auto mb-3" />
            <p className="text-slate-500 text-sm">No predictions match your filters.</p>
          </div>
        ) : (
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-200 text-left">
                <th className="px-5 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wide">Image</th>
                <th className="px-5 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wide">User</th>
                <th className="px-5 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wide">Prediction</th>
                <th className="px-5 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wide hidden md:table-cell">Confidence</th>
                <th className="px-5 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wide hidden lg:table-cell">Date</th>
                <th className="px-5 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wide">Status</th>
                <th className="px-5 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wide"></th>
              </tr>
            </thead>
            <tbody>
              {items.map((p) => {
                const info = getDiseaseInfo(p.predicted_class);
                return (
                  <tr key={p.id} className="border-b border-slate-100 hover:bg-slate-50 transition-colors">
                    <td className="px-5 py-3">
                      <img src={p.image_url} alt="" className="w-10 h-10 rounded-lg object-cover bg-slate-100" />
                    </td>
                    <td className="px-5 py-3">
                      <div className="text-sm font-medium text-slate-900">{p.user_name}</div>
                      <div className="text-xs text-slate-400">{p.user_email}</div>
                    </td>
                    <td className="px-5 py-3 text-sm text-slate-900">{info.name}</td>
                    <td className="px-5 py-3 hidden md:table-cell">
                      <span className={`text-sm font-semibold ${info.isHealthy ? "text-seaweed-600" : "text-coral-600"}`}>
                        {p.confidence_pct}%
                      </span>
                    </td>
                    <td className="px-5 py-3 text-sm text-slate-500 hidden lg:table-cell">{formatDateTime(p.created_at)}</td>
                    <td className="px-5 py-3">
                      <StatusBadge status={p.status} />
                    </td>
                    <td className="px-5 py-3 text-right">
                      <button onClick={() => setSelected(p)} className="text-ocean-600 hover:text-ocean-700 text-sm font-medium inline-flex items-center gap-1">
                        <Eye className="w-4 h-4" /> View
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <span className="text-sm text-slate-400">Page {page} of {totalPages} &bull; {total} total</span>
          <div className="flex gap-2">
            <button onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={page === 1} className="btn-secondary !px-3 !py-2 disabled:opacity-30">
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button onClick={() => setPage((p) => Math.min(totalPages, p + 1))} disabled={page === totalPages} className="btn-secondary !px-3 !py-2 disabled:opacity-30">
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      <AnimatePresence>
        {selected && <DetailModal prediction={selected} onClose={() => setSelected(null)} />}
      </AnimatePresence>
    </div>
  );
}

function StatusBadge({ status }) {
  if (status === "success") {
    return (
      <span className="inline-flex items-center gap-1 text-xs font-semibold text-seaweed-600">
        <CheckCircle2 className="w-3.5 h-3.5" /> Success
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 text-xs font-semibold text-coral-600">
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
          <h3 className="font-display text-xl font-bold text-slate-900">Prediction Detail</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-700">
            <X className="w-5 h-5" />
          </button>
        </div>
        <img src={prediction.image_url} alt="" className="w-full h-52 object-cover rounded-2xl mb-5" />
        <div className="space-y-3">
          <Row label="User" value={prediction.user_name} />
          <Row label="Email" value={prediction.user_email} />
          <Row label="Diagnosis" value={info.name} />
          <Row label="Confidence" value={`${prediction.confidence_pct}%`} valueClass={info.isHealthy ? "text-seaweed-600" : "text-coral-600"} />
          <Row label="Date" value={formatDateTime(prediction.created_at)} />
          <Row label="Status" value={<StatusBadge status={prediction.status} />} />
          {prediction.error_message && <Row label="Error" value={prediction.error_message} valueClass="text-coral-600" />}
        </div>
        {prediction.all_scores && (
          <div className="mt-5">
            <div className="text-xs text-slate-400 uppercase tracking-wide font-semibold mb-2">All Scores</div>
            <div className="space-y-2">
              {Object.entries(prediction.all_scores).map(([cls, score]) => (
                <div key={cls} className="flex justify-between text-sm">
                  <span className="text-slate-500">{getDiseaseInfo(cls).shortName}</span>
                  <span className="text-slate-900 font-medium">{(score * 100).toFixed(1)}%</span>
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
      <span className="text-slate-400">{label}</span>
      <span className={`font-medium text-slate-900 ${valueClass}`}>{value}</span>
    </div>
  );
}
