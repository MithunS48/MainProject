/**
 * WaterQuality.jsx
 * ────────────────
 * IoT Water-Quality Dashboard — Farmer view.
 *
 * Shows:
 *  • Current pH + temperature readings
 *  • pH / temperature status badges
 *  • Derived values: [H⁺] concentration, pOH
 *  • Device online/offline status + last-seen time
 *  • Configurable-threshold alerts
 *  • Historical pH chart (Recharts LineChart — same library as admin pages)
 *  • Historical temperature chart
 *  • Device registration form (if no device linked yet)
 *
 * Data flow:
 *  1. On mount, getDeviceStatus()  → populates gauges + alerts
 *  2. getIoTHistory()              → populates charts
 *  3. Every 30 s, step 1 repeats   → live polling (no WebSocket needed)
 */

import { useEffect, useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from "recharts";
import {
  Droplets, Thermometer, Wifi, WifiOff, AlertTriangle,
  CheckCircle2, RefreshCcw, Plus, Clock, FlaskConical,
  Activity, Info, ChevronDown, ChevronUp, Download,
  MapPin, Trash2, Edit2, Check, ShieldAlert, ShieldCheck,
  Lightbulb, Fish, Bug,
} from "lucide-react";
import { getWaterQualityAdvice } from "../../utils/waterQualityAdvisor";
import toast from "react-hot-toast";
import {
  getDeviceStatus, getIoTHistory, registerDevice,
  injectMockReading, injectMockBurst, clearMockReadings,
} from "../../api/endpoints";
import { formatDateTime } from "../../utils/format";
import { exportWaterQualityReport } from "../../utils/exportPdf";
import { useAuth } from "../../context/AuthContext";

// ─── constants ────────────────────────────────────────────────────────────────
const POLL_INTERVAL_MS = 30_000;
const CHART_POINTS     = 50;
const PONDS_KEY        = "iot_ponds";      // localStorage key for pond list
const ACTIVE_KEY       = "iot_device_id";  // active device id

// ─── pond list helpers ────────────────────────────────────────────────────────
function loadPonds() {
  try { return JSON.parse(localStorage.getItem(PONDS_KEY) || "[]"); }
  catch { return []; }
}
function savePonds(ponds) {
  localStorage.setItem(PONDS_KEY, JSON.stringify(ponds));
}

// ─── helpers ──────────────────────────────────────────────────────────────────
function phColor(status) {
  if (status === "acidic")   return "#f97316";
  if (status === "alkaline") return "#a855f7";
  return "#22c55e";
}

function tempColor(status) {
  if (status === "too_low")  return "#3b82f6";
  if (status === "too_high") return "#ef4444";
  return "#22c55e";
}

function statusLabel(status) {
  const map = {
    acidic: "Acidic",
    normal: "Normal",
    alkaline: "Alkaline",
    too_low: "Too Low",
    too_high: "Too High",
  };
  return map[status] ?? status;
}

function formatChartTime(dt) {
  if (!dt) return "";
  const d = new Date(dt);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// ─── sub-components ───────────────────────────────────────────────────────────

function GaugeCard({ icon: Icon, title, value, unit, status, color, secondary }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-2xl p-6 bg-white border border-sky-100 shadow-sm"
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2 text-slate-500 text-sm font-medium">
          <Icon className="w-4 h-4" /> {title}
        </div>
        <span
          className="text-xs font-semibold px-2.5 py-1 rounded-full"
          style={{ background: `${color}15`, color }}
        >
          {statusLabel(status)}
        </span>
      </div>
      <div className="text-4xl font-extrabold text-slate-800 mb-1">
        {value !== null && value !== undefined ? value : "—"}
        <span className="text-xl font-medium text-slate-400 ml-1">{unit}</span>
      </div>
      {secondary && (
        <div className="text-xs text-slate-400 mt-2 space-y-0.5">
          {secondary.map((s) => (
            <div key={s}>{s}</div>
          ))}
        </div>
      )}
    </motion.div>
  );
}

function AlertBanner({ alerts }) {
  if (!alerts || alerts.length === 0) return null;
  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-2xl p-4 bg-amber-50 border border-amber-200"
    >
      <div className="flex items-center gap-2 text-amber-700 font-semibold text-sm mb-2">
        <AlertTriangle className="w-4 h-4" /> {alerts.length} Alert{alerts.length > 1 ? "s" : ""}
      </div>
      <ul className="space-y-1">
        {alerts.map((a, i) => (
          <li key={i} className="text-sm text-amber-700 flex items-start gap-2">
            <span className="mt-0.5">•</span> {a}
          </li>
        ))}
      </ul>
    </motion.div>
  );
}

function DeviceStatusBadge({ online, lastSeen }) {
  return (
    <div className="flex items-center gap-2">
      {online ? (
        <span className="inline-flex items-center gap-1.5 text-sm font-semibold text-green-600">
          <Wifi className="w-4 h-4" />
          Online
        </span>
      ) : (
        <span className="inline-flex items-center gap-1.5 text-sm font-semibold text-slate-400">
          <WifiOff className="w-4 h-4" />
          Offline
        </span>
      )}
      {lastSeen && (
        <span className="text-xs text-slate-400 flex items-center gap-1">
          <Clock className="w-3 h-3" />
          {formatDateTime(lastSeen)}
        </span>
      )}
    </div>
  );
}

function DerivedInfo({ reading }) {
  const [open, setOpen] = useState(false);
  if (!reading) return null;
  return (
    <div className="rounded-2xl bg-white border border-sky-100 shadow-sm overflow-hidden">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between p-4 text-slate-700 text-sm font-semibold hover:bg-sky-50 transition-colors"
      >
        <span className="flex items-center gap-2">
          <FlaskConical className="w-4 h-4 text-sky-500" />
          Derived Water-Chemistry Values
        </span>
        {open ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="p-4 pt-0 grid grid-cols-2 gap-4 text-sm">
              <div className="rounded-xl bg-sky-50 p-3">
                <div className="text-xs text-slate-400 mb-1">Hydrogen-ion concentration [H⁺]</div>
                <div className="font-semibold text-slate-800">
                  {reading.h_ion_conc?.toExponential(3)} mol/L
                </div>
                <div className="text-xs text-slate-400 mt-1">= 10^(−pH)</div>
              </div>
              <div className="rounded-xl bg-sky-50 p-3">
                <div className="text-xs text-slate-400 mb-1">pOH (at ~25 °C)</div>
                <div className="font-semibold text-slate-800">{reading.poh?.toFixed(3)}</div>
                <div className="text-xs text-slate-400 mt-1">= 14 − pH (approximate)</div>
              </div>
              <div className="rounded-xl bg-sky-50 p-3 col-span-2">
                <div className="flex items-start gap-2 text-xs text-slate-500">
                  <Info className="w-3.5 h-3.5 shrink-0 mt-0.5 text-sky-400" />
                  Only values calculable from pH and temperature are shown.
                  Turbidity, TDS, dissolved oxygen, etc. require dedicated sensors.
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function HistoryChart({ data, dataKey, color, label, unit, refMin, refMax }) {
  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-40 text-slate-400 text-sm">
        No historical data yet.
      </div>
    );
  }
  return (
    <ResponsiveContainer width="100%" height={180}>
      <LineChart data={[...data].reverse()} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis
          dataKey="received_at"
          tickFormatter={formatChartTime}
          tick={{ fontSize: 10, fill: "#94a3b8" }}
          interval="preserveStartEnd"
        />
        <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} />
        <Tooltip
          formatter={(v) => [`${v} ${unit}`, label]}
          labelFormatter={(l) => formatChartTime(l)}
          contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e2e8f0" }}
        />
        {refMin !== undefined && (
          <ReferenceLine y={refMin} stroke="#f97316" strokeDasharray="4 2" label={{ value: `Min ${refMin}`, fontSize: 10, fill: "#f97316", position: "insideTopLeft" }} />
        )}
        {refMax !== undefined && (
          <ReferenceLine y={refMax} stroke="#ef4444" strokeDasharray="4 2" label={{ value: `Max ${refMax}`, fontSize: 10, fill: "#ef4444", position: "insideTopRight" }} />
        )}
        <Line
          type="monotone"
          dataKey={dataKey}
          stroke={color}
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

function RegisterDeviceForm({ onRegistered }) {
  const [form, setForm] = useState({
    device_id: "ESP32-001",
    api_key: "",
    label: "",
    ph_min: 6.5, ph_max: 8.5,
    temp_min: 20.0, temp_max: 32.0,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handle = async (e) => {
    e.preventDefault();
    setError("");
    if (!form.api_key || form.api_key.length < 8) {
      setError("API key must be at least 8 characters.");
      return;
    }
    setLoading(true);
    try {
      await registerDevice(form);
      toast.success("Device registered! Save your API key — it cannot be recovered.");
      onRegistered(form.device_id);
    } catch (err) {
      const detail = err.response?.data?.detail || "Registration failed.";
      setError(detail);
      toast.error(detail);
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-2xl bg-white border border-sky-100 shadow-sm p-6 max-w-lg"
    >
      <h3 className="font-semibold text-slate-800 text-lg mb-1 flex items-center gap-2">
        <Plus className="w-5 h-5 text-sky-500" /> Register your ESP32 Device
      </h3>
      <p className="text-sm text-slate-500 mb-5">
        Enter a unique device ID and create an API key. The API key is shown
        only once — configure it in your ESP32 firmware.
      </p>

      {error && (
        <div className="mb-4 text-sm text-red-600 bg-red-50 border border-red-200 rounded-xl px-3 py-2 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" /> {error}
        </div>
      )}

      <form onSubmit={handle} className="space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-medium text-slate-500 mb-1">Device ID</label>
            <input
              value={form.device_id}
              onChange={(e) => setForm({ ...form, device_id: e.target.value })}
              placeholder="ESP32-001"
              className="w-full px-3 py-2 rounded-xl border border-slate-200 text-sm focus:outline-none focus:border-sky-400"
              required
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-500 mb-1">Label (optional)</label>
            <input
              value={form.label}
              onChange={(e) => setForm({ ...form, label: e.target.value })}
              placeholder="Pond 1"
              className="w-full px-3 py-2 rounded-xl border border-slate-200 text-sm focus:outline-none focus:border-sky-400"
            />
          </div>
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-500 mb-1">
            API Key <span className="text-red-500">*</span>
          </label>
          <input
            type="password"
            value={form.api_key}
            onChange={(e) => setForm({ ...form, api_key: e.target.value })}
            placeholder="Min 8 characters — save this, it cannot be recovered"
            className="w-full px-3 py-2 rounded-xl border border-slate-200 text-sm focus:outline-none focus:border-sky-400"
            required minLength={8}
          />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-medium text-slate-500 mb-1">pH min</label>
            <input type="number" step="0.1" value={form.ph_min}
              onChange={(e) => setForm({ ...form, ph_min: parseFloat(e.target.value) })}
              className="w-full px-3 py-2 rounded-xl border border-slate-200 text-sm focus:outline-none focus:border-sky-400" />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-500 mb-1">pH max</label>
            <input type="number" step="0.1" value={form.ph_max}
              onChange={(e) => setForm({ ...form, ph_max: parseFloat(e.target.value) })}
              className="w-full px-3 py-2 rounded-xl border border-slate-200 text-sm focus:outline-none focus:border-sky-400" />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-500 mb-1">Temp min (°C)</label>
            <input type="number" step="0.5" value={form.temp_min}
              onChange={(e) => setForm({ ...form, temp_min: parseFloat(e.target.value) })}
              className="w-full px-3 py-2 rounded-xl border border-slate-200 text-sm focus:outline-none focus:border-sky-400" />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-500 mb-1">Temp max (°C)</label>
            <input type="number" step="0.5" value={form.temp_max}
              onChange={(e) => setForm({ ...form, temp_max: parseFloat(e.target.value) })}
              className="w-full px-3 py-2 rounded-xl border border-slate-200 text-sm focus:outline-none focus:border-sky-400" />
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full py-2.5 rounded-xl bg-gradient-to-r from-sky-500 to-blue-600 text-white font-semibold text-sm hover:from-sky-400 hover:to-blue-500 transition-all disabled:opacity-50"
        >
          {loading ? "Registering…" : "Register Device"}
        </button>
      </form>
    </motion.div>
  );
}

// ─── Water Quality Advisor ────────────────────────────────────────────────────

const LEVEL_STYLES = {
  normal:  { bg: "bg-green-50",  border: "border-green-200",  text: "text-green-700",  badge: "bg-green-100 text-green-700"  },
  warning: { bg: "bg-amber-50",  border: "border-amber-200",  text: "text-amber-700",  badge: "bg-amber-100 text-amber-700"  },
  danger:  { bg: "bg-red-50",    border: "border-red-200",    text: "text-red-700",    badge: "bg-red-100 text-red-700"      },
};

function AdvisorSection({ icon: Icon, title, iconColor, children }) {
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-2">
        <Icon className={`w-4 h-4 ${iconColor}`} />
        {title}
      </h3>
      {children}
    </div>
  );
}

function AdvisorBlock({ label, level, items, accent }) {
  const colors = {
    slate:  "bg-slate-50 border-slate-100 text-slate-500 bullet-slate-300",
    ocean:  "bg-sky-50 border-sky-100 text-sky-600",
    orange: "bg-orange-50 border-orange-100 text-orange-600",
    red:    "bg-red-50 border-red-100 text-red-600",
  };
  const style = LEVEL_STYLES[level] || LEVEL_STYLES.warning;

  if (!items || items.length === 0) return null;
  return (
    <div className={`rounded-xl border p-4 space-y-2 ${
      accent === "slate"  ? "bg-slate-50 border-slate-100" :
      accent === "ocean"  ? "bg-sky-50 border-sky-100" :
      accent === "orange" ? "bg-orange-50 border-orange-100" :
                            "bg-red-50 border-red-100"
    }`}>
      <div className="flex items-center gap-2">
        <span className={`text-xs font-bold px-2.5 py-1 rounded-full ${style.badge}`}>
          {label}
        </span>
      </div>
      <ul className="space-y-1.5">
        {items.map((item, i) => (
          <li key={i} className="text-sm text-slate-600 flex items-start gap-2">
            <span className={`mt-1 w-1.5 h-1.5 rounded-full shrink-0 ${
              accent === "ocean"  ? "bg-sky-400"   :
              accent === "orange" ? "bg-orange-400" :
              accent === "red"    ? "bg-red-400"   :
                                    "bg-slate-300"
            }`} />
            {item}
          </li>
        ))}
      </ul>
    </div>
  );
}

function WaterQualityAdvisor({ ph, temp }) {
  const advice = getWaterQualityAdvice(ph, temp);
  const overallStyle = LEVEL_STYLES[advice.overallLevel];
  const OverallIcon = advice.overallLevel === "danger"
    ? ShieldAlert
    : advice.overallLevel === "warning"
    ? AlertTriangle
    : ShieldCheck;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-2xl bg-white border border-sky-100 shadow-sm p-6 space-y-6"
    >
      {/* ── Overall banner ───────────────────────────────────────────── */}
      <div className={`rounded-xl px-4 py-3 flex items-center gap-3 border ${overallStyle.bg} ${overallStyle.border}`}>
        <OverallIcon className={`w-5 h-5 shrink-0 ${overallStyle.text}`} />
        <span className={`text-sm font-semibold ${overallStyle.text}`}>
          {advice.overallLevel === "normal"
            ? "Water conditions are optimal — your fish are safe and healthy."
            : advice.overallLevel === "warning"
            ? "Water conditions need attention — your fish are under stress."
            : "Critical water conditions — take immediate action to prevent fish loss!"}
        </span>
        <span className={`ml-auto text-xs font-bold px-2.5 py-1 rounded-full ${overallStyle.badge}`}>
          {advice.overallLevel.toUpperCase()}
        </span>
      </div>

      {/* ── pH section ───────────────────────────────────────────────── */}
      {advice.ph && advice.ph.level !== "normal" && (
        <div className="space-y-4 pb-4 border-b border-slate-100">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold text-slate-800 flex items-center gap-2">
              <Droplets className="w-4 h-4 text-sky-500" />
              pH Analysis — <span className={`text-sm ${LEVEL_STYLES[advice.ph.level].text}`}>{advice.ph.label} (pH {ph?.toFixed(2)})</span>
            </h2>
          </div>

          {/* Cause */}
          <AdvisorSection icon={Info} title="What causes this pH level?" iconColor="text-slate-400">
            <AdvisorBlock
              label="Common Causes"
              level={advice.ph.level}
              accent="slate"
              items={
                ph < 6.5
                  ? [
                      "Decomposition of organic matter (dead fish, uneaten feed, leaves) produces CO₂ and acids.",
                      "Acid rain or naturally acidic water source feeding the pond.",
                      "Overstocking causes excess metabolic waste and CO₂ buildup.",
                      "Heavy rainfall diluting alkalinity buffers in the water.",
                      "Lack of aquatic plants — photosynthesis normally consumes CO₂ and raises pH.",
                    ]
                  : [
                      "Algae bloom (phytoplankton) — photosynthesis consumes CO₂ and raises pH sharply.",
                      "Excessive fertiliser or manure input fuelling algae growth.",
                      "Low water exchange allowing alkalinity to accumulate.",
                      "Hot sunny weather accelerating photosynthesis rates.",
                      "Carbonate-rich soil or water source.",
                    ]
              }
            />
          </AdvisorSection>

          {/* Fish impact */}
          <AdvisorSection icon={Fish} title="What is happening to your fish right now?" iconColor="text-orange-400">
            <AdvisorBlock
              label="Fish Impact"
              level={advice.ph.level}
              accent={advice.ph.level === "danger" ? "red" : "orange"}
              items={advice.ph.fishImpact}
            />
          </AdvisorSection>

          {/* Solutions */}
          <AdvisorSection icon={Lightbulb} title="How to fix it — step by step" iconColor="text-amber-400">
            <AdvisorBlock
              label="Recommended Actions"
              level={advice.ph.level}
              accent="ocean"
              items={advice.ph.recommendations}
            />
          </AdvisorSection>
        </div>
      )}

      {advice.ph && advice.ph.level === "normal" && (
        <div className="flex items-center gap-3 text-sm text-green-700 bg-green-50 border border-green-100 rounded-xl px-4 py-3">
          <CheckCircle2 className="w-4 h-4 shrink-0" />
          <span><strong>pH {ph?.toFixed(2)}</strong> — Optimal. No pH adjustment needed.</span>
        </div>
      )}

      {/* ── Temperature section ──────────────────────────────────────── */}
      {advice.temp && advice.temp.level !== "normal" && (
        <div className="space-y-4 pb-4 border-b border-slate-100">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold text-slate-800 flex items-center gap-2">
              <Thermometer className="w-4 h-4 text-orange-400" />
              Temperature Analysis — <span className={`text-sm ${LEVEL_STYLES[advice.temp.level].text}`}>{advice.temp.label} ({temp?.toFixed(1)}°C)</span>
            </h2>
          </div>

          {/* Cause */}
          <AdvisorSection icon={Info} title="What causes this temperature?" iconColor="text-slate-400">
            <AdvisorBlock
              label="Common Causes"
              level={advice.temp.level}
              accent="slate"
              items={
                temp < 22
                  ? [
                      "Cold weather or nighttime temperature drops in shallow ponds.",
                      "Seasonal change — winter or early spring conditions.",
                      "Groundwater inflow from cold underground sources.",
                      "Shading reducing solar warming of the pond.",
                      "High altitude location with naturally cooler temperatures.",
                    ]
                  : [
                      "Prolonged direct sunlight on shallow ponds during hot season.",
                      "Insufficient pond depth — shallow water heats up rapidly.",
                      "No shading or cover over the pond surface.",
                      "High ambient air temperature during summer months.",
                      "Low water exchange rate — heat accumulates over time.",
                    ]
              }
            />
          </AdvisorSection>

          {/* Fish impact */}
          <AdvisorSection icon={Fish} title="What is happening to your fish right now?" iconColor="text-orange-400">
            <AdvisorBlock
              label="Fish Impact"
              level={advice.temp.level}
              accent={advice.temp.level === "danger" ? "red" : "orange"}
              items={advice.temp.fishImpact}
            />
          </AdvisorSection>

          {/* Solutions */}
          <AdvisorSection icon={Lightbulb} title="How to fix it — step by step" iconColor="text-amber-400">
            <AdvisorBlock
              label="Recommended Actions"
              level={advice.temp.level}
              accent="ocean"
              items={advice.temp.recommendations}
            />
          </AdvisorSection>
        </div>
      )}

      {advice.temp && advice.temp.level === "normal" && (
        <div className="flex items-center gap-3 text-sm text-green-700 bg-green-50 border border-green-100 rounded-xl px-4 py-3">
          <CheckCircle2 className="w-4 h-4 shrink-0" />
          <span><strong>{temp?.toFixed(1)}°C</strong> — Optimal temperature. No action needed.</span>
        </div>
      )}

      {/* ── Disease Risks ─────────────────────────────────────────────── */}
      {advice.diseaseRisks.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-2">
            <Bug className="w-4 h-4 text-red-400" />
            Diseases Likely to Appear in These Conditions
          </h3>
          <div className="flex flex-wrap gap-2">
            {advice.diseaseRisks.map((disease, i) => (
              <span
                key={i}
                className="inline-flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-full bg-red-50 text-red-700 border border-red-100"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-red-400 inline-block" />
                {disease}
              </span>
            ))}
          </div>
          <p className="text-xs text-slate-400 bg-slate-50 rounded-xl px-3 py-2 border border-slate-100">
            ⚠️ These diseases are more likely to develop if current conditions persist.
            Inspect your fish daily for early symptoms and use the{" "}
            <strong className="text-slate-600">Detect Disease</strong> tool if you notice
            unusual behavior, lesions, or discoloration.
          </p>
        </div>
      )}

      {/* ── All clear ─────────────────────────────────────────────────── */}
      {advice.overallLevel === "normal" && (
        <div className="text-center py-4">
          <CheckCircle2 className="w-10 h-10 text-green-400 mx-auto mb-2" />
          <p className="text-sm font-semibold text-green-700">All conditions optimal</p>
          <p className="text-xs text-slate-400 mt-1">
            pH and temperature are within ideal ranges. Continue regular monitoring.
          </p>
        </div>
      )}
    </motion.div>
  );
}


function TestModePanel({ deviceId, onInjected }) {
  const [busy, setBusy] = useState("");
  const [customPh, setCustomPh] = useState("");
  const [customTemp, setCustomTemp] = useState("");

  const injectSingle = async () => {
    setBusy("single");
    try {
      const params = {};
      if (customPh)   params.ph         = parseFloat(customPh);
      if (customTemp) params.temperature = parseFloat(customTemp);
      await injectMockReading(deviceId, params);
      toast.success("Mock reading injected!");
      onInjected();
    } catch (err) {
      toast.error(err.response?.data?.detail || "Injection failed.");
    } finally {
      setBusy("");
    }
  };

  const injectBurst = async () => {
    setBusy("burst");
    try {
      const res = await injectMockBurst(deviceId, 20);
      toast.success(res.data.message);
      onInjected();
    } catch (err) {
      toast.error(err.response?.data?.detail || "Injection failed.");
    } finally {
      setBusy("");
    }
  };

  const clearAll = async () => {
    setBusy("clear");
    try {
      const res = await clearMockReadings(deviceId);
      toast.success(res.data.message);
      onInjected();
    } catch (err) {
      toast.error(err.response?.data?.detail || "Clear failed.");
    } finally {
      setBusy("");
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-2xl bg-white border border-dashed border-sky-200 shadow-sm p-5"
    >
      <div className="flex items-center gap-2 mb-1">
        <FlaskConical className="w-4 h-4 text-sky-500" />
        <h3 className="font-semibold text-slate-800 text-sm">Test Mode — Simulate Sensor Data</h3>
      </div>
      <p className="text-xs text-slate-400 mb-4">
        Inject mock readings to test the dashboard without an ESP32 connected.
      </p>

      <div className="grid grid-cols-2 gap-3 mb-4">
        <div>
          <label className="block text-xs font-medium text-slate-500 mb-1">pH (optional)</label>
          <input
            type="number" step="0.1" value={customPh}
            onChange={(e) => setCustomPh(e.target.value)}
            placeholder="Random"
            className="w-full px-3 py-1.5 rounded-lg border border-slate-200 text-sm focus:outline-none focus:border-sky-400"
          />
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-500 mb-1">Temp °C (optional)</label>
          <input
            type="number" step="0.5" value={customTemp}
            onChange={(e) => setCustomTemp(e.target.value)}
            placeholder="Random"
            className="w-full px-3 py-1.5 rounded-lg border border-slate-200 text-sm focus:outline-none focus:border-sky-400"
          />
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        <button
          onClick={injectSingle}
          disabled={!!busy}
          className="px-3 py-1.5 rounded-lg bg-sky-500 text-white text-xs font-semibold hover:bg-sky-400 disabled:opacity-50 transition-colors"
        >
          {busy === "single" ? "Injecting…" : "+ 1 Reading"}
        </button>
        <button
          onClick={injectBurst}
          disabled={!!busy}
          className="px-3 py-1.5 rounded-lg bg-ocean-500 text-white text-xs font-semibold hover:bg-ocean-400 disabled:opacity-50 transition-colors"
        >
          {busy === "burst" ? "Injecting…" : "+ 20 Readings"}
        </button>
        <button
          onClick={clearAll}
          disabled={!!busy}
          className="px-3 py-1.5 rounded-lg border border-red-200 text-red-600 text-xs font-semibold hover:bg-red-50 disabled:opacity-50 transition-colors"
        >
          {busy === "clear" ? "Clearing…" : "Clear All"}
        </button>
      </div>
    </motion.div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────
export default function WaterQuality() {
  const { user } = useAuth();

  // ── pond list (multi-pond) ──────────────────────────────────────────────
  const [ponds, setPonds]           = useState(loadPonds);
  const [deviceId, setDeviceId]     = useState(
    () => localStorage.getItem(ACTIVE_KEY) || ""
  );
  const [deviceInput, setDeviceInput] = useState(deviceId);
  const [editingLabel, setEditingLabel] = useState(false);
  const [labelInput, setLabelInput] = useState("");

  // ── data ────────────────────────────────────────────────────────────────
  const [status, setStatus]         = useState(null);
  const [history, setHistory]       = useState([]);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState("");
  const [showRegister, setShowRegister] = useState(false);
  const [exporting, setExporting]   = useState(false);
  const pollRef = useRef(null);

  const fetchData = useCallback(async (id) => {
    if (!id) return;
    setLoading(true);
    setError("");
    try {
      const [statusRes, histRes] = await Promise.all([
        getDeviceStatus(id),
        getIoTHistory(id, { page: 1, page_size: CHART_POINTS }),
      ]);
      setStatus(statusRes.data);
      setHistory(histRes.data.items || []);
    } catch (err) {
      if (err.response?.status === 404) {
        setError(`Device "${id}" not found. Register it first.`);
      } else {
        setError(err.response?.data?.detail || "Failed to fetch device data.");
      }
      setStatus(null);
      setHistory([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!deviceId) return;
    fetchData(deviceId);
    pollRef.current = setInterval(() => fetchData(deviceId), POLL_INTERVAL_MS);
    return () => clearInterval(pollRef.current);
  }, [deviceId, fetchData]);

  // ── pond management ────────────────────────────────────────────────────
  const switchPond = (id) => {
    localStorage.setItem(ACTIVE_KEY, id);
    setDeviceId(id);
    setDeviceInput(id);
    setStatus(null);
    setHistory([]);
  };

  const removePond = (id) => {
    const updated = ponds.filter((p) => p.id !== id);
    savePonds(updated);
    setPonds(updated);
    if (deviceId === id) {
      const next = updated[0]?.id || "";
      switchPond(next);
    }
  };

  const applyDevice = () => {
    const id = deviceInput.trim();
    if (!id) return;
    // Add to pond list if not already present
    if (!ponds.find((p) => p.id === id)) {
      const updated = [...ponds, { id, label: id }];
      savePonds(updated);
      setPonds(updated);
    }
    switchPond(id);
  };

  const handleRegistered = (id) => {
    const updated = ponds.find((p) => p.id === id)
      ? ponds
      : [...ponds, { id, label: id }];
    savePonds(updated);
    setPonds(updated);
    switchPond(id);
    setShowRegister(false);
    fetchData(id);
  };

  const saveLabel = () => {
    const updated = ponds.map((p) =>
      p.id === deviceId ? { ...p, label: labelInput || p.id } : p
    );
    savePonds(updated);
    setPonds(updated);
    setEditingLabel(false);
  };

  const handleExportPDF = async () => {
    setExporting(true);
    try {
      await exportWaterQualityReport(status, history, user);
    } finally {
      setExporting(false);
    }
  };

  const r           = status?.latest;
  const activePond  = ponds.find((p) => p.id === deviceId);

  return (
    <div className="space-y-6 max-w-5xl mx-auto">

      {/* ── Header ── */}
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-800 flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-sky-500 to-blue-600 flex items-center justify-center shadow-lg shadow-sky-500/25">
                <Droplets className="w-5 h-5 text-white" />
              </div>
              Water Quality Monitor
            </h1>
            <p className="text-slate-500 text-sm mt-1">
              Live pH and temperature readings. Refreshes every 30 s.
            </p>
          </div>
          <div className="flex items-center gap-2">
            {status && (
              <DeviceStatusBadge online={status.online} lastSeen={status.last_seen} />
            )}
            {status && (
              <button
                onClick={handleExportPDF}
                disabled={exporting}
                className="inline-flex items-center gap-2 px-3 py-2 rounded-xl border border-sky-200 bg-white text-sky-700 text-sm font-medium hover:bg-sky-50 disabled:opacity-40 transition-colors shadow-sm"
              >
                <Download className="w-4 h-4" />
                {exporting ? "…" : "PDF"}
              </button>
            )}
          </div>
        </div>
      </motion.div>

      {/* ── Multi-pond selector ── */}
      {ponds.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {ponds.map((pond) => (
            <div key={pond.id} className="flex items-center gap-1">
              <button
                onClick={() => switchPond(pond.id)}
                className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-sm font-medium transition-all ${
                  pond.id === deviceId
                    ? "bg-sky-500 text-white shadow-sm"
                    : "bg-white border border-slate-200 text-slate-600 hover:bg-sky-50"
                }`}
              >
                <MapPin className="w-3 h-3" />
                {pond.label || pond.id}
                {pond.id === deviceId && status?.online && (
                  <span className="w-1.5 h-1.5 rounded-full bg-green-300 inline-block" />
                )}
              </button>
              {pond.id === deviceId && (
                <button
                  onClick={() => removePond(pond.id)}
                  className="p-1 rounded-lg text-slate-300 hover:text-red-400 hover:bg-red-50 transition-colors"
                  title="Remove pond"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      {/* ── Active pond label edit ── */}
      {activePond && (
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <MapPin className="w-3.5 h-3.5 text-sky-400" />
          {editingLabel ? (
            <>
              <input
                value={labelInput}
                onChange={(e) => setLabelInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && saveLabel()}
                className="px-2 py-0.5 rounded-lg border border-sky-300 text-sm focus:outline-none focus:border-sky-500 w-32"
                autoFocus
              />
              <button onClick={saveLabel} className="text-sky-600 hover:text-sky-700">
                <Check className="w-4 h-4" />
              </button>
            </>
          ) : (
            <>
              <span className="font-medium text-slate-700">{activePond.label || activePond.id}</span>
              <button
                onClick={() => { setLabelInput(activePond.label || activePond.id); setEditingLabel(true); }}
                className="text-slate-300 hover:text-sky-500 transition-colors"
                title="Rename pond"
              >
                <Edit2 className="w-3 h-3" />
              </button>
            </>
          )}
        </div>
      )}

      {/* ── Device input + register ── */}
      <div className="flex flex-wrap gap-3 items-center">
        <div className="flex gap-2 flex-1 min-w-0">
          <input
            value={deviceInput}
            onChange={(e) => setDeviceInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && applyDevice()}
            placeholder="Device ID — e.g. ESP32-001"
            className="flex-1 px-4 py-2.5 rounded-xl border border-slate-200 text-sm focus:outline-none focus:border-sky-400 bg-white shadow-sm"
          />
          <button onClick={applyDevice} className="px-5 py-2.5 rounded-xl bg-sky-500 text-white font-semibold text-sm hover:bg-sky-400 transition-colors shadow-sm">
            Load
          </button>
          <button
            onClick={() => deviceId && fetchData(deviceId)}
            title="Refresh now"
            className="p-2.5 rounded-xl border border-slate-200 bg-white text-slate-500 hover:bg-sky-50 transition-colors shadow-sm"
          >
            <RefreshCcw className="w-4 h-4" />
          </button>
        </div>
        <button
          onClick={() => setShowRegister((s) => !s)}
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl border border-sky-200 bg-sky-50 text-sky-700 text-sm font-medium hover:bg-sky-100 transition-colors"
        >
          <Plus className="w-4 h-4" /> Register Device
        </button>
      </div>

      <AnimatePresence>
        {showRegister && <RegisterDeviceForm onRegistered={handleRegistered} />}
      </AnimatePresence>

      {/* ── Test Mode (mock sensor data) ─────────────────────────────────── */}
      {deviceId && (
        <TestModePanel deviceId={deviceId} onInjected={() => fetchData(deviceId)} />
      )}

      {error && (
        <div className="rounded-xl bg-red-50 border border-red-200 text-red-600 text-sm px-4 py-3 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" /> {error}
        </div>
      )}

      {loading && !status && (
        <div className="grid sm:grid-cols-2 gap-5">
          {[0, 1].map((i) => (
            <div key={i} className="rounded-2xl bg-slate-100 animate-pulse h-36" />
          ))}
        </div>
      )}

      {r && (
        <>
          <AlertBanner alerts={r.alerts} />
          <div className="grid sm:grid-cols-2 gap-5">
            <GaugeCard
              icon={Droplets}
              title="pH Level"
              value={r.ph?.toFixed(2)}
              unit=""
              status={r.ph_status}
              color={phColor(r.ph_status)}
              secondary={[
                `Normal range: ${status.ph_min} – ${status.ph_max}`,
                `[H⁺] = ${r.h_ion_conc?.toExponential(3)} mol/L`,
                `pOH ≈ ${r.poh?.toFixed(2)}`,
              ]}
            />
            <GaugeCard
              icon={Thermometer}
              title="Water Temperature"
              value={r.temperature?.toFixed(1)}
              unit="°C"
              status={r.temp_status}
              color={tempColor(r.temp_status)}
              secondary={[
                `Normal range: ${status.temp_min} – ${status.temp_max} °C`,
                r.device_ts ? `Device time: ${formatDateTime(r.device_ts)}` : null,
              ].filter(Boolean)}
            />
          </div>
          <DerivedInfo reading={r} />

          {/* ── Water Quality Advisor ─────────────────────────────── */}
          <WaterQualityAdvisor ph={r.ph} temp={r.temperature} />
        </>
      )}

      {history.length > 0 && (
        <div className="grid sm:grid-cols-2 gap-5">
          <div className="rounded-2xl bg-white border border-sky-100 shadow-sm p-5">
            <div className="flex items-center gap-2 text-slate-700 font-semibold text-sm mb-4">
              <Activity className="w-4 h-4 text-sky-500" />
              pH History (last {history.length} readings)
            </div>
            <HistoryChart data={history} dataKey="ph" color="#0ea5e9" label="pH" unit="" refMin={status?.ph_min} refMax={status?.ph_max} />
          </div>
          <div className="rounded-2xl bg-white border border-sky-100 shadow-sm p-5">
            <div className="flex items-center gap-2 text-slate-700 font-semibold text-sm mb-4">
              <Thermometer className="w-4 h-4 text-orange-400" />
              Temperature History (last {history.length} readings)
            </div>
            <HistoryChart data={history} dataKey="temperature" color="#f97316" label="Temp" unit="°C" refMin={status?.temp_min} refMax={status?.temp_max} />
          </div>
        </div>
      )}

      {!loading && !error && !status && !showRegister && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="rounded-2xl bg-white border border-sky-100 shadow-sm p-10 text-center"
        >
          <Droplets className="w-12 h-12 text-sky-200 mx-auto mb-3" />
          <p className="text-slate-500 text-sm mb-4">
            Enter your ESP32 device ID above to view live water-quality data,
            or register a new device.
          </p>
          <button
            onClick={() => setShowRegister(true)}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-sky-500 text-white font-semibold text-sm hover:bg-sky-400 transition-colors"
          >
            <Plus className="w-4 h-4" /> Register First Device
          </button>
        </motion.div>
      )}

      {status && (
        <div className="rounded-2xl bg-white border border-sky-100 shadow-sm p-4 flex flex-wrap gap-4 text-xs text-slate-400">
          <span><strong className="text-slate-600">Device:</strong> {status.device_id}</span>
          {status.label && <span><strong className="text-slate-600">Label:</strong> {status.label}</span>}
          <span>
            <strong className="text-slate-600">Status:</strong>{" "}
            <span className={status.online ? "text-green-600 font-semibold" : "text-slate-400"}>
              {status.online ? "Online" : "Offline"}
            </span>
          </span>
          <span><strong className="text-slate-600">Ponds tracked:</strong> {ponds.length}</span>
          <span><strong className="text-slate-600">Poll:</strong> 30 s</span>
          <span className="ml-auto text-slate-300">Last fetch: {new Date().toLocaleTimeString()}</span>
        </div>
      )}
    </div>
  );
}
