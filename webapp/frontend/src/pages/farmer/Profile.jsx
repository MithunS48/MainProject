import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { User, Mail, Calendar, Fish, Edit3, Save, X, ShieldCheck } from "lucide-react";
import toast from "react-hot-toast";
import { useAuth } from "../../context/AuthContext";
import { updateMyProfile, getMyPredictions } from "../../api/endpoints";
import { getDiseaseInfo } from "../../utils/diseaseInfo";
import { formatDate, formatDateTime } from "../../utils/format";

export default function Profile() {
  const { user, refreshUser } = useAuth();
  const [editing, setEditing] = useState(false);
  const [fullName, setFullName] = useState(user?.full_name || "");
  const [email, setEmail] = useState(user?.email || "");
  const [saving, setSaving] = useState(false);
  const [recent, setRecent] = useState([]);

  useEffect(() => {
    setFullName(user?.full_name || "");
    setEmail(user?.email || "");
  }, [user]);

  useEffect(() => {
    (async () => {
      try {
        const res = await getMyPredictions({ page: 1, page_size: 5, sort: "date_desc" });
        setRecent(res.data.items || []);
      } catch {
        setRecent([]);
      }
    })();
  }, []);

  const handleSave = async () => {
    setSaving(true);
    try {
      await updateMyProfile({ full_name: fullName, email });
      await refreshUser();
      toast.success("Profile updated successfully");
      setEditing(false);
    } catch (err) {
      toast.error(err.response?.data?.detail || "Failed to update profile");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="max-w-3xl space-y-6">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="font-display text-2xl sm:text-3xl font-bold text-slate-900">My Profile</h1>
        <p className="text-slate-500 mt-1">View and manage your account information.</p>
      </motion.div>

      <div className="glass-card rounded-3xl p-6 sm:p-8">
        <div className="flex items-center gap-4 mb-6">
          <div className="w-16 h-16 rounded-full bg-gradient-to-br from-ocean-400 to-seaweed-500 flex items-center justify-center text-xl font-bold text-white shrink-0">
            {user?.full_name?.[0]?.toUpperCase() || "U"}
          </div>
          <div className="flex-1">
            <div className="font-display text-lg font-bold text-slate-900">{user?.full_name}</div>
            <div className="text-sm text-slate-500">{user?.email}</div>
          </div>
          {!editing && (
            <button onClick={() => setEditing(true)} className="btn-secondary !py-2 !px-4 text-sm">
              <Edit3 className="w-3.5 h-3.5" /> Edit
            </button>
          )}
        </div>

        {editing ? (
          <div className="space-y-4 border-t border-slate-100 pt-6">
            <div>
              <label className="text-sm font-medium text-slate-600 mb-1.5 block">Full Name</label>
              <div className="relative">
                <User className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                <input value={fullName} onChange={(e) => setFullName(e.target.value)} className="input-field pl-11" />
              </div>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-600 mb-1.5 block">Email</label>
              <div className="relative">
                <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                <input value={email} onChange={(e) => setEmail(e.target.value)} className="input-field pl-11" />
              </div>
            </div>
            <div className="flex gap-3">
              <button onClick={handleSave} disabled={saving} className="btn-primary text-sm !py-2.5">
                <Save className="w-4 h-4" /> {saving ? "Saving..." : "Save Changes"}
              </button>
              <button
                onClick={() => {
                  setEditing(false);
                  setFullName(user?.full_name || "");
                  setEmail(user?.email || "");
                }}
                className="btn-secondary text-sm !py-2.5"
              >
                <X className="w-4 h-4" /> Cancel
              </button>
            </div>
          </div>
        ) : (
          <div className="grid sm:grid-cols-2 gap-4 border-t border-slate-100 pt-6">
            <InfoTile icon={Calendar} label="Member Since" value={formatDate(user?.created_at)} />
            <InfoTile icon={Fish} label="Total Analyses" value={user?.analyses_count ?? 0} />
            <InfoTile icon={ShieldCheck} label="Account Role" value={user?.role === "admin" ? "Administrator" : "Farmer / User"} />
            <InfoTile icon={User} label="Account Status" value={user?.is_active ? "Active" : "Inactive"} />
          </div>
        )}
      </div>

      <div className="glass-card rounded-2xl p-6">
        <h3 className="font-display font-semibold text-slate-900 mb-4">Recent Predictions</h3>
        {recent.length === 0 ? (
          <p className="text-slate-400 text-sm text-center py-6">No predictions yet.</p>
        ) : (
          <div className="space-y-2">
            {recent.map((p) => {
              const info = getDiseaseInfo(p.predicted_class);
              return (
                <div key={p.id} className="flex items-center gap-3 px-2 py-2.5 rounded-xl hover:bg-slate-50">
                  <img src={p.image_url} alt="" className="w-10 h-10 rounded-lg object-cover bg-slate-100 shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm text-slate-900 font-medium truncate">{info.name}</div>
                    <div className="text-xs text-slate-400">{formatDateTime(p.created_at)}</div>
                  </div>
                  <div className={`text-sm font-semibold ${info.isHealthy ? "text-seaweed-600" : "text-coral-600"}`}>
                    {p.confidence_pct}%
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

function InfoTile({ icon: Icon, label, value }) {
  return (
    <div className="flex items-center gap-3 bg-slate-50 rounded-xl px-4 py-3">
      <Icon className="w-4 h-4 text-ocean-500 shrink-0" />
      <div>
        <div className="text-xs text-slate-400">{label}</div>
        <div className="text-sm font-semibold text-slate-900">{value}</div>
      </div>
    </div>
  );
}
