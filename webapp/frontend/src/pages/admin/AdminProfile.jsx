import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { User, Mail, Calendar, ShieldCheck, Edit3, Save, X } from "lucide-react";
import toast from "react-hot-toast";
import { useAuth } from "../../context/AuthContext";
import { updateMyProfile } from "../../api/endpoints";
import { formatDate } from "../../utils/format";

export default function AdminProfile() {
  const { user, refreshUser } = useAuth();
  const [editing, setEditing] = useState(false);
  const [fullName, setFullName] = useState(user?.full_name || "");
  const [email, setEmail] = useState(user?.email || "");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setFullName(user?.full_name || "");
    setEmail(user?.email || "");
  }, [user]);

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
    <div className="max-w-2xl space-y-6">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="font-display text-2xl sm:text-3xl font-bold text-slate-900">My Profile</h1>
        <p className="text-slate-500 mt-1">Manage your administrator account information.</p>
      </motion.div>

      <div className="glass-card rounded-3xl p-6 sm:p-8">
        <div className="flex items-center gap-4 mb-6">
          <div className="w-16 h-16 rounded-full bg-gradient-to-br from-coral-400 to-coral-600 flex items-center justify-center text-xl font-bold text-white shrink-0">
            {user?.full_name?.[0]?.toUpperCase() || "A"}
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
            <InfoTile icon={ShieldCheck} label="Account Role" value="Administrator" />
          </div>
        )}
      </div>
    </div>
  );
}

function InfoTile({ icon: Icon, label, value }) {
  return (
    <div className="flex items-center gap-3 bg-slate-50 rounded-xl px-4 py-3">
      <Icon className="w-4 h-4 text-coral-500 shrink-0" />
      <div>
        <div className="text-xs text-slate-400">{label}</div>
        <div className="text-sm font-semibold text-slate-900">{value}</div>
      </div>
    </div>
  );
}
