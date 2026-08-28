import { useEffect, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search, Users, ShieldCheck, Sprout, ToggleLeft, ToggleRight, Trash2,
  ChevronLeft, ChevronRight, AlertTriangle, X,
} from "lucide-react";
import toast from "react-hot-toast";
import { getAdminUsers, updateAdminUser, deleteAdminUser } from "../../api/endpoints";
import { useAuth } from "../../context/AuthContext";
import { formatDate } from "../../utils/format";

const PAGE_SIZE = 15;

export default function UserManagement() {
  const { user: me } = useAuth();
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [role, setRole] = useState("all");
  const [loading, setLoading] = useState(true);
  const [confirmDelete, setConfirmDelete] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await getAdminUsers({
        page,
        page_size: PAGE_SIZE,
        search: search || undefined,
        role: role !== "all" ? role : undefined,
      });
      setItems(res.data.items || []);
      setTotal(res.data.total || 0);
    } catch {
      setItems([]);
      setTotal(0);
    } finally {
      setLoading(false);
    }
  }, [page, search, role]);

  useEffect(() => {
    load();
  }, [load]);

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  const handleToggleActive = async (u) => {
    if (u.id === me.id) {
      toast.error("You cannot deactivate your own account.");
      return;
    }
    try {
      await updateAdminUser(u.id, { is_active: !u.is_active });
      toast.success(`User ${u.is_active ? "deactivated" : "activated"}`);
      load();
    } catch (err) {
      toast.error(err.response?.data?.detail || "Failed to update user");
    }
  };

  const handleDelete = async () => {
    if (!confirmDelete) return;
    try {
      await deleteAdminUser(confirmDelete.id);
      toast.success("User deleted");
      setConfirmDelete(null);
      load();
    } catch (err) {
      toast.error(err.response?.data?.detail || "Failed to delete user");
    }
  };

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="font-display text-2xl sm:text-3xl font-bold text-white">User Management</h1>
        <p className="text-white/60 mt-1">View, search, activate/deactivate, and manage all registered accounts.</p>
      </motion.div>

      <div className="glass-card rounded-2xl p-4 flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-white/40" />
          <input
            value={search}
            onChange={(e) => {
              setPage(1);
              setSearch(e.target.value);
            }}
            placeholder="Search by name or email..."
            className="input-field pl-10 !py-2.5"
          />
        </div>
        <select
          value={role}
          onChange={(e) => {
            setPage(1);
            setRole(e.target.value);
          }}
          className="input-field !py-2.5 sm:w-48 cursor-pointer"
        >
          <option value="all">All Roles</option>
          <option value="farmer">Farmer / User</option>
          <option value="admin">Admin</option>
        </select>
      </div>

      <div className="glass-card rounded-2xl overflow-hidden">
        {loading ? (
          <div className="text-center py-16 text-white/40 text-sm">Loading users...</div>
        ) : items.length === 0 ? (
          <div className="text-center py-16">
            <Users className="w-10 h-10 text-white/20 mx-auto mb-3" />
            <p className="text-white/50 text-sm">No users match your filters.</p>
          </div>
        ) : (
          <table className="w-full">
            <thead>
              <tr className="border-b border-white/10 text-left">
                <th className="px-5 py-3 text-xs font-semibold text-white/40 uppercase tracking-wide">Name</th>
                <th className="px-5 py-3 text-xs font-semibold text-white/40 uppercase tracking-wide hidden sm:table-cell">Email</th>
                <th className="px-5 py-3 text-xs font-semibold text-white/40 uppercase tracking-wide">Role</th>
                <th className="px-5 py-3 text-xs font-semibold text-white/40 uppercase tracking-wide hidden md:table-cell">Joined</th>
                <th className="px-5 py-3 text-xs font-semibold text-white/40 uppercase tracking-wide hidden lg:table-cell">Analyses</th>
                <th className="px-5 py-3 text-xs font-semibold text-white/40 uppercase tracking-wide">Status</th>
                <th className="px-5 py-3 text-xs font-semibold text-white/40 uppercase tracking-wide"></th>
              </tr>
            </thead>
            <tbody>
              {items.map((u) => (
                <tr key={u.id} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                  <td className="px-5 py-3">
                    <div className="flex items-center gap-2.5">
                      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-ocean-400 to-seaweed-500 flex items-center justify-center text-xs font-bold text-white shrink-0">
                        {u.full_name?.[0]?.toUpperCase()}
                      </div>
                      <span className="text-sm font-medium text-white">{u.full_name}</span>
                    </div>
                  </td>
                  <td className="px-5 py-3 text-sm text-white/50 hidden sm:table-cell">{u.email}</td>
                  <td className="px-5 py-3">
                    <span
                      className={`inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-1 rounded-full ${
                        u.role === "admin"
                          ? "bg-coral-500/15 text-coral-300"
                          : "bg-ocean-500/15 text-ocean-300"
                      }`}
                    >
                      {u.role === "admin" ? <ShieldCheck className="w-3 h-3" /> : <Sprout className="w-3 h-3" />}
                      {u.role === "admin" ? "Admin" : "Farmer"}
                    </span>
                  </td>
                  <td className="px-5 py-3 text-sm text-white/50 hidden md:table-cell">{formatDate(u.created_at)}</td>
                  <td className="px-5 py-3 text-sm text-white/50 hidden lg:table-cell">{u.analyses_count}</td>
                  <td className="px-5 py-3">
                    <button onClick={() => handleToggleActive(u)} className="flex items-center gap-1.5">
                      {u.is_active ? (
                        <ToggleRight className="w-6 h-6 text-seaweed-400" />
                      ) : (
                        <ToggleLeft className="w-6 h-6 text-white/30" />
                      )}
                      <span className={`text-xs font-medium ${u.is_active ? "text-seaweed-400" : "text-white/40"}`}>
                        {u.is_active ? "Active" : "Inactive"}
                      </span>
                    </button>
                  </td>
                  <td className="px-5 py-3 text-right">
                    <button
                      onClick={() => setConfirmDelete(u)}
                      className="text-coral-400 hover:text-coral-300"
                      disabled={u.id === me?.id}
                      title={u.id === me?.id ? "You cannot delete your own account" : "Delete user"}
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <span className="text-sm text-white/40">
            Page {page} of {totalPages} &bull; {total} users
          </span>
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
        {confirmDelete && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4"
            onClick={() => setConfirmDelete(null)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="glass-card rounded-2xl p-6 max-w-sm w-full"
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl bg-coral-500/20 flex items-center justify-center shrink-0">
                  <AlertTriangle className="w-5 h-5 text-coral-400" />
                </div>
                <h3 className="font-display font-semibold text-white">Delete User?</h3>
              </div>
              <p className="text-sm text-white/60 mb-6">
                This will permanently delete <strong className="text-white">{confirmDelete.full_name}</strong> and
                all of their prediction history. This action cannot be undone.
              </p>
              <div className="flex gap-3">
                <button onClick={handleDelete} className="flex-1 bg-coral-500 hover:bg-coral-600 text-white rounded-xl py-2.5 text-sm font-semibold transition-colors">
                  Delete
                </button>
                <button onClick={() => setConfirmDelete(null)} className="flex-1 btn-secondary !py-2.5 text-sm">
                  Cancel
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
