import { useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  Fish, LayoutDashboard, ScanLine, History, BookOpen, UserCircle,
  Menu, X, LogOut, Users, ClipboardList, BarChart3, FlaskConical,
  AlertTriangle, Settings2, ChevronRight,
} from "lucide-react";
import { useAuth } from "../context/AuthContext";
import toast from "react-hot-toast";

const FARMER_NAV = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/dashboard/detect", label: "Detect Disease", icon: ScanLine },
  { to: "/dashboard/history", label: "Prediction History", icon: History },
  { to: "/dashboard/diseases", label: "Disease Information", icon: BookOpen },
  { to: "/dashboard/profile", label: "Profile", icon: UserCircle },
];

const ADMIN_NAV = [
  { to: "/admin", label: "Admin Dashboard", icon: LayoutDashboard },
  { to: "/admin/users", label: "Users", icon: Users },
  { to: "/admin/predictions", label: "Predictions", icon: ClipboardList },
  { to: "/admin/analytics", label: "Analytics", icon: BarChart3 },
  { to: "/admin/model-performance", label: "Model Performance", icon: Settings2 },
  { to: "/admin/research", label: "Research Results", icon: FlaskConical },
  { to: "/admin/error-analysis", label: "Error Analysis", icon: AlertTriangle },
  { to: "/admin/profile", label: "Profile", icon: UserCircle },
];

export default function DashboardLayout({ children, variant = "farmer" }) {
  const [open, setOpen] = useState(false);
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const nav = variant === "admin" ? ADMIN_NAV : FARMER_NAV;

  const handleLogout = () => {
    logout();
    toast.success("Logged out successfully");
    navigate("/login");
  };

  const SidebarContent = () => (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-2 px-5 py-6">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-ocean-400 to-seaweed-500 flex items-center justify-center shadow-glow-ocean shrink-0">
          <Fish className="w-5 h-5 text-white" />
        </div>
        <div>
          <div className="font-display font-bold text-white leading-tight">AquaScan</div>
          <div className="text-[10px] text-white/40 uppercase tracking-wider font-semibold">
            {variant === "admin" ? "Admin Panel" : "Farmer Portal"}
          </div>
        </div>
      </div>

      <nav className="flex-1 px-3 space-y-1 overflow-y-auto">
        {nav.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/dashboard" || item.to === "/admin"}
            onClick={() => setOpen(false)}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all group ${
                isActive
                  ? "bg-gradient-to-r from-ocean-500/25 to-seaweed-500/15 text-white shadow-[inset_0_0_0_1px_rgba(28,171,242,0.3)]"
                  : "text-white/60 hover:bg-white/5 hover:text-white"
              }`
            }
          >
            <item.icon className="w-4.5 h-4.5 shrink-0" />
            <span className="flex-1">{item.label}</span>
            <ChevronRight className="w-3.5 h-3.5 opacity-0 group-hover:opacity-50 transition-opacity" />
          </NavLink>
        ))}
      </nav>

      <div className="p-3 border-t border-white/10 mt-2">
        <div className="flex items-center gap-3 px-2 py-2 mb-2">
          <div className="w-9 h-9 rounded-full bg-gradient-to-br from-ocean-400 to-seaweed-500 flex items-center justify-center text-sm font-bold text-white shrink-0">
            {user?.full_name?.[0]?.toUpperCase() || "U"}
          </div>
          <div className="min-w-0">
            <div className="text-sm font-semibold text-white truncate">{user?.full_name}</div>
            <div className="text-xs text-white/40 truncate capitalize">{user?.role}</div>
          </div>
        </div>
        <button
          onClick={handleLogout}
          className="flex items-center gap-2 w-full px-3.5 py-2.5 rounded-xl text-sm font-medium text-coral-300 hover:bg-coral-500/10 transition-colors"
        >
          <LogOut className="w-4 h-4" /> Log Out
        </button>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-ocean-950 flex">
      {/* Desktop sidebar */}
      <aside className="hidden lg:flex w-72 shrink-0 flex-col bg-white/[0.03] border-r border-white/10 fixed inset-y-0">
        <SidebarContent />
      </aside>

      {/* Mobile sidebar */}
      <AnimatePresence>
        {open && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/60 z-40 lg:hidden"
              onClick={() => setOpen(false)}
            />
            <motion.aside
              initial={{ x: -300 }}
              animate={{ x: 0 }}
              exit={{ x: -300 }}
              transition={{ type: "tween", duration: 0.25 }}
              className="fixed inset-y-0 left-0 w-72 bg-ocean-950 border-r border-white/10 z-50 lg:hidden"
            >
              <SidebarContent />
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      {/* Main content */}
      <div className="flex-1 lg:ml-72 min-w-0">
        <header className="lg:hidden sticky top-0 z-30 bg-ocean-950/90 backdrop-blur-lg border-b border-white/10 px-4 py-3 flex items-center justify-between">
          <button onClick={() => setOpen(true)} className="text-white p-2">
            <Menu className="w-6 h-6" />
          </button>
          <div className="flex items-center gap-2">
            <Fish className="w-5 h-5 text-ocean-400" />
            <span className="font-display font-bold text-white">AquaScan</span>
          </div>
          <div className="w-9" />
        </header>
        <main className="p-4 sm:p-6 lg:p-8 max-w-[1400px] mx-auto">{children}</main>
      </div>
    </div>
  );
}
