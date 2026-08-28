import { useState, useMemo } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  Fish, Mail, Lock, User, Eye, EyeOff, ArrowRight, UserPlus,
  Sprout, ShieldCheck, CheckCircle2, XCircle, KeyRound,
} from "lucide-react";
import toast from "react-hot-toast";
import { useAuth } from "../../context/AuthContext";
import OceanBackground from "../../components/OceanBackground";

function PasswordRule({ met, label }) {
  return (
    <div className={`flex items-center gap-1.5 text-xs ${met ? "text-seaweed-600" : "text-slate-400"}`}>
      {met ? <CheckCircle2 className="w-3.5 h-3.5" /> : <XCircle className="w-3.5 h-3.5" />}
      {label}
    </div>
  );
}

export default function RegisterPage() {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [role, setRole] = useState("farmer");
  const [inviteCode, setInviteCode] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);

  const { register } = useAuth();
  const navigate = useNavigate();

  const rules = useMemo(
    () => ({
      length: password.length >= 8,
      upper: /[A-Z]/.test(password),
      number: /[0-9]/.test(password),
      match: password.length > 0 && password === confirmPassword,
    }),
    [password, confirmPassword]
  );

  const allValid = rules.length && rules.upper && rules.number && rules.match && fullName && email;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!allValid) {
      setError("Please meet all password requirements and ensure passwords match.");
      return;
    }

    setLoading(true);
    try {
      await register({
        full_name: fullName,
        email,
        password,
        confirm_password: confirmPassword,
        role,
        admin_invite_code: role === "admin" ? inviteCode : undefined,
      });
      setSuccess(true);
      toast.success("Account created successfully!");
      setTimeout(() => navigate(role === "admin" ? "/admin" : "/dashboard"), 1200);
    } catch (err) {
      setError(err.response?.data?.detail || "Registration failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  if (success) {
    return (
      <div className="min-h-screen relative flex items-center justify-center px-5 bg-white">
        <OceanBackground fishCount={3} bubbleCount={14} />
        <motion.div
          initial={{ scale: 0.7, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: "spring", stiffness: 200, damping: 15 }}
          className="glass-card rounded-3xl p-10 text-center max-w-sm relative z-10"
        >
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: "spring", stiffness: 300 }}
            className="w-20 h-20 rounded-full bg-seaweed-50 flex items-center justify-center mx-auto mb-6"
          >
            <CheckCircle2 className="w-12 h-12 text-seaweed-500" />
          </motion.div>
          <h2 className="font-display text-2xl font-bold text-slate-900 mb-2">Account Created!</h2>
          <p className="text-slate-500 text-sm">Redirecting you to your dashboard...</p>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen relative flex items-center justify-center px-5 py-16 bg-white">
      <OceanBackground fishCount={3} bubbleCount={14} />
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md relative z-10"
      >
        <div className="flex flex-col items-center mb-8">
          <Link to="/" className="flex items-center gap-2 mb-6">
            <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-ocean-400 to-seaweed-500 flex items-center justify-center shadow-glow-ocean">
              <Fish className="w-6 h-6 text-white" />
            </div>
            <span className="font-display font-bold text-xl text-slate-900">AquaScan</span>
          </Link>
          <h1 className="font-display text-2xl font-bold text-slate-900">Create Account</h1>
          <p className="text-slate-500 text-sm mt-1">Start detecting fish disease with AI</p>
        </div>

        <form onSubmit={handleSubmit} className="glass-card rounded-3xl p-8 space-y-5 shadow-glass">
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              className="bg-coral-50 border border-coral-200 text-coral-600 text-sm rounded-xl px-4 py-3"
            >
              {error}
            </motion.div>
          )}

          {/* Role selector */}
          <div>
            <label className="text-sm font-medium text-slate-600 mb-2 block">I am a...</label>
            <div className="grid grid-cols-2 gap-3">
              <button
                type="button"
                onClick={() => setRole("farmer")}
                className={`rounded-xl p-3.5 border text-left transition-all ${
                  role === "farmer"
                    ? "border-ocean-400 bg-ocean-50 shadow-glow-ocean"
                    : "border-slate-200 bg-white hover:bg-slate-50"
                }`}
              >
                <Sprout className={`w-5 h-5 mb-1.5 ${role === "farmer" ? "text-ocean-600" : "text-slate-400"}`} />
                <div className="text-sm font-semibold text-slate-900">Farmer / User</div>
              </button>
              <button
                type="button"
                onClick={() => setRole("admin")}
                className={`rounded-xl p-3.5 border text-left transition-all ${
                  role === "admin"
                    ? "border-coral-400 bg-coral-50 shadow-glow-coral"
                    : "border-slate-200 bg-white hover:bg-slate-50"
                }`}
              >
                <ShieldCheck className={`w-5 h-5 mb-1.5 ${role === "admin" ? "text-coral-500" : "text-slate-400"}`} />
                <div className="text-sm font-semibold text-slate-900">Admin</div>
              </button>
            </div>
          </div>

          <AnimatePresence>
            {role === "admin" && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="overflow-hidden"
              >
                <label className="text-sm font-medium text-slate-600 mb-1.5 block">
                  Admin Invite Code <span className="text-coral-500">*required</span>
                </label>
                <div className="relative">
                  <KeyRound className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                  <input
                    type="text"
                    value={inviteCode}
                    onChange={(e) => setInviteCode(e.target.value)}
                    placeholder="Contact system owner for code"
                    className="input-field pl-11"
                  />
                </div>
                <p className="text-xs text-slate-400 mt-1.5">
                  Administrator accounts are protected. Public sign-up without a
                  valid invite code will be rejected.
                </p>
              </motion.div>
            )}
          </AnimatePresence>

          <div>
            <label className="text-sm font-medium text-slate-600 mb-1.5 block">Full Name</label>
            <div className="relative">
              <User className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
              <input
                type="text"
                required
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="Jane Farmer"
                className="input-field pl-11"
              />
            </div>
          </div>

          <div>
            <label className="text-sm font-medium text-slate-600 mb-1.5 block">Email</label>
            <div className="relative">
              <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="input-field pl-11"
              />
            </div>
          </div>

          <div>
            <label className="text-sm font-medium text-slate-600 mb-1.5 block">Password</label>
            <div className="relative">
              <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
              <input
                type={showPassword ? "text" : "password"}
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="input-field pl-11 pr-11"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
              >
                {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          <div>
            <label className="text-sm font-medium text-slate-600 mb-1.5 block">Confirm Password</label>
            <div className="relative">
              <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
              <input
                type={showConfirm ? "text" : "password"}
                required
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="••••••••"
                className="input-field pl-11 pr-11"
              />
              <button
                type="button"
                onClick={() => setShowConfirm(!showConfirm)}
                className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
              >
                {showConfirm ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 bg-slate-50 rounded-xl p-3">
            <PasswordRule met={rules.length} label="8+ characters" />
            <PasswordRule met={rules.upper} label="1 uppercase letter" />
            <PasswordRule met={rules.number} label="1 number" />
            <PasswordRule met={rules.match} label="Passwords match" />
          </div>

          <button type="submit" disabled={loading} className="btn-primary w-full !py-3.5">
            {loading ? (
              <span className="w-5 h-5 border-2 border-white/40 border-t-white rounded-full animate-spin" />
            ) : (
              <>
                <UserPlus className="w-4 h-4" /> Create Account
              </>
            )}
          </button>

          <p className="text-center text-sm text-slate-500">
            Already have an account?{" "}
            <Link to="/login" className="text-ocean-600 font-semibold hover:text-ocean-700">
              Log in <ArrowRight className="w-3 h-3 inline" />
            </Link>
          </p>
        </form>
      </motion.div>
    </div>
  );
}
