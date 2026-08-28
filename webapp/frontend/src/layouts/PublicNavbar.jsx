import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Fish, Menu, X } from "lucide-react";
import { useAuth } from "../context/AuthContext";

const NAV_LINKS = [
  { label: "How It Works", href: "#how-it-works" },
  { label: "Technology", href: "#technology" },
  { label: "Diseases", href: "#diseases" },
  { label: "Why AquaScan", href: "#why-us" },
];

export default function PublicNavbar() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const { user } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <motion.nav
      initial={{ y: -30, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
      className={`fixed top-0 inset-x-0 z-50 transition-all duration-300 ${
        scrolled ? "bg-ocean-950/80 backdrop-blur-lg border-b border-white/10 py-3" : "py-5"
      }`}
    >
      <div className="max-w-7xl mx-auto px-5 sm:px-8 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2 group">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-ocean-400 to-seaweed-500 flex items-center justify-center shadow-glow-ocean group-hover:scale-110 transition-transform">
            <Fish className="w-5 h-5 text-white" />
          </div>
          <span className="font-display font-bold text-lg text-white">AquaScan</span>
        </Link>

        <div className="hidden md:flex items-center gap-8">
          {NAV_LINKS.map((link) => (
            <a
              key={link.href}
              href={link.href}
              className="text-sm font-medium text-white/70 hover:text-white transition-colors"
            >
              {link.label}
            </a>
          ))}
        </div>

        <div className="hidden md:flex items-center gap-3">
          {user ? (
            <button
              onClick={() => navigate(user.role === "admin" ? "/admin" : "/dashboard")}
              className="btn-primary !py-2.5 !px-5 text-sm"
            >
              Go to Dashboard
            </button>
          ) : (
            <>
              <Link to="/login" className="text-sm font-semibold text-white/80 hover:text-white px-4 py-2">
                Log In
              </Link>
              <Link to="/register" className="btn-primary !py-2.5 !px-5 text-sm">
                Get Started
              </Link>
            </>
          )}
        </div>

        <button className="md:hidden text-white" onClick={() => setOpen(!open)}>
          {open ? <X /> : <Menu />}
        </button>
      </div>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="md:hidden overflow-hidden bg-ocean-950/95 backdrop-blur-lg border-t border-white/10 mt-3"
          >
            <div className="px-5 py-4 flex flex-col gap-4">
              {NAV_LINKS.map((link) => (
                <a
                  key={link.href}
                  href={link.href}
                  onClick={() => setOpen(false)}
                  className="text-white/80 font-medium"
                >
                  {link.label}
                </a>
              ))}
              <div className="flex flex-col gap-2 pt-2 border-t border-white/10">
                {user ? (
                  <button
                    onClick={() => {
                      setOpen(false);
                      navigate(user.role === "admin" ? "/admin" : "/dashboard");
                    }}
                    className="btn-primary w-full"
                  >
                    Go to Dashboard
                  </button>
                ) : (
                  <>
                    <Link to="/login" onClick={() => setOpen(false)} className="btn-secondary w-full">
                      Log In
                    </Link>
                    <Link to="/register" onClick={() => setOpen(false)} className="btn-primary w-full">
                      Get Started
                    </Link>
                  </>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.nav>
  );
}
