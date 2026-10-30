/**
 * PWAInstallPrompt.jsx
 * ─────────────────────
 * Shows an install banner when the browser fires the beforeinstallprompt event.
 * Also shows an update-available banner when a new service worker is ready.
 *
 * Place this component once in App.jsx (outside <Routes>).
 * It renders nothing until there is something to show.
 */

import { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Download, X, RefreshCcw, Fish } from "lucide-react";

export default function PWAInstallPrompt() {
  const [installPrompt, setInstallPrompt] = useState(null);
  const [showInstall, setShowInstall]     = useState(false);
  const [needRefresh, setNeedRefresh]     = useState(false);
  const swRef = useRef(null);
  const [dismissed] = useState(
    () => localStorage.getItem("pwa_install_dismissed") === "1"
  );

  // Register service worker
  useEffect(() => {
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.register("/sw.js").then((reg) => {
        swRef.current = reg;
        reg.addEventListener("updatefound", () => {
          const newWorker = reg.installing;
          if (!newWorker) return;
          newWorker.addEventListener("statechange", () => {
            if (newWorker.state === "installed" && navigator.serviceWorker.controller) {
              setNeedRefresh(true);
            }
          });
        });
      }).catch(console.warn);
    }
  }, []);

  // Capture install prompt
  useEffect(() => {
    const handler = (e) => {
      e.preventDefault();
      setInstallPrompt(e);
      if (!dismissed) setShowInstall(true);
    };
    window.addEventListener("beforeinstallprompt", handler);
    return () => window.removeEventListener("beforeinstallprompt", handler);
  }, [dismissed]);

  const handleInstall = async () => {
    if (!installPrompt) return;
    installPrompt.prompt();
    const { outcome } = await installPrompt.userChoice;
    if (outcome === "accepted") setShowInstall(false);
    setInstallPrompt(null);
  };

  const dismissInstall = () => {
    setShowInstall(false);
    localStorage.setItem("pwa_install_dismissed", "1");
  };

  const updateApp = () => {
    swRef.current?.waiting?.postMessage({ type: "SKIP_WAITING" });
    window.location.reload();
  };

  return (
    <>
      {/* ── Install banner ── */}
      <AnimatePresence>
        {showInstall && (
          <motion.div
            initial={{ y: 100, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 100, opacity: 0 }}
            transition={{ type: "spring", stiffness: 280, damping: 24 }}
            className="fixed bottom-4 left-4 right-4 sm:left-auto sm:right-4 sm:w-80 z-50"
          >
            <div className="bg-white rounded-2xl shadow-xl border border-sky-100 p-4">
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-sky-500 to-blue-600 flex items-center justify-center shrink-0">
                  <Fish className="w-5 h-5 text-white" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-semibold text-slate-800 text-sm">Install AquaScan</p>
                  <p className="text-slate-500 text-xs mt-0.5">
                    Add to your home screen for fast access — works offline too.
                  </p>
                </div>
                <button
                  onClick={dismissInstall}
                  className="text-slate-300 hover:text-slate-500 transition-colors shrink-0"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              <div className="flex gap-2 mt-3">
                <button
                  onClick={handleInstall}
                  className="flex-1 inline-flex items-center justify-center gap-2 py-2 rounded-xl bg-sky-500 text-white text-sm font-semibold hover:bg-sky-400 transition-colors"
                >
                  <Download className="w-4 h-4" /> Install
                </button>
                <button
                  onClick={dismissInstall}
                  className="px-4 py-2 rounded-xl border border-slate-200 text-slate-500 text-sm hover:bg-slate-50 transition-colors"
                >
                  Not now
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Update banner ── */}
      <AnimatePresence>
        {needRefresh && (
          <motion.div
            initial={{ y: -80, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: -80, opacity: 0 }}
            transition={{ type: "spring", stiffness: 280, damping: 24 }}
            className="fixed top-4 left-1/2 -translate-x-1/2 z-50 w-72"
          >
            <div className="bg-white rounded-2xl shadow-xl border border-sky-100 px-4 py-3 flex items-center gap-3">
              <RefreshCcw className="w-5 h-5 text-sky-500 shrink-0" />
              <span className="text-sm text-slate-700 flex-1">New version available!</span>
              <button
                onClick={updateApp}
                className="px-3 py-1.5 rounded-xl bg-sky-500 text-white text-xs font-semibold hover:bg-sky-400 transition-colors shrink-0"
              >
                Update
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
