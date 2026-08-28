import { motion } from "framer-motion";
import CountUp from "./CountUp";

export default function StatCard({ icon: Icon, label, value, suffix = "", decimals = 0, accent = "ocean", delay = 0 }) {
  const accentMap = {
    ocean: "from-ocean-400 to-ocean-600 shadow-glow-ocean",
    seaweed: "from-seaweed-400 to-seaweed-600 shadow-glow-seaweed",
    coral: "from-coral-400 to-coral-600 shadow-glow-coral",
    sand: "from-sand-300 to-sand-500",
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay }}
      whileHover={{ y: -6 }}
      className="glass-card rounded-2xl p-5 sm:p-6 flex items-center gap-4"
    >
      <div className={`shrink-0 rounded-xl bg-gradient-to-br ${accentMap[accent]} p-3`}>
        {Icon && <Icon className="w-6 h-6 text-white" strokeWidth={2} />}
      </div>
      <div>
        <div className="text-2xl sm:text-3xl font-display font-bold text-white">
          <CountUp end={value} decimals={decimals} />
          {suffix}
        </div>
        <div className="text-xs sm:text-sm text-white/60 font-medium">{label}</div>
      </div>
    </motion.div>
  );
}
