export default function Badge({ children, color = "ocean", className = "" }) {
  const colorMap = {
    ocean: "bg-ocean-500/15 text-ocean-300 border-ocean-500/30",
    seaweed: "bg-seaweed-500/15 text-seaweed-300 border-seaweed-500/30",
    coral: "bg-coral-500/15 text-coral-300 border-coral-500/30",
    sand: "bg-sand-400/15 text-sand-300 border-sand-400/30",
    slate: "bg-white/10 text-white/60 border-white/15",
  };
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs font-semibold ${colorMap[color]} ${className}`}
    >
      {children}
    </span>
  );
}
