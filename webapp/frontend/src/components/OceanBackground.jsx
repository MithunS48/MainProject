import { useMemo } from "react";

/**
 * Lightweight animated aquaculture background: soft gradient wash,
 * subtle grid, rising bubbles, and a couple of drifting fish silhouettes.
 * Pure CSS animation (see tailwind.config.js keyframes) — no JS render
 * loop, so it stays smooth and cheap. Designed to sit BEHIND content at
 * low opacity so it never competes for attention or muddies text.
 */
export default function OceanBackground({ variant = "light", fishCount = 3, bubbleCount = 14 }) {
  const bubbles = useMemo(
    () =>
      Array.from({ length: bubbleCount }).map((_, i) => ({
        id: i,
        left: Math.random() * 100,
        size: 4 + Math.random() * 9,
        delay: Math.random() * 8,
        duration: 7 + Math.random() * 8,
      })),
    [bubbleCount]
  );

  const fishes = useMemo(
    () =>
      Array.from({ length: fishCount }).map((_, i) => ({
        id: i,
        top: 12 + Math.random() * 68,
        size: 26 + Math.random() * 26,
        delay: Math.random() * 10,
        duration: 18 + Math.random() * 10,
        opacity: 0.06 + Math.random() * 0.07,
      })),
    [fishCount]
  );

  const isDark = variant === "dark";

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none -z-10">
      <div
        className={
          isDark
            ? "absolute inset-0 bg-ocean-gradient"
            : "absolute inset-0 bg-gradient-to-b from-ocean-50 via-white to-white"
        }
      />
      <div className="absolute inset-0 bg-aqua-radial" />
      {isDark && <div className="absolute inset-0 bg-noise-grid opacity-40" />}

      {fishes.map((f) => (
        <svg
          key={f.id}
          viewBox="0 0 64 32"
          className="absolute animate-swim"
          style={{
            top: `${f.top}%`,
            width: `${f.size}px`,
            height: `${f.size / 2}px`,
            opacity: f.opacity,
            animationDuration: `${f.duration}s`,
            animationDelay: `${f.delay}s`,
          }}
        >
          <path
            d="M2 16 C10 4, 34 2, 46 12 C52 6, 60 8, 62 12 C58 16, 52 18, 46 20 C34 30, 10 28, 2 16 Z"
            fill={isDark ? "#7fdcff" : "#0d6da7"}
          />
        </svg>
      ))}

      {bubbles.map((b) => (
        <div
          key={b.id}
          className="absolute bottom-0 rounded-full animate-bubble"
          style={{
            left: `${b.left}%`,
            width: `${b.size}px`,
            height: `${b.size}px`,
            background: isDark
              ? "radial-gradient(circle at 30% 30%, rgba(255,255,255,0.65), rgba(255,255,255,0.05))"
              : "radial-gradient(circle at 30% 30%, rgba(28,171,242,0.35), rgba(28,171,242,0.03))",
            animationDuration: `${b.duration}s`,
            animationDelay: `${b.delay}s`,
          }}
        />
      ))}
    </div>
  );
}
