import { Suspense, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Html, Float, Text3D, Center, OrbitControls, Stars } from "@react-three/drei";
import { motion, AnimatePresence } from "framer-motion";
import { Link } from "react-router-dom";
import {
  Fish, ScanLine, Layers, BrainCircuit, ArrowRight,
  CheckCircle2, Upload, Cpu, ClipboardCheck, ShieldCheck,
  Sparkles, TrendingUp, Microscope, Activity, ChevronDown,
} from "lucide-react";
import * as THREE from "three";
import OceanScene from "../components/three/OceanScene";
import PublicNavbar from "../layouts/PublicNavbar";
import PublicFooter from "../layouts/PublicFooter";
import { DISEASE_ORDER, getDiseaseInfo } from "../utils/diseaseInfo";

// ─── Floating 3D Card ────────────────────────────────────────────────────────
function FloatingCard({ position, color, label, value }) {
  const mesh = useRef();
  useFrame((state) => {
    mesh.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.15;
    mesh.current.position.y =
      position[1] + Math.sin(state.clock.elapsedTime * 0.8 + position[0]) * 0.12;
  });
  return (
    <Float speed={1.5} rotationIntensity={0.3} floatIntensity={0.5}>
      <mesh ref={mesh} position={position} castShadow>
        <boxGeometry args={[1.6, 0.9, 0.08]} />
        <meshStandardMaterial
          color={color}
          transparent
          opacity={0.18}
          roughness={0}
          metalness={0.8}
          envMapIntensity={1}
        />
        <Html
          center
          distanceFactor={4}
          style={{ pointerEvents: "none", width: 140 }}
        >
          <div className="text-center">
            <div className="text-sky-600 font-bold text-xl leading-none">{value}</div>
            <div className="text-blue-200 text-xs mt-1 opacity-80">{label}</div>
          </div>
        </Html>
      </mesh>
    </Float>
  );
}

// ─── 3D Rotating Helix Structure — moved to right side ───────────────────────
function HelixOrbs() {
  const group = useRef();
  useFrame((state) => {
    group.current.rotation.y = state.clock.elapsedTime * 0.3;
  });
  const orbs = Array.from({ length: 20 }, (_, i) => {
    const angle = (i / 20) * Math.PI * 4;
    const r = 1.0;
    return {
      x: Math.cos(angle) * r,
      y: (i / 20) * 3.5 - 1.75,
      z: Math.sin(angle) * r,
      color: i % 3 === 0 ? "#00d4ff" : i % 3 === 1 ? "#00ff88" : "#ff6b35",
    };
  });
  return (
    <group ref={group} position={[7, 0, -3]}>
      {orbs.map((o, i) => (
        <mesh key={i} position={[o.x, o.y, o.z]}>
          <sphereGeometry args={[0.07, 8, 8]} />
          <meshStandardMaterial
            color={o.color}
            emissive={o.color}
            emissiveIntensity={0.6}
          />
        </mesh>
      ))}
    </group>
  );
}

// ─── 3D Stat Spheres — pushed to far left/right edges ────────────────────────
function StatSpheres() {
  const stats = [
    { pos: [-8, 2.5, -4],  color: "#00d4ff", label: "98.29%", sub: "Accuracy" },
    { pos: [-7, -2.0, -4], color: "#00ff88", label: "0.9989", sub: "AUC"      },
    { pos: [ 8, 2.5, -4],  color: "#ff6b35", label: "4",      sub: "Classes"  },
    { pos: [ 7, -2.0, -4], color: "#a855f7", label: "2048d",  sub: "Features" },
  ];
  return (
    <>
      {stats.map((s, i) => (
        <Float key={i} speed={1 + i * 0.2} floatIntensity={0.3}>
          <mesh position={s.pos}>
            <sphereGeometry args={[0.4, 16, 16]} />
            <meshStandardMaterial
              color={s.color}
              transparent
              opacity={0.12}
              roughness={0}
              metalness={1}
            />
            <Html center distanceFactor={5} style={{ pointerEvents: "none" }}>
              <div className="text-center">
                <div
                  className="font-bold text-base leading-none"
                  style={{ color: s.color, textShadow: `0 0 10px ${s.color}` }}
                >
                  {s.label}
                </div>
                <div className="text-blue-200 text-xs opacity-70">{s.sub}</div>
              </div>
            </Html>
          </mesh>
        </Float>
      ))}
    </>
  );
}

// ─── 3D Pipeline Nodes — pushed below centre, not overlapping title ───────────
function PipelineNode({ position, label, color, index }) {
  const mesh = useRef();
  useFrame((state) => {
    mesh.current.rotation.y = state.clock.elapsedTime * 0.5 + index;
    mesh.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.3 + index) * 0.2;
  });
  return (
    <Float speed={1.2} floatIntensity={0.3}>
      <group position={position}>
        <mesh ref={mesh}>
          <octahedronGeometry args={[0.22, 0]} />
          <meshStandardMaterial
            color={color}
            emissive={color}
            emissiveIntensity={0.5}
            roughness={0.1}
            metalness={0.9}
          />
        </mesh>
        <Html center distanceFactor={5} style={{ pointerEvents: "none" }}>
          <div
            className="text-xs font-semibold px-2 py-0.5 rounded-full"
            style={{
              background: "rgba(220,240,255,0.9)",
              color,
              border: `1px solid ${color}`,
              whiteSpace: "nowrap",
              textShadow: `0 0 6px ${color}`,
              fontSize: "10px",
            }}
          >
            {label}
          </div>
        </Html>
      </group>
    </Float>
  );
}

// ─── Hero 3D Canvas ───────────────────────────────────────────────────────────
function HeroCanvas() {
  return (
    <Canvas
      camera={{ position: [0, 0, 8], fov: 60 }}
      gl={{ antialias: true, alpha: false }}
      dpr={[1, 1.5]}
      shadows
    >
      <Suspense fallback={null}>
        <ambientLight intensity={1.4} color="#ffffff" />
        <pointLight position={[0, 8, 4]} intensity={1.5} color="#56ccf2" />
        <OceanScene />
        <StatSpheres />
        <HelixOrbs />
        <PipelineNode position={[-9,  3.5, -2]} label="MobileNetV2" color="#00d4ff" index={0} />
        <PipelineNode position={[-9, -3.5, -2]} label="ConvNeXt"    color="#00ff88" index={1} />
        <PipelineNode position={[ 9,  3.5, -2]} label="Fusion"      color="#a855f7" index={2} />
        <PipelineNode position={[ 9, -3.5, -2]} label="SVM"         color="#ff6b35" index={3} />
      </Suspense>
    </Canvas>
  );
}

// ─── Section wrapper ──────────────────────────────────────────────────────────
function Section({ id, className = "", children }) {
  return (
    <section id={id} className={`relative py-24 px-5 sm:px-10 ${className}`}>
      <div className="max-w-7xl mx-auto">{children}</div>
    </section>
  );
}

const fadeUp = { hidden: { opacity: 0, y: 40 }, visible: { opacity: 1, y: 0 } };

function SectionHeading({ eyebrow, title, subtitle }) {
  return (
    <motion.div
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true }}
      variants={fadeUp}
      transition={{ duration: 0.6 }}
      className="text-center max-w-2xl mx-auto mb-14"
    >
      <span className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-sm font-medium bg-sky-100/80 text-sky-600 border border-sky-300 mb-4">
        <Sparkles className="w-3.5 h-3.5" /> {eyebrow}
      </span>
      <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-800">{title}</h2>
      {subtitle && <p className="mt-4 text-slate-500 text-base">{subtitle}</p>}
    </motion.div>
  );
}

// ─── HOW IT WORKS steps ───────────────────────────────────────────────────────
const STEPS = [
  { icon: Upload,        title: "Upload Photo",       desc: "Capture or upload any fish image from your phone or camera." },
  { icon: Cpu,           title: "CNN Feature Extract", desc: "MobileNetV2 & ConvNeXt extract deep visual features independently." },
  { icon: Layers,        title: "Feature Fusion",      desc: "Both feature sets merge into a 2048-dimensional representation." },
  { icon: BrainCircuit,  title: "SVM Classification",  desc: "Polynomial SVM classifies fused features into one of 4 categories." },
  { icon: ClipboardCheck, title: "Instant Diagnosis",  desc: "Confidence score, disease info, and guidance delivered instantly." },
];

// ─── WHY US ───────────────────────────────────────────────────────────────────
const WHY = [
  { icon: TrendingUp,  title: "98.29% Accuracy",    desc: "Validated on 2,105 held-out test images with F1-score of 0.9829." },
  { icon: Microscope,  title: "Dual-CNN Fusion",     desc: "MobileNetV2 + ConvNeXt features combined for richer representations." },
  { icon: Activity,    title: "0.9989 AUC",          desc: "Near-perfect class separability across all four disease categories." },
  { icon: Fish,        title: "Farmer-Friendly",     desc: "Simple upload-and-diagnose flow designed for real-world aquaculture." },
];

// ─── Main Page ────────────────────────────────────────────────────────────────
export default function Landing3D() {
  const [hovered, setHovered] = useState(null);

  return (
    <div className="bg-[#e8f4fd] min-h-screen text-slate-800">
      <PublicNavbar />

      {/* ── HERO ── */}
      <section className="relative h-screen overflow-hidden">
        {/* 3D Canvas — full background */}
        <div className="absolute inset-0">
          <HeroCanvas />
        </div>

        {/* Overlay gradient */}
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-[#e8f4fd] pointer-events-none" />

        {/* Hero text overlay */}
        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none px-5">
          <motion.div
            initial={{ opacity: 0, y: -30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <motion.span
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2 }}
              className="inline-flex items-center gap-2 px-5 py-2 rounded-full bg-sky-100/80 border border-cyan-500/40 text-sky-600 text-sm font-medium mb-6 backdrop-blur-sm"
            >
              <ShieldCheck className="w-4 h-4" />
              AI-Assisted Aquaculture Diagnostics
            </motion.span>

            <h1 className="font-extrabold text-5xl sm:text-6xl lg:text-7xl leading-[1.08] tracking-tight">
              <span className="text-slate-800">AI-Based Fish</span>
              <br />
              <span
                className="bg-clip-text text-transparent"
                style={{
                  backgroundImage: "linear-gradient(135deg, #0284c7 0%, #0ea5e9 50%, #38bdf8 100%)",
                }}
              >
                Disease Detection
              </span>
            </h1>

            <p className="mt-6 text-lg text-slate-500 max-w-xl mx-auto leading-relaxed">
              Upload a fish photo. Get an instant AI-powered diagnosis using
              dual-CNN feature fusion and Polynomial SVM — 98.29% accurate.
            </p>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="mt-10 flex flex-wrap justify-center gap-4 pointer-events-auto"
            >
              <Link
                to="/register"
                className="inline-flex items-center gap-2 px-7 py-3.5 rounded-xl font-semibold text-sm bg-gradient-to-r from-sky-500 to-blue-600 text-white hover:from-cyan-400 hover:to-blue-500 transition-all shadow-lg shadow-sky-500/25 hover:shadow-sky-400/40 hover:scale-105 active:scale-95"
              >
                Get Started <ArrowRight className="w-4 h-4" />
              </Link>
              <a
                href="#how-it-works"
                className="inline-flex items-center gap-2 px-7 py-3.5 rounded-xl font-semibold text-sm bg-white/80 border border-sky-200 text-slate-700 hover:bg-white shadow-sm transition-all backdrop-blur-sm hover:scale-105"
              >
                See How It Works
              </a>
            </motion.div>
          </motion.div>
        </div>

        {/* Scroll hint */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2 }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-1 text-slate-500 text-xs pointer-events-none"
        >
          <span>Scroll to explore</span>
          <motion.div
            animate={{ y: [0, 6, 0] }}
            transition={{ repeat: Infinity, duration: 1.5 }}
          >
            <ChevronDown className="w-4 h-4" />
          </motion.div>
        </motion.div>
      </section>

      {/* ── HOW IT WORKS ── */}
      <Section id="how-it-works">
        <SectionHeading
          eyebrow="How It Works"
          title="From Photo to Diagnosis in Seconds"
          subtitle="The same research-grade pipeline — deployed for real-world use."
        />
        <div className="grid md:grid-cols-5 gap-5">
          {STEPS.map((step, i) => (
            <motion.div
              key={step.title}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.1 }}
              whileHover={{ y: -8, scale: 1.03 }}
              className="relative rounded-2xl p-6 border border-sky-200 bg-white/80 shadow-sm backdrop-blur-sm group cursor-default"
              style={{ boxShadow: "0 2px 16px rgba(14,165,233,0.08)" }}
            >
              <div className="absolute -top-3 -left-3 w-8 h-8 rounded-full bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center text-sm font-bold text-slate-800 shadow-lg shadow-cyan-500/30">
                {i + 1}
              </div>
              <step.icon className="w-8 h-8 text-sky-500 mb-4 group-hover:text-sky-600 transition-colors" />
              <h3 className="font-bold text-slate-800 mb-2 text-sm">{step.title}</h3>
              <p className="text-xs text-slate-500 leading-relaxed">{step.desc}</p>
              {/* Glow on hover */}
              <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none"
                style={{ background: "radial-gradient(circle at 50% 0%, rgba(0,212,255,0.08), transparent 70%)" }} />
            </motion.div>
          ))}
        </div>
      </Section>

      {/* ── 3D DEMO VIDEO ── */}
      <Section id="demo" className="bg-gradient-to-b from-[#e8f4fd] to-[#daeef9]">
        <SectionHeading
          eyebrow="Live Demo"
          title="See AquaScan in Action"
          subtitle="Watch how AquaScan detects fish diseases in real time."
        />
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
          className="relative max-w-4xl mx-auto"
        >
          <div className="absolute -inset-1 bg-gradient-to-r from-cyan-500 via-blue-500 to-purple-500 rounded-3xl blur-xl opacity-30 animate-pulse" />
          <div className="relative rounded-3xl overflow-hidden border border-sky-200 bg-[#e8f4fd]">
            <div className="flex items-center gap-2 px-4 py-3 bg-[#daeef9] border-b border-sky-200">
              <div className="w-3 h-3 rounded-full bg-red-400/80" />
              <div className="w-3 h-3 rounded-full bg-yellow-400/80" />
              <div className="w-3 h-3 rounded-full bg-green-400/80" />
              <div className="flex-1 mx-4 h-6 bg-sky-100 rounded-lg flex items-center px-3 border border-sky-200">
                <span className="text-sky-600 text-xs">aquascan.ai/detect</span>
              </div>
            </div>
            <video className="w-full" controls preload="metadata" poster="/demo_poster.jpg" playsInline>
              <source src="/AquaScan_Demo.mp4" type="video/mp4" />
            </video>
          </div>
          <p className="text-center text-sm text-slate-400 mt-4">
            Full demo — upload, AI pipeline, confidence scores, and diagnosis
          </p>
        </motion.div>
      </Section>

      {/* ── AI TECHNOLOGY ── */}
      <Section id="technology">
        <SectionHeading
          eyebrow="AI Technology"
          title="Dual-CNN Feature Fusion + Polynomial SVM"
          subtitle="The final deployed model fuses features from two CNNs before SVM classification."
        />

        {/* 3D Pipeline canvas */}
        <div className="h-64 rounded-3xl overflow-hidden border border-sky-200 mb-12 relative">
          <Canvas camera={{ position: [0, 0, 6], fov: 55 }} dpr={[1, 1.5]}>
            <Suspense fallback={null}>
              <ambientLight intensity={0.6} color="#0077b6" />
              <pointLight position={[5, 5, 5]} intensity={1} color="#00d4ff" />
              <Stars radius={50} depth={20} count={500} factor={2} fade />
              {[
                { pos: [-5, 0, 0], label: "Input Image", color: "#00d4ff", i: 0 },
                { pos: [-2.5, 0, 0], label: "MobileNetV2", color: "#00ff88", i: 1 },
                { pos: [0, 0, 0], label: "ConvNeXt", color: "#a855f7", i: 2 },
                { pos: [2.5, 0, 0], label: "Fusion 2048d", color: "#ff6b35", i: 3 },
                { pos: [5, 0, 0], label: "SVM → Class", color: "#ffd60a", i: 4 },
              ].map((n) => (
                <PipelineNode key={n.label} position={n.pos} label={n.label} color={n.color} index={n.i} />
              ))}
            </Suspense>
          </Canvas>
          <div className="absolute inset-0 pointer-events-none bg-gradient-to-r from-[#e8f4fd] via-transparent to-[#e8f4fd]" />
        </div>

        <div className="grid sm:grid-cols-3 gap-6 max-w-3xl mx-auto text-center">
          {[
            { val: "98.29%", label: "Test Accuracy", color: "text-sky-500" },
            { val: "0.9829", label: "F1-Score",      color: "text-green-600" },
            { val: "0.9989", label: "Macro AUC",     color: "text-purple-600" },
          ].map((s) => (
            <motion.div
              key={s.label}
              whileHover={{ scale: 1.05 }}
              className="rounded-2xl p-6 border border-sky-200 bg-white/80 shadow-sm"
            >
              <div className={`text-3xl font-extrabold ${s.color}`}>{s.val}</div>
              <div className="text-slate-500 text-sm mt-1">{s.label}</div>
            </motion.div>
          ))}
        </div>
      </Section>

      {/* ── DISEASES ── */}
      <Section id="diseases" className="bg-gradient-to-b from-[#e8f4fd] to-[#daeef9]">
        <SectionHeading
          eyebrow="Supported Diseases"
          title="Four Categories, One Confident Model"
          subtitle="AquaScan recognises the following fish health categories from a single image."
        />
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {DISEASE_ORDER.map((key, i) => {
            const d = getDiseaseInfo(key);
            const colors = [
              { border: "border-red-200",    glow: "rgba(239,68,68,0.06)",    text: "text-red-500",    icon: "text-red-500"    },
              { border: "border-blue-200",   glow: "rgba(59,130,246,0.06)",   text: "text-blue-500",   icon: "text-blue-500"   },
              { border: "border-green-200",  glow: "rgba(34,197,94,0.06)",    text: "text-green-600",  icon: "text-green-600"  },
              { border: "border-orange-200", glow: "rgba(249,115,22,0.06)",   text: "text-orange-500", icon: "text-orange-500" },
            ];
            const c = colors[i];
            return (
              <motion.div
                key={key}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                whileHover={{ y: -8, scale: 1.02 }}
                onHoverStart={() => setHovered(key)}
                onHoverEnd={() => setHovered(null)}
                className={`rounded-2xl p-6 border ${c.border} bg-white/80 shadow-sm relative overflow-hidden cursor-default`}
              >
                <div
                  className="absolute inset-0 rounded-2xl transition-opacity duration-300"
                  style={{
                    background: `radial-gradient(circle at 50% 0%, ${c.glow}, transparent 70%)`,
                    opacity: hovered === key ? 1 : 0,
                  }}
                />
                <Fish className={`w-8 h-8 ${c.icon} mb-4 relative z-10`} />
                <h3 className={`font-bold ${c.text} mb-2 relative z-10`}>{d.shortName}</h3>
                <p className="text-sm text-slate-500 leading-relaxed relative z-10">{d.summary}</p>
              </motion.div>
            );
          })}
        </div>
      </Section>

      {/* ── WHY THIS SYSTEM ── */}
      <Section id="why-us">
        <SectionHeading
          eyebrow="Why This System"
          title="Research-Grade Accuracy, Farmer-Friendly Design"
        />
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {WHY.map((item, i) => (
            <motion.div
              key={item.title}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.1 }}
              whileHover={{ y: -6 }}
              className="rounded-2xl p-6 border border-sky-200 bg-white/80 shadow-sm group"
            >
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center mb-4 shadow-lg shadow-cyan-500/20 group-hover:scale-110 transition-transform">
                <item.icon className="w-6 h-6 text-slate-800" />
              </div>
              <h3 className="font-bold text-slate-800 mb-2">{item.title}</h3>
              <p className="text-sm text-slate-500 leading-relaxed">{item.desc}</p>
            </motion.div>
          ))}
        </div>
      </Section>

      {/* ── CTA ── */}
      <Section id="cta">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="relative rounded-3xl overflow-hidden"
        >
          {/* 3D Canvas background */}
          <div className="absolute inset-0 h-full">
            <Canvas camera={{ position: [0, 0, 6], fov: 60 }} dpr={[1, 1.2]}>
              <Suspense fallback={null}>
                <ambientLight intensity={0.3} color="#0077b6" />
                <Stars radius={60} depth={30} count={400} factor={3} fade />
                <OceanScene />
              </Suspense>
            </Canvas>
          </div>
          <div className="absolute inset-0 bg-gradient-to-br from-cyan-950/90 to-blue-950/90 backdrop-blur-sm" />
          <div className="relative z-10 p-10 sm:p-16 text-center">
            <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-800">
              Ready to protect your fish stock?
            </h2>
            <p className="text-slate-500 mt-4 max-w-xl mx-auto">
              Join farmers using AI-assisted diagnostics to catch disease early
              and act with confidence.
            </p>
            <div className="mt-8 flex justify-center gap-4">
              <Link
                to="/register"
                className="inline-flex items-center gap-2 px-8 py-4 rounded-xl font-semibold bg-gradient-to-r from-sky-500 to-blue-600 text-white hover:from-cyan-400 hover:to-blue-500 transition-all shadow-lg shadow-cyan-500/30 hover:scale-105 active:scale-95"
              >
                Create Free Account <ArrowRight className="w-4 h-4" />
              </Link>
            </div>
            <div className="flex items-center justify-center gap-3 mt-6 text-slate-500 text-sm">
              <CheckCircle2 className="w-4 h-4" /> No credit card required
              <span>•</span>
              <CheckCircle2 className="w-4 h-4" /> Instant AI diagnosis
            </div>
          </div>
        </motion.div>
      </Section>

      <PublicFooter />
    </div>
  );
}

