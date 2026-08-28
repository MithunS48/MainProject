import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import {
  Fish, ScanLine, Layers, BrainCircuit, ArrowRight, CheckCircle2,
  Upload, Cpu, ClipboardCheck, ShieldCheck, Sparkles, Users,
  TrendingUp, Microscope, Activity,
} from "lucide-react";
import PublicNavbar from "../layouts/PublicNavbar";
import PublicFooter from "../layouts/PublicFooter";
import OceanBackground from "../components/OceanBackground";
import StatCard from "../components/ui/StatCard";
import heroFish from "../assets/hero-fish.png";
import { DISEASE_ORDER, getDiseaseInfo } from "../utils/diseaseInfo";

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0 },
};

function Section({ id, className = "", children }) {
  return (
    <section id={id} className={`relative py-20 sm:py-28 px-5 sm:px-8 ${className}`}>
      <div className="max-w-7xl mx-auto">{children}</div>
    </section>
  );
}

function SectionHeading({ eyebrow, title, subtitle }) {
  return (
    <motion.div
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: "-80px" }}
      variants={fadeUp}
      transition={{ duration: 0.6 }}
      className="text-center max-w-2xl mx-auto mb-14"
    >
      <span className="pill bg-ocean-50 text-ocean-700 border border-ocean-100 mb-4">
        <Sparkles className="w-3.5 h-3.5" /> {eyebrow}
      </span>
      <h2 className="section-title">{title}</h2>
      {subtitle && (
        <p className="mt-4 text-base sm:text-lg text-slate-500">{subtitle}</p>
      )}
    </motion.div>
  );
}

const HOW_IT_WORKS_STEPS = [
  { icon: Upload, title: "Upload a Fish Photo", desc: "Farmers capture or upload a clear photo of the fish using any phone or camera." },
  { icon: Cpu, title: "AI Feature Extraction", desc: "MobileNetV2 and ConvNeXt independently extract deep visual features from the image." },
  { icon: Layers, title: "Feature Fusion", desc: "The two feature sets are fused into a single, richer 2048-dimensional representation." },
  { icon: BrainCircuit, title: "Polynomial SVM Classification", desc: "A trained Support Vector Machine classifies the fused features into one of 4 categories." },
  { icon: ClipboardCheck, title: "Instant Diagnosis", desc: "The farmer receives an easy-to-understand result with confidence score and guidance." },
];

const TECH_PIPELINE = [
  { label: "Input Image", detail: "224×224×3 RGB", icon: Fish },
  { label: "MobileNetV2", detail: "1,280 features", icon: ScanLine },
  { label: "ConvNeXt", detail: "768 features", icon: Layers },
  { label: "Feature Fusion", detail: "2,048-d vector", icon: Sparkles },
  { label: "Polynomial SVM", detail: "C=1, degree=3", icon: BrainCircuit },
  { label: "Diagnosis", detail: "4 classes", icon: ClipboardCheck },
];

const WHY_US = [
  { icon: TrendingUp, title: "98.29% Accuracy", desc: "Validated on 2,105 held-out test images with a weighted F1-score of 0.9829." },
  { icon: Microscope, title: "Dual-CNN Feature Fusion", desc: "Combines MobileNetV2's efficiency with ConvNeXt's modern architecture for richer features." },
  { icon: Activity, title: "0.9989 Macro AUC", desc: "Near-perfect class separability across all four disease categories." },
  { icon: Users, title: "Built for Farmers", desc: "Simple upload-and-diagnose flow designed for real-world aquaculture use, not just researchers." },
];

export default function LandingPage() {
  return (
    <div className="bg-white min-h-screen">
      <PublicNavbar />

      {/* HERO */}
      <section className="relative pt-32 sm:pt-40 pb-20 px-5 sm:px-8 overflow-hidden">
        <OceanBackground fishCount={4} bubbleCount={16} />
        <div className="max-w-7xl mx-auto w-full grid lg:grid-cols-2 gap-14 items-center">
          <motion.div
            initial={{ opacity: 0, x: -40 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.7 }}
          >
            <span className="pill bg-ocean-50 text-ocean-700 mb-6 border border-ocean-100">
              <ShieldCheck className="w-3.5 h-3.5" /> AI-Assisted Aquaculture Diagnostics
            </span>
            <h1 className="font-display text-4xl sm:text-5xl lg:text-[3.4rem] font-extrabold leading-[1.1] text-slate-900">
              AI-Based Fish Disease{" "}
              <span className="shimmer-text">Detection</span>
            </h1>
            <p className="mt-6 text-lg text-slate-500 max-w-lg leading-relaxed">
              Detect fish diseases quickly using advanced deep-learning
              feature fusion and machine learning. Upload a photo, get an
              instant, easy-to-understand diagnosis.
            </p>
            <div className="mt-9 flex flex-wrap gap-4">
              <Link to="/register" className="btn-primary text-base">
                Get Started <ArrowRight className="w-4 h-4" />
              </Link>
              <a href="#how-it-works" className="btn-secondary text-base">
                See How It Works
              </a>
            </div>

            <div className="mt-12 grid grid-cols-2 gap-4 max-w-md">
              <StatCard icon={TrendingUp} label="Accuracy" value={98.29} suffix="%" decimals={2} accent="seaweed" />
              <StatCard icon={Activity} label="AUC Score" value={0.9989} suffix="" decimals={4} accent="ocean" delay={0.1} />
              <StatCard icon={Layers} label="Disease Classes" value={4} accent="coral" delay={0.2} />
              <StatCard icon={BrainCircuit} label="AI-Powered" value={100} suffix="%" accent="sand" delay={0.3} />
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.85 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="relative flex justify-center"
          >
            <div className="absolute inset-0 bg-ocean-400/15 blur-[100px] rounded-full" />
            <motion.img
              src={heroFish}
              alt="AI-analyzed fish"
              className="relative w-full max-w-md rounded-3xl shadow-glow-ocean animate-float-slow"
            />
          </motion.div>
        </div>
      </section>

      {/* HOW IT WORKS */}
      <Section id="how-it-works">
        <SectionHeading
          eyebrow="How It Works"
          title="From Photo to Diagnosis in Seconds"
          subtitle="A simple, guided pipeline that runs the exact same trained ML model used in the research behind this system."
        />
        <div className="grid md:grid-cols-5 gap-6">
          {HOW_IT_WORKS_STEPS.map((step, i) => (
            <motion.div
              key={step.title}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.08 }}
              whileHover={{ y: -8 }}
              className="glass-card rounded-2xl p-6 relative"
            >
              <div className="absolute -top-3 -left-3 w-8 h-8 rounded-full bg-gradient-to-br from-ocean-400 to-seaweed-500 flex items-center justify-center text-sm font-bold text-white shadow-glow-ocean">
                {i + 1}
              </div>
              <step.icon className="w-8 h-8 text-ocean-500 mb-4" />
              <h3 className="font-display font-semibold text-slate-900 mb-2">{step.title}</h3>
              <p className="text-sm text-slate-500 leading-relaxed">{step.desc}</p>
            </motion.div>
          ))}
        </div>
      </Section>

      {/* AI TECHNOLOGY */}
      <Section id="technology" className="bg-slate-50">
        <SectionHeading
          eyebrow="AI Technology"
          title="Dual-CNN Feature Fusion + Polynomial SVM"
          subtitle="The final deployed model fuses features from two convolutional neural networks before classification."
        />

        <div className="flex flex-wrap items-center justify-center gap-3 mb-16">
          {TECH_PIPELINE.map((step, i) => (
            <div key={step.label} className="flex items-center gap-3">
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: i * 0.1 }}
                className="glass-card rounded-2xl px-5 py-4 flex flex-col items-center gap-2 min-w-[130px] hover:shadow-glow-ocean transition-shadow"
              >
                <step.icon className="w-6 h-6 text-ocean-500" />
                <div className="text-sm font-semibold text-slate-900 text-center">{step.label}</div>
                <div className="text-xs text-slate-500 text-center">{step.detail}</div>
              </motion.div>
              {i < TECH_PIPELINE.length - 1 && (
                <ArrowRight className="w-5 h-5 text-slate-300 hidden sm:block" />
              )}
            </div>
          ))}
        </div>

        <div className="grid sm:grid-cols-3 gap-6 max-w-4xl mx-auto text-center">
          <div className="glass-card rounded-2xl p-6">
            <div className="text-3xl font-display font-bold text-seaweed-600">98.29%</div>
            <div className="text-sm text-slate-500 mt-1">Test Accuracy</div>
          </div>
          <div className="glass-card rounded-2xl p-6">
            <div className="text-3xl font-display font-bold text-ocean-600">0.9829</div>
            <div className="text-sm text-slate-500 mt-1">F1-Score</div>
          </div>
          <div className="glass-card rounded-2xl p-6">
            <div className="text-3xl font-display font-bold text-coral-500">0.9989</div>
            <div className="text-sm text-slate-500 mt-1">Macro AUC</div>
          </div>
        </div>
      </Section>

      {/* SUPPORTED DISEASES */}
      <Section id="diseases">
        <SectionHeading
          eyebrow="Supported Diseases"
          title="Four Categories, One Confident Model"
          subtitle="AquaScan recognizes the following fish health categories from a single image."
        />
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {DISEASE_ORDER.map((key, i) => {
            const d = getDiseaseInfo(key);
            const colorMap = {
              coral: "from-coral-50 to-white border-coral-100 text-coral-600",
              seaweed: "from-seaweed-50 to-white border-seaweed-100 text-seaweed-600",
              sand: "from-sand-50 to-white border-sand-200 text-sand-600",
            };
            return (
              <motion.div
                key={key}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                whileHover={{ y: -6 }}
                className={`rounded-2xl p-6 bg-gradient-to-b border ${colorMap[d.color]} glass-card`}
              >
                <Fish className="w-8 h-8 mb-4" />
                <h3 className="font-display font-semibold text-slate-900 mb-2">{d.shortName}</h3>
                <p className="text-sm text-slate-500 leading-relaxed">{d.summary}</p>
              </motion.div>
            );
          })}
        </div>
      </Section>

      {/* WHY THIS SYSTEM */}
      <Section id="why-us" className="bg-slate-50">
        <SectionHeading
          eyebrow="Why This System"
          title="Research-Grade Accuracy, Farmer-Friendly Design"
        />
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {WHY_US.map((item, i) => (
            <motion.div
              key={item.title}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.1 }}
              className="glass-card rounded-2xl p-6"
            >
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-ocean-500 to-seaweed-500 flex items-center justify-center mb-4">
                <item.icon className="w-6 h-6 text-white" />
              </div>
              <h3 className="font-display font-semibold text-slate-900 mb-2">{item.title}</h3>
              <p className="text-sm text-slate-500 leading-relaxed">{item.desc}</p>
            </motion.div>
          ))}
        </div>
      </Section>

      {/* CTA */}
      <Section id="cta">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="relative rounded-3xl overflow-hidden bg-gradient-to-br from-ocean-600 to-seaweed-600 p-10 sm:p-16 text-center"
        >
          <OceanBackground variant="dark" fishCount={2} bubbleCount={10} />
          <h2 className="font-display text-3xl sm:text-4xl font-bold text-white relative z-10">
            Ready to protect your fish stock?
          </h2>
          <p className="text-white/80 mt-4 max-w-xl mx-auto relative z-10">
            Join farmers using AI-assisted diagnostics to catch disease early
            and act with confidence.
          </p>
          <div className="mt-8 flex justify-center gap-4 relative z-10">
            <Link to="/register" className="btn-primary bg-white !bg-none text-ocean-700 hover:!text-ocean-800">
              Create Free Account <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
          <div className="flex items-center justify-center gap-2 mt-6 text-white/70 text-sm relative z-10">
            <CheckCircle2 className="w-4 h-4" /> No credit card required &nbsp;•&nbsp;
            <CheckCircle2 className="w-4 h-4" /> Instant AI diagnosis
          </div>
        </motion.div>
      </Section>

      <PublicFooter />
    </div>
  );
}
