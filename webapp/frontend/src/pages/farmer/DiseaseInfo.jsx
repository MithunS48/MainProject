import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ShieldCheck, ChevronDown, Stethoscope, ShieldAlert, Sprout, Fish } from "lucide-react";
import { DISEASE_ORDER, getDiseaseInfo } from "../../utils/diseaseInfo";

const colorMap = {
  coral: {
    bg: "from-coral-50 to-white border-coral-100",
    text: "text-coral-600",
    icon: "bg-coral-100 text-coral-500",
  },
  seaweed: {
    bg: "from-seaweed-50 to-white border-seaweed-100",
    text: "text-seaweed-600",
    icon: "bg-seaweed-100 text-seaweed-600",
  },
  sand: {
    bg: "from-sand-50 to-white border-sand-100",
    text: "text-sand-600",
    icon: "bg-sand-100 text-sand-600",
  },
  ocean: {
    bg: "from-ocean-50 to-white border-ocean-100",
    text: "text-ocean-600",
    icon: "bg-ocean-100 text-ocean-600",
  },
};

export default function DiseaseInfo() {
  const [open, setOpen] = useState(DISEASE_ORDER[0]);

  return (
    <div className="space-y-6 max-w-4xl">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="font-display text-2xl sm:text-3xl font-bold text-slate-900">Disease Information</h1>
        <p className="text-slate-500 mt-1">
          Learn about the four fish health categories AquaScan can identify — symptoms, prevention, and
          management guidance.
        </p>
      </motion.div>

      <div className="bg-ocean-50 border border-ocean-100 rounded-xl px-4 py-3.5 flex items-start gap-3 text-sm text-slate-600">
        <ShieldCheck className="w-5 h-5 text-ocean-500 shrink-0 mt-0.5" />
        <p>
          <strong className="text-slate-800">Important:</strong> This information is AI-assisted and
          educational in nature. It does not constitute a certified veterinary diagnosis. Always consult
          a qualified aquaculture or veterinary professional for confirmed diagnosis and treatment.
        </p>
      </div>

      <div className="space-y-3">
        {DISEASE_ORDER.map((key, i) => {
          const d = getDiseaseInfo(key);
          const c = colorMap[d.color] || colorMap.ocean;
          const isOpen = open === key;
          return (
            <motion.div
              key={key}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.06 }}
              className={`rounded-2xl border bg-gradient-to-b glass-card ${c.bg} overflow-hidden`}
            >
              <button
                onClick={() => setOpen(isOpen ? null : key)}
                className="w-full flex items-center gap-4 p-5 text-left"
              >
                <div className={`w-11 h-11 rounded-xl flex items-center justify-center shrink-0 ${c.icon}`}>
                  {d.isHealthy ? <Sprout className="w-5 h-5" /> : <Fish className="w-5 h-5" />}
                </div>
                <div className="flex-1">
                  <h3 className="font-display font-semibold text-slate-900">{d.name}</h3>
                  <p className="text-sm text-slate-500 mt-0.5 line-clamp-1">{d.summary}</p>
                </div>
                <motion.div animate={{ rotate: isOpen ? 180 : 0 }} transition={{ duration: 0.2 }}>
                  <ChevronDown className="w-5 h-5 text-slate-400" />
                </motion.div>
              </button>

              <AnimatePresence>
                {isOpen && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.25 }}
                    className="overflow-hidden"
                  >
                    <div className="px-5 pb-6 pt-1 grid sm:grid-cols-3 gap-5 border-t border-slate-100">
                      <InfoColumn
                        icon={Stethoscope}
                        title="Symptoms"
                        items={d.symptoms}
                        accent={c.text}
                      />
                      <InfoColumn icon={ShieldAlert} title="Prevention" items={d.prevention} accent={c.text} />
                      <InfoColumn icon={ShieldCheck} title="Management" items={d.management} accent={c.text} />
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}

function InfoColumn({ icon: Icon, title, items, accent }) {
  return (
    <div className="pt-4">
      <h4 className={`text-xs font-semibold uppercase tracking-wide mb-2.5 flex items-center gap-1.5 ${accent}`}>
        <Icon className="w-3.5 h-3.5" /> {title}
      </h4>
      <ul className="space-y-1.5">
        {items?.map((item) => (
          <li key={item} className="text-sm text-slate-500 leading-relaxed">
            • {item}
          </li>
        ))}
      </ul>
    </div>
  );
}
