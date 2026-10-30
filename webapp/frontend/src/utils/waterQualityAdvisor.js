/**
 * waterQualityAdvisor.js
 * ──────────────────────
 * Given pH and temperature readings, returns:
 *  - status (normal / warning / danger)
 *  - what is happening to the fish right now
 *  - step-by-step recommendations to fix each issue
 *  - likely/upcoming diseases if the condition persists
 */

// ─── Optimal ranges for freshwater fish (tilapia / carp / catfish) ────────────
export const OPTIMAL = {
  ph:   { min: 6.5, max: 8.5 },
  temp: { min: 22,  max: 30  },
};

// ─── pH advisory table ────────────────────────────────────────────────────────
function phAdvice(ph) {
  if (ph === null || ph === undefined) return null;

  if (ph < 4.0) {
    return {
      level: "danger",
      label: "Critically Acidic",
      fishImpact: [
        "Fish will experience acute acid shock — mass mortality is imminent.",
        "Gills suffer severe tissue damage; fish gasp at the surface.",
        "Blood pH drops rapidly, causing organ failure within hours.",
      ],
      recommendations: [
        "Immediately add agricultural lime (CaCO₃) at 200–300 kg/ha to raise pH quickly.",
        "Perform a 50–70% emergency water exchange with neutral water.",
        "Stop all feeding — undigested feed worsens acidification.",
        "Increase aeration to help fish survive while pH is corrected.",
        "Retest pH every 30 minutes until it reaches at least 6.0.",
      ],
      diseases: ["EUS (Epizootic Ulcerative Syndrome)", "Columnaris", "Bacterial Gill Disease"],
    };
  }

  if (ph < 6.0) {
    return {
      level: "danger",
      label: "Severely Acidic",
      fishImpact: [
        "Fish immune systems are severely suppressed.",
        "Gill function is impaired — fish show labored breathing.",
        "Reproduction halts; juveniles may die within 24–48 hours.",
        "Harmful heavy metals (aluminium, manganese) become soluble and toxic.",
      ],
      recommendations: [
        "Apply agricultural lime (CaCO₃) at 100–200 kg/ha across the pond.",
        "Perform a 30–40% water exchange with clean, neutral-pH water.",
        "Reduce stocking density temporarily to lower metabolic load.",
        "Check and treat any organic waste or decaying material in the pond.",
        "Retest pH every hour and target 6.5 before resuming normal feeding.",
      ],
      diseases: ["EUS", "Gill Disease", "Red Spot Disease", "Bacterial infections"],
    };
  }

  if (ph < 6.5) {
    return {
      level: "warning",
      label: "Mildly Acidic",
      fishImpact: [
        "Fish are mildly stressed — immune response is slightly reduced.",
        "Growth rate slows as fish expend energy managing pH stress.",
        "Risk of opportunistic infections increases.",
      ],
      recommendations: [
        "Add agricultural lime (CaCO₃) at 50–100 kg/ha gradually.",
        "Perform a 20% water exchange to dilute acidity.",
        "Increase aeration to promote natural CO₂ off-gassing (raises pH).",
        "Check for organic waste accumulation (dead fish, uneaten feed).",
      ],
      diseases: ["EUS (early stage)", "Gill Disease", "Bacterial skin infections"],
    };
  }

  if (ph <= 8.5) {
    return {
      level: "normal",
      label: "Optimal",
      fishImpact: [
        "pH is within the ideal range for most freshwater fish.",
        "Fish immune systems are functioning normally.",
        "Growth, reproduction, and feeding are not affected by pH.",
      ],
      recommendations: [
        "Maintain current water management practices.",
        "Monitor pH daily — natural photosynthesis can cause pH swings.",
        "Avoid overfeeding which can cause organic acid buildup.",
      ],
      diseases: [],
    };
  }

  if (ph <= 9.0) {
    return {
      level: "warning",
      label: "Mildly Alkaline",
      fishImpact: [
        "Fish are mildly stressed — gill and skin mucus membranes are irritated.",
        "High pH increases toxicity of ammonia (NH₃ form is more toxic above pH 8.5).",
        "Feeding may decrease slightly.",
      ],
      recommendations: [
        "Reduce algae overgrowth (phytoplankton photosynthesis raises pH).",
        "Add clean freshwater to dilute alkalinity.",
        "Reduce fertiliser or organic inputs that fuel algae growth.",
        "Increase water exchange rate to 10–15% daily.",
        "Test ammonia levels — treat if elevated (use zeolite or water exchange).",
      ],
      diseases: ["Ammonia toxicity", "Gill Disease", "Bacterial infections"],
    };
  }

  if (ph <= 10.0) {
    return {
      level: "danger",
      label: "Severely Alkaline",
      fishImpact: [
        "Highly alkaline water causes chemical burns on gills and skin.",
        "Ammonia toxicity is dramatically increased at this pH.",
        "Fish show erratic swimming, excess mucus production, and surface gasping.",
        "Internal organ damage may occur with prolonged exposure.",
      ],
      recommendations: [
        "Perform immediate 40–50% water exchange with neutral water.",
        "Control algae bloom — use shading nets or reduce fertiliser.",
        "Add CO₂ injection or citric acid (small amounts) to lower pH gradually.",
        "Remove any heavy algae mats from the pond surface.",
        "Test and treat ammonia as a priority — it is extremely toxic at this pH.",
      ],
      diseases: ["Ammonia Toxicity", "Gill Disease", "EUS", "Bacterial Ulcer Disease"],
    };
  }

  return {
    level: "danger",
    label: "Critically Alkaline",
    fishImpact: [
      "pH above 10 is lethal — fish will die within hours without intervention.",
      "All gill tissue is chemically burned; respiration fails.",
      "Ammonia is almost entirely in its most toxic (un-ionised) form.",
    ],
    recommendations: [
      "Emergency 60–70% water exchange immediately.",
      "Remove all organic inputs and halt feeding entirely.",
      "Add CO₂ or buffered acid solution under expert guidance.",
      "Isolate and observe fish for acute distress — prepare for losses.",
    ],
    diseases: ["Acute Ammonia Toxicity", "Gill Necrosis", "Bacterial Septicaemia"],
  };
}

// ─── Temperature advisory table ───────────────────────────────────────────────
function tempAdvice(temp) {
  if (temp === null || temp === undefined) return null;

  if (temp < 10) {
    return {
      level: "danger",
      label: "Critically Cold",
      fishImpact: [
        "Fish are in cold-shock — metabolism has nearly stopped.",
        "Immune system is severely suppressed; fish cannot fight infections.",
        "Feeding stops entirely; growth halts.",
        "Mortality risk is high for tropical species.",
      ],
      recommendations: [
        "Use water heaters or solar pond covers to raise temperature gradually (max 1°C/hour).",
        "Stop feeding entirely until temperature reaches 18°C.",
        "Move fish to a warmer, sheltered tank/pond if possible.",
        "Increase pond depth to reduce surface cooling.",
        "Avoid any handling of fish in this condition — stress can be fatal.",
      ],
      diseases: ["Saprolegnia (fungal infection)", "Bacterial Cold-Water Disease", "Columnaris"],
    };
  }

  if (temp < 18) {
    return {
      level: "warning",
      label: "Too Cold",
      fishImpact: [
        "Fish metabolism is significantly slowed.",
        "Immune response is weakened — fish are more susceptible to fungi and bacteria.",
        "Growth rate is reduced by 30–50%.",
        "Appetite decreases; feed conversion efficiency drops.",
      ],
      recommendations: [
        "Raise temperature gradually using heaters or greenhouse pond covers.",
        "Reduce feeding frequency and amount — fish cannot digest food efficiently.",
        "Maintain aeration — cold water holds more oxygen but low metabolism still needs it.",
        "Watch for fungal infections (white cotton-like patches) and treat early.",
      ],
      diseases: ["Saprolegniasis (fungal)", "Bacterial Gill Disease", "Ich (White Spot)"],
    };
  }

  if (temp < 22) {
    return {
      level: "warning",
      label: "Slightly Cool",
      fishImpact: [
        "Fish are slightly below optimal temperature — mild stress.",
        "Growth is moderately reduced.",
        "Some reduction in feeding activity.",
      ],
      recommendations: [
        "If possible, use shallow black-bottom ponds to absorb solar heat.",
        "Reduce feeding by 20% to avoid uneaten feed accumulating.",
        "Monitor closely for fungal patches on skin or fins.",
      ],
      diseases: ["Saprolegniasis (early risk)", "Ich"],
    };
  }

  if (temp <= 30) {
    return {
      level: "normal",
      label: "Optimal",
      fishImpact: [
        "Temperature is ideal for freshwater fish growth and health.",
        "Immune system is fully active.",
        "Feeding, growth, and reproduction are all at their best.",
      ],
      recommendations: [
        "Maintain current conditions.",
        "Continue regular aeration to keep dissolved oxygen levels high.",
        "Monitor temperature during hot afternoons as ponds can overheat.",
      ],
      diseases: [],
    };
  }

  if (temp <= 34) {
    return {
      level: "warning",
      label: "Too Warm",
      fishImpact: [
        "Fish are experiencing heat stress — oxygen demand increases.",
        "Dissolved oxygen drops sharply at high temperatures.",
        "Bacterial growth in the water accelerates significantly.",
        "Fish may gasp at the surface (oxygen depletion sign).",
      ],
      recommendations: [
        "Increase aeration immediately — use additional aerators or paddlewheels.",
        "Perform 20–30% water exchange with cooler water.",
        "Add shading nets over 30–40% of the pond surface.",
        "Reduce feeding by 30–50% to lower metabolic oxygen demand.",
        "Avoid any fish handling during peak heat hours.",
        "Test dissolved oxygen — if below 5 mg/L, treat as emergency.",
      ],
      diseases: ["Bacterial Hemorrhagic Septicaemia", "Red Spot Disease", "Columnaris", "Gill Disease"],
    };
  }

  return {
    level: "danger",
    label: "Critically Hot",
    fishImpact: [
      "Temperature is at lethal threshold — fish will die without immediate action.",
      "Oxygen levels are critically low; fish suffocate.",
      "Bacterial and algae bloom conditions are extreme.",
      "Mass mortality is likely within hours.",
    ],
    recommendations: [
      "Emergency water exchange — pump in 40–50% cooler water immediately.",
      "Deploy all available aerators at maximum capacity.",
      "Install shade netting over entire pond surface.",
      "Stop feeding entirely.",
      "Move fish to cooler tanks if exchange water is not available.",
    ],
    diseases: ["Bacterial Septicaemia", "Red Spot Disease", "Acute Oxygen Depletion", "Columnaris"],
  };
}

// ─── Combined advisor ─────────────────────────────────────────────────────────
export function getWaterQualityAdvice(ph, temp) {
  const phResult   = phAdvice(ph);
  const tempResult = tempAdvice(temp);

  // Overall severity: danger > warning > normal
  const levels = ["normal", "warning", "danger"];
  const overallLevel = levels[
    Math.max(
      levels.indexOf(phResult?.level   || "normal"),
      levels.indexOf(tempResult?.level || "normal"),
    )
  ];

  // Deduplicated disease risk list
  const allDiseases = [
    ...(phResult?.diseases   || []),
    ...(tempResult?.diseases || []),
  ];
  const diseaseRisks = [...new Set(allDiseases)];

  return {
    overallLevel,
    ph:   phResult,
    temp: tempResult,
    diseaseRisks,
    hasIssues: overallLevel !== "normal",
  };
}
