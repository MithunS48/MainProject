// Disease information shown to farmers. Descriptions are educational
// summaries; the base name/description strings match the ones already
// defined in the ML backend (src/predict.py CLASS_LABELS / CLASS_DESCRIPTIONS).
// This is presented as AI-assisted information only — NOT a veterinary
// diagnosis. See the disclaimer shown throughout the app.

export const DISEASE_INFO = {
  EUS: {
    key: "EUS",
    name: "EUS (Epizootic Ulcerative Syndrome)",
    shortName: "EUS",
    color: "coral",
    isHealthy: false,
    summary:
      "A serious fungal (oomycete) disease that causes deep, red-bordered ulcers on the fish's body, often following stress, injury or poor water quality.",
    symptoms: [
      "Deep ulcers or lesions on the body, often red or brownish at the edges",
      "Reddish or grayish discoloration and skin erosion",
      "Loss of appetite and lethargy",
      "Erratic or slow swimming near the surface",
      "Fin and tail rot in advanced cases",
    ],
    prevention: [
      "Maintain good water quality (pH, dissolved oxygen, ammonia levels)",
      "Avoid overcrowding and physical injury during handling",
      "Quarantine new fish before introducing them to a pond/tank",
      "Remove and isolate visibly affected fish promptly",
    ],
    management: [
      "Improve water quality immediately (partial water change, aeration)",
      "Reduce stocking density and stress factors",
      "Consult a veterinary/aquaculture expert for confirmatory diagnosis and any treatment",
      "Avoid transferring water/equipment between affected and healthy ponds",
    ],
  },
  gill: {
    key: "gill",
    name: "Gill Disease",
    shortName: "Gill Disease",
    color: "sand",
    isHealthy: false,
    summary:
      "A condition affecting the gills — often caused by parasites, bacteria, or poor water quality — that impairs respiration and overall fish health.",
    symptoms: [
      "Rapid or labored gill movement (gasping at the surface)",
      "Pale, discolored, or mucus-covered gills",
      "Fish gathering near water inlets or aerators",
      "Reduced feeding and slower growth",
      "Increased susceptibility to secondary infections",
    ],
    prevention: [
      "Maintain adequate dissolved oxygen and clean water",
      "Regularly monitor ammonia and nitrite levels",
      "Avoid sudden temperature or pH fluctuations",
      "Practice routine health checks on stock",
    ],
    management: [
      "Increase aeration and water exchange",
      "Reduce feeding temporarily to lower metabolic/oxygen demand",
      "Seek professional aquaculture/veterinary advice for parasite or bacterial treatment",
      "Isolate severely affected fish where possible",
    ],
  },
  healthy: {
    key: "healthy",
    name: "Healthy",
    shortName: "Healthy",
    color: "seaweed",
    isHealthy: true,
    summary:
      "No visible signs of disease were detected in this image. The fish appears healthy based on the AI model's analysis.",
    symptoms: [
      "Clear, undamaged skin with normal coloration",
      "Bright, alert eyes",
      "Smooth, intact fins with no fraying",
      "Active, balanced swimming behavior",
      "Normal appetite and feeding response",
    ],
    prevention: [
      "Continue good water quality management",
      "Maintain balanced feeding and stocking density",
      "Perform regular visual health checks",
      "Keep records of water parameters over time",
    ],
    management: [
      "Continue routine monitoring practices",
      "Re-check periodically, especially after environmental changes",
      "Maintain quarantine procedures for any new stock",
    ],
  },
  red_spot: {
    key: "red_spot",
    name: "Red Spot Disease",
    shortName: "Red Spot",
    color: "coral",
    isHealthy: false,
    summary:
      "A bacterial infection that causes hemorrhagic (red) spots and lesions on the skin, fins, and sometimes internal organs of the fish.",
    symptoms: [
      "Small to large red or hemorrhagic spots on the body and fins",
      "Ulceration or open sores in severe cases",
      "Reddening at the base of fins",
      "Swelling or fluid accumulation in some cases",
      "Lethargy and reduced feeding",
    ],
    prevention: [
      "Maintain clean water and reduce organic waste buildup",
      "Avoid overstocking and minimize handling stress",
      "Disinfect equipment shared between ponds/tanks",
      "Quarantine and inspect new stock before introduction",
    ],
    management: [
      "Improve water quality and reduce stressors immediately",
      "Isolate affected fish to limit potential spread",
      "Consult a veterinary/aquaculture professional for confirmatory testing and appropriate treatment",
      "Monitor the rest of the stock closely for similar symptoms",
    ],
  },
};

export const DISEASE_ORDER = ["EUS", "gill", "healthy", "red_spot"];

export function getDiseaseInfo(key) {
  return DISEASE_INFO[key] || DISEASE_INFO.healthy;
}
