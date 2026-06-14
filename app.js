/* ============================================================
   AquaScan — Complete app.js
   All features: auth, per-user history, admin panel,
   analyze, batch, camera, gradcam, chatbot, PDF, charts, i18n
   ============================================================ */

const API_BASE_URL = "http://localhost:8000";
const MAX_FILE_SIZE = 10 * 1024 * 1024;
const AUTH_KEY = "aquascan_auth";
const CONF_WARN = 0.60;
const DEFAULT_API_KEY = "aquascan-dev-key";

/* ── Auth ── */
const Auth = {
  get()        { try { return JSON.parse(localStorage.getItem(AUTH_KEY)) || null; } catch { return null; } },
  set(d)       { localStorage.setItem(AUTH_KEY, JSON.stringify(d)); },
  clear()      { localStorage.removeItem(AUTH_KEY); },
  isLoggedIn() { return !!this.get(); },
  token()      { return this.get()?.access_token || null; },
  username()   { return this.get()?.username || "guest"; },
  role()       { return this.get()?.role || null; }
};

/* ── Authenticated fetch ── */
async function apiFetch(path, opts = {}) {
  const headers = {
    "X-API-Key": DEFAULT_API_KEY,
    ...(Auth.token() ? { "Authorization": "Bearer " + Auth.token() } : {}),
    ...(opts.headers || {})
  };
  if (opts.body instanceof FormData) delete headers["Content-Type"];
  const res = await fetch(API_BASE_URL + path, { ...opts, headers });
  if (res.status === 401) { Auth.clear(); showLoginModal("Session expired. Please log in again."); throw new Error("Unauthorized"); }
  return res;
}

/* ── Per-user history (localStorage keyed by username) ── */
function histKey() { return "aquascan_h_" + Auth.username(); }
function loadHistory() { try { return JSON.parse(localStorage.getItem(histKey())) || []; } catch { return []; } }
function saveHistory(h) { localStorage.setItem(histKey(), JSON.stringify(h)); }
function addHistoryEntry(e) { const h = loadHistory(); h.unshift(e); saveHistory(h); }
function clearHistory() { localStorage.removeItem(histKey()); }

/* ── i18n ── */
let currentLang = "en";
const TRANS = {
  en: { nav_dashboard:"Dashboard", nav_analyze:"Analyze", nav_batch:"Batch Scan", nav_history:"History", nav_reports:"Reports", nav_admin:"Admin Panel", dash_title:"Fish Health Monitor", dash_sub:"AI-powered fish disease diagnostics.", dash_cta:"+ New Analysis", stat_total:"Total Scans", stat_healthy:"Healthy", stat_diseased:"Diseased", stat_today:"Today", label_healthy:"Healthy", label_eus:"EUS (Epizootic Ulcerative Syndrome)", label_gill:"Bacterial Gill Disease", label_red_spot:"Bacterial Red Spot Disease", sev_none:"None", sev_mild:"Mild", sev_moderate:"Moderate", sev_severe:"Severe", badge_healthy:"✅ No Disease Detected", badge_disease:"⚠️ Disease Detected", confidence:"Confidence", severity:"Severity:", treatment_title:"Recommended Treatment", treatment_warn:"⚠️ Consult a licensed veterinarian before treatment.", gradcam_title:"AI Focus Heatmap (Grad-CAM)", gradcam_desc:"Red/yellow areas show where the AI focused.", all_probs:"All Class Probabilities", low_conf_title:"Low Confidence Warning", low_conf_msg:"Model confidence below 60%. Try a clearer photo.", analyzing:"Analyzing…", analyze_now:"Analyze Now", new_analysis:"+ New Analysis", save_pdf:"⬇ Download PDF", error_title:"Something went wrong", try_again:"Try Again", no_scans:"No scans yet.", health_rate:"Health Rate", most_common:"Most Common Disease", first_scan:"First Scan", export_csv:"Export CSV" },
  kn: { nav_dashboard:"ಡ್ಯಾಶ್‌ಬೋರ್ಡ್", nav_analyze:"ವಿಶ್ಲೇಷಣೆ", nav_batch:"ಬ್ಯಾಚ್ ಸ್ಕ್ಯಾನ್", nav_history:"ಇತಿಹಾಸ", nav_reports:"ವರದಿಗಳು", nav_admin:"ಆಡಳಿತ ಫಲಕ", dash_title:"ಮೀನಿನ ಆರೋಗ್ಯ ಮಾನಿಟರ್", dash_sub:"AI ಆಧಾರಿತ ಮೀನು ರೋಗ ರೋಗನಿರ್ಣಯ.", dash_cta:"+ ಹೊಸ ವಿಶ್ಲೇಷಣೆ", stat_total:"ಒಟ್ಟು ಸ್ಕ್ಯಾನ್‌ಗಳು", stat_healthy:"ಆರೋಗ್ಯಕರ", stat_diseased:"ರೋಗಪೀಡಿತ", stat_today:"ಇಂದು", label_healthy:"ಆರೋಗ್ಯಕರ", label_eus:"EUS (ಎಪಿಝೂಟಿಕ್ ಅಲ್ಸರೇಟಿವ್ ಸಿಂಡ್ರೋಮ್)", label_gill:"ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಗಿಲ್ ರೋಗ", label_red_spot:"ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಕೆಂಪು ಚುಕ್ಕೆ ರೋಗ", sev_none:"ಇಲ್ಲ", sev_mild:"ಸೌಮ್ಯ", sev_moderate:"ಮಧ್ಯಮ", sev_severe:"ತೀವ್ರ", badge_healthy:"✅ ರೋಗ ಪತ್ತೆಯಾಗಿಲ್ಲ", badge_disease:"⚠️ ರೋಗ ಪತ್ತೆಯಾಗಿದೆ", confidence:"ವಿಶ್ವಾಸ", severity:"ತೀವ್ರತೆ:", treatment_title:"ಶಿಫಾರಸು ಚಿಕಿತ್ಸೆ", treatment_warn:"⚠️ ಚಿಕಿತ್ಸೆ ಮೊದಲು ಮೀನು ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ.", gradcam_title:"AI ಗಮನ ಹೀಟ್‌ಮ್ಯಾಪ್", gradcam_desc:"ಕೆಂಪು/ಹಳದಿ ಪ್ರದೇಶಗಳು AI ಗಮನಿಸಿದ ಜಾಗಗಳನ್ನು ತೋರಿಸುತ್ತವೆ.", all_probs:"ಎಲ್ಲಾ ವರ್ಗ ಸಂಭಾವ್ಯತೆಗಳು", low_conf_title:"ಕಡಿಮೆ ವಿಶ್ವಾಸ ಎಚ್ಚರಿಕೆ", low_conf_msg:"ಮಾದರಿ 60% ಕ್ಕಿಂತ ಕಡಿಮೆ ವಿಶ್ವಾಸ. ಸ್ಪಷ್ಟ ಫೋಟೋ ಪ್ರಯತ್ನಿಸಿ.", analyzing:"ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ…", analyze_now:"ಈಗ ವಿಶ್ಲೇಷಿಸಿ", new_analysis:"+ ಹೊಸ ವಿಶ್ಲೇಷಣೆ", save_pdf:"⬇ PDF ಡೌನ್‌ಲೋಡ್", error_title:"ಏನೋ ತಪ್ಪಾಗಿದೆ", try_again:"ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ", no_scans:"ಇನ್ನೂ ಸ್ಕ್ಯಾನ್ ಇಲ್ಲ.", health_rate:"ಆರೋಗ್ಯ ದರ", most_common:"ಸಾಮಾನ್ಯ ರೋಗ", first_scan:"ಮೊದಲ ಸ್ಕ್ಯಾನ್", export_csv:"CSV ರಫ್ತು" },
  te: { nav_dashboard:"డాష్‌బోర్డ్", nav_analyze:"విశ్లేషణ", nav_batch:"బ్యాచ్ స్కాన్", nav_history:"చరిత్ర", nav_reports:"నివేదికలు", nav_admin:"అడ్మిన్ ప్యానెల్", dash_title:"చేప ఆరోగ్య మానిటర్", dash_sub:"AI ఆధారిత చేప వ్యాధి నిర్ధారణ.", dash_cta:"+ కొత్త విశ్లేషణ", stat_total:"మొత్తం స్కాన్‌లు", stat_healthy:"ఆరోగ్యకరమైన", stat_diseased:"వ్యాధిగ్రస్తమైన", stat_today:"ఈరోజు", label_healthy:"ఆరోగ్యకరమైన", label_eus:"EUS (ఎపిజూటిక్ అల్సరేటివ్ సిండ్రోమ్)", label_gill:"బాక్టీరియల్ గిల్ వ్యాధి", label_red_spot:"బాక్టీరియల్ ఎర్ర మచ్చ వ్యాధి", sev_none:"లేదు", sev_mild:"తేలిక", sev_moderate:"మధ్యస్థ", sev_severe:"తీవ్రమైన", badge_healthy:"✅ వ్యాధి గుర్తించబడలేదు", badge_disease:"⚠️ వ్యాధి గుర్తించబడింది", confidence:"విశ్వాసం", severity:"తీవ్రత:", treatment_title:"సిఫార్సు చికిత్స", treatment_warn:"⚠️ చికిత్స ముందు వైద్యుడిని సంప్రదించండి.", gradcam_title:"AI దృష్టి హీట్‌మ్యాప్", gradcam_desc:"ఎరుపు/పసుపు ప్రాంతాలు AI దృష్టి పెట్టిన చోటు.", all_probs:"అన్ని తరగతి సంభావ్యతలు", low_conf_title:"తక్కువ విశ్వాసం హెచ్చరిక", low_conf_msg:"మోడల్ 60% కంటే తక్కువ విశ్వాసం. స్పష్టమైన ఫోటో ప్రయత్నించండి.", analyzing:"విశ్లేషిస్తోంది…", analyze_now:"ఇప్పుడు విశ్లేషించండి", new_analysis:"+ కొత్త విశ్లేషణ", save_pdf:"⬇ PDF డౌన్‌లోడ్", error_title:"ఏదో తప్పు జరిగింది", try_again:"మళ్ళీ ప్రయత్నించండి", no_scans:"ఇంకా స్కాన్‌లు లేవు.", health_rate:"ఆరోగ్య రేటు", most_common:"సాధారణ వ్యాధి", first_scan:"మొదటి స్కాన్", export_csv:"CSV ఎగుమతి" }
};
function t(k) { return (TRANS[currentLang] && TRANS[currentLang][k]) || TRANS.en[k] || k; }
function applyI18n() { document.querySelectorAll("[data-i18n]").forEach(el => { el.textContent = t(el.dataset.i18n); }); }

/* ── Disease content ── */
const DISEASE_DB = {
  healthy:  { icon:"✅", description:"", treatment:[] },
  eus:      { icon:"🔴", description:"A fungal/bacterial disease causing ulcerative lesions on the fish body.", treatment:["Improve water quality and reduce organic load","Reduce stocking density to lower stress","Apply antifungal treatments (e.g., potassium permanganate bath at 2–4 mg/L)","Consult a veterinarian for antibiotic therapy if secondary bacterial infection is present"] },
  gill:     { icon:"🟠", description:"A bacterial infection affecting gill tissue, causing respiratory distress.", treatment:["Improve water quality and increase aeration","Reduce organic load and ammonia levels","Apply salt baths (0.5–1%) or potassium permanganate treatments","Administer antibiotics as prescribed by a veterinarian"] },
  red_spot: { icon:"🟡", description:"A bacterial infection causing red hemorrhagic spots on the fish skin.", treatment:["Isolate affected fish immediately to prevent spread","Improve water quality and reduce stress factors","Apply antibiotic baths (e.g., oxytetracycline at 10–20 mg/L)","Consult a veterinarian for systemic antibiotic treatment if infection is severe"] }
};
function getDiseaseContent(cls) {
  const d = DISEASE_DB[cls] || DISEASE_DB.healthy;
  return { ...d, label: t("label_" + (cls || "healthy")) };
}
function formatConfidence(c) { return (c * 100).toFixed(1) + "%"; }
function isHealthy(cls) { return cls === "healthy"; }

/* ── DOM helpers ── */
function show(id) { const el = typeof id === "string" ? document.getElementById(id) : id; if (el) el.hidden = false; }
function hide(id) { const el = typeof id === "string" ? document.getElementById(id) : id; if (el) el.hidden = true; }
function setText(id, v) { const el = document.getElementById(id); if (el) el.textContent = v; }

/* ── Toast ── */
function showToast(msg, type = "blue") {
  const el = document.createElement("div");
  el.className = `toast toast-${type}`; el.textContent = msg;
  document.body.appendChild(el);
  requestAnimationFrame(() => el.classList.add("toast-show"));
  setTimeout(() => { el.classList.remove("toast-show"); setTimeout(() => el.remove(), 400); }, 3500);
}

/* ── Login modal ── */
function showLoginModal(msg) {
  show("login-overlay");
  if (msg) { const m = document.getElementById("login-modal-msg"); if (m) { m.textContent = msg; show(m); } }
}
function hideLoginModal() {
  hide("login-overlay");
  const m = document.getElementById("login-modal-msg"); if (m) { m.textContent = ""; hide(m); }
}
window.switchTab = function(tab) {
  const lf = document.getElementById("login-form"), sf = document.getElementById("signup-form");
  const tl = document.getElementById("tab-login"),  ts = document.getElementById("tab-signup");
  if (tab === "login") { lf.hidden = false; sf.hidden = true; tl.classList.add("active"); ts.classList.remove("active"); }
  else                 { lf.hidden = true; sf.hidden = false; ts.classList.add("active"); tl.classList.remove("active"); }
  ["login-error","signup-error","login-modal-msg"].forEach(id => { const e = document.getElementById(id); if (e) e.hidden = true; });
};

function updateAuthUI() {
  const loggedIn = Auth.isLoggedIn();
  const ui = document.getElementById("sidebar-user-info");
  const lb = document.getElementById("login-nav-btn");
  const lo = document.getElementById("logout-nav-btn");
  const adminNav = document.getElementById("admin-nav-item");
  if (ui) ui.textContent = loggedIn ? "👤 " + Auth.username() : "👤 Not logged in";
  if (lb) lb.hidden = loggedIn; if (lo) lo.hidden = !loggedIn;
  if (adminNav) adminNav.hidden = Auth.role() !== "admin";
  const mw = document.querySelector(".main-wrapper");
  const ov = document.getElementById("login-overlay");
  if (loggedIn) { if (mw) mw.style.filter = ""; if (ov) ov.hidden = true; }
  else          { if (mw) mw.style.filter = "blur(4px) brightness(0.6)"; if (ov) ov.hidden = false; }
}

/* ── Charts ── */
const charts = {};
function destroyChart(k) { if (charts[k]) { charts[k].destroy(); charts[k] = null; } }
const CC = { healthy:"#16a34a", eus:"#dc2626", gill:"#d97706", red_spot:"#7c3aed", primary:"#6366f1", cyan:"#06b6d4" };
function lastNDays(n) {
  const days = [];
  for (let i = n-1; i >= 0; i--) { const d = new Date(); d.setDate(d.getDate()-i); days.push(d.toLocaleDateString("en-GB",{month:"short",day:"numeric"})); }
  return days;
}
function aggregateByDay(hist, days) {
  const map = {}; days.forEach(d => { map[d] = {healthy:0,total:0}; });
  hist.forEach(e => { const lbl = new Date(e.timestamp * 1000 || e.timestamp).toLocaleDateString("en-GB",{month:"short",day:"numeric"}); if (map[lbl]) { map[lbl].total++; if (isHealthy(e.predicted_class)) map[lbl].healthy++; } });
  return map;
}
function classCounts(hist) {
  const c = {healthy:0,eus:0,gill:0,red_spot:0};
  hist.forEach(e => { if (c[e.predicted_class] !== undefined) c[e.predicted_class]++; });
  return c;
}
function buildTrendChart(canvasId, hist, nDays, key) {
  destroyChart(key); const ctx = document.getElementById(canvasId); if (!ctx) return;
  const days = lastNDays(nDays), byDay = aggregateByDay(hist, days);
  charts[key] = new Chart(ctx, { data: { labels: days, datasets: [
    { type:"line", label:"% Healthy", data: days.map(d => byDay[d].total > 0 ? Math.round((byDay[d].healthy/byDay[d].total)*100) : null), borderColor:CC.healthy, backgroundColor:"rgba(22,163,74,0.08)", borderWidth:2.5, tension:0.4, fill:true, yAxisID:"yP", spanGaps:true, pointRadius:3 },
    { type:"bar", label:"Total Scans", data: days.map(d => byDay[d].total), backgroundColor:"rgba(99,102,241,0.18)", borderColor:"rgba(99,102,241,0.4)", borderWidth:1, borderRadius:4, yAxisID:"yC" }
  ]}, options:{ responsive:true, maintainAspectRatio:false, interaction:{mode:"index",intersect:false}, plugins:{legend:{position:"top",labels:{font:{size:11},boxWidth:12}}}, scales:{ yP:{type:"linear",position:"left",min:0,max:100,ticks:{callback:v=>v+"%",font:{size:11}},grid:{color:"rgba(0,0,0,0.05)"}}, yC:{type:"linear",position:"right",min:0,ticks:{stepSize:1,font:{size:11}},grid:{drawOnChartArea:false}}, x:{ticks:{font:{size:10},maxTicksLimit:10},grid:{display:false}} } } });
}
function buildPieChart(canvasId, hist, key) {
  destroyChart(key); const ctx = document.getElementById(canvasId); if (!ctx) return;
  const c = classCounts(hist), total = Object.values(c).reduce((a,b)=>a+b,0);
  charts[key] = new Chart(ctx, { type:"doughnut", data:{ labels:["Healthy","EUS","Bacterial Gill","Red Spot"], datasets:[{data:[c.healthy,c.eus,c.gill,c.red_spot], backgroundColor:[CC.healthy,CC.eus,CC.gill,CC.red_spot], borderWidth:2, borderColor:"#fff"}]}, options:{ responsive:true, maintainAspectRatio:false, cutout:"58%", plugins:{ legend:{position:"bottom",labels:{font:{size:11},boxWidth:12}}, tooltip:{callbacks:{label:cx=>{ const pct=total>0?((cx.parsed/total)*100).toFixed(1):0; return ` ${cx.label}: ${cx.parsed} (${pct}%)`; }}} } } });
}
function buildBarChart(canvasId, hist, key) {
  destroyChart(key); const ctx = document.getElementById(canvasId); if (!ctx) return;
  const c = classCounts(hist);
  charts[key] = new Chart(ctx, { type:"bar", data:{ labels:["Healthy","EUS","Bacterial Gill","Red Spot"], datasets:[{label:"Scans", data:[c.healthy,c.eus,c.gill,c.red_spot], backgroundColor:["rgba(22,163,74,0.75)","rgba(220,38,38,0.75)","rgba(217,119,6,0.75)","rgba(124,58,237,0.75)"], borderColor:[CC.healthy,CC.eus,CC.gill,CC.red_spot], borderWidth:2, borderRadius:6}]}, options:{ responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}}, scales:{ y:{min:0,beginAtZero:true,ticks:{stepSize:1,font:{size:11}},grid:{color:"rgba(0,0,0,0.05)"}}, x:{ticks:{font:{size:11}},grid:{display:false}} } } });
}

/* ── Dashboard ── */
function populateDashboard() {
  const hist = loadHistory();
  const today = new Date().toDateString();
  setText("stat-total",    hist.length);
  setText("stat-healthy",  hist.filter(e=>isHealthy(e.predicted_class)).length);
  setText("stat-diseased", hist.filter(e=>!isHealthy(e.predicted_class)).length);
  setText("stat-today",    hist.filter(e=>new Date((e.timestamp*1000||e.timestamp)).toDateString()===today).length);
  const tbody = document.getElementById("recent-tbody");
  if (tbody) {
    if (!hist.length) { tbody.innerHTML = `<tr><td colspan="6" class="empty-row"><a href="#" data-nav="analyze">${t("no_scans")} Run your first analysis →</a></td></tr>`; tbody.querySelectorAll("[data-nav]").forEach(wireNavLink); }
    else { tbody.innerHTML = hist.slice(0,5).map((e,i) => { const c=getDiseaseContent(e.predicted_class),h=isHealthy(e.predicted_class); return `<tr><td>${i+1}</td><td>${new Date((e.timestamp*1000||e.timestamp)).toLocaleString()}</td><td>${c.label}</td><td>${formatConfidence(e.confidence)}</td><td><span class="severity-badge ${e.severity||"none"}">${t("sev_"+(e.severity||"none"))}</span></td><td><span class="badge ${h?"badge-green":"badge-red"}">${h?"✅":"⚠️"}</span></td></tr>`; }).join(""); }
  }
  buildTrendChart("trendChart",hist,14,"trend");
  buildPieChart("pieChart",hist,"pie");
  buildBarChart("barChart",hist,"bar");
}

/* ── History ── */
async function populateHistory(filter="all", search="") {
  // Try to sync from server
  try {
    const res = await apiFetch("/scans/mine");
    if (res.ok) { const data = await res.json(); saveHistory(data.scans || []); }
  } catch { /* use local cache */ }
  const hist = loadHistory();
  const timeline = document.getElementById("history-timeline");
  const emptyEl  = document.getElementById("history-empty");
  if (!timeline) return;
  timeline.querySelectorAll(".history-item").forEach(el => el.remove());
  const filtered = hist.filter(e => filter==="all" || e.predicted_class===filter).filter(e => !search || getDiseaseContent(e.predicted_class).label.toLowerCase().includes(search.toLowerCase()));
  if (!filtered.length) { emptyEl && (emptyEl.hidden=false); return; }
  emptyEl && (emptyEl.hidden=true);
  filtered.forEach(e => {
    const c=getDiseaseContent(e.predicted_class), h=isHealthy(e.predicted_class);
    const item = document.createElement("div"); item.className=`history-item ${h?"healthy":"disease"}`;
    item.innerHTML = `<span class="history-icon">${c.icon}</span><div class="history-meta"><div class="history-label">${c.label}</div><div class="history-date">${new Date((e.timestamp*1000||e.timestamp)).toLocaleString()}</div></div><span class="severity-badge ${e.severity||"none"}">${t("sev_"+(e.severity||"none"))}</span><span class="history-conf">${formatConfidence(e.confidence)}</span><span class="badge ${h?"badge-green":"badge-red"}">${h?"✅":"⚠️"}</span>`;
    timeline.appendChild(item);
  });
}

/* ── Reports ── */
function populateReports() {
  const hist = loadHistory();
  setText("rpt-total", hist.length);
  if (!hist.length) { setText("rpt-health-rate","—"); setText("rpt-most-common","—"); setText("rpt-first-scan","—"); }
  else {
    const hc = hist.filter(e=>isHealthy(e.predicted_class)).length;
    setText("rpt-health-rate", Math.round((hc/hist.length)*100)+"%");
    const dis = hist.filter(e=>!isHealthy(e.predicted_class));
    if (!dis.length) { setText("rpt-most-common","None"); }
    else { const freq={}; dis.forEach(e=>{freq[e.predicted_class]=(freq[e.predicted_class]||0)+1;}); const top=Object.entries(freq).sort((a,b)=>b[1]-a[1])[0][0]; setText("rpt-most-common",getDiseaseContent(top).label); }
    setText("rpt-first-scan", new Date((hist[hist.length-1].timestamp*1000||hist[hist.length-1].timestamp)).toLocaleDateString());
  }
  buildTrendChart("reportTrendChart",hist,30,"rTrend");
  buildPieChart("reportPieChart",hist,"rPie");
  destroyChart("rWeekly"); const wCtx=document.getElementById("reportWeeklyChart");
  if (wCtx) {
    const weeks=[],wC=[];
    for(let w=7;w>=0;w--){const s=new Date();s.setDate(s.getDate()-w*7);const e2=new Date();e2.setDate(e2.getDate()-(w-1)*7);weeks.push(`W${8-w}`);wC.push(hist.filter(e=>{const t2=new Date((e.timestamp*1000||e.timestamp));return t2>=s&&t2<e2;}).length);}
    charts.rWeekly=new Chart(wCtx,{type:"bar",data:{labels:weeks,datasets:[{label:"Scans",data:wC,backgroundColor:"rgba(99,102,241,0.7)",borderColor:CC.primary,borderWidth:2,borderRadius:6}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{min:0,beginAtZero:true,ticks:{stepSize:1,font:{size:11}}},x:{ticks:{font:{size:11}},grid:{display:false}}}}});
  }
  destroyChart("rConf"); const cCtx=document.getElementById("reportConfChart");
  if (cCtx) {
    const buckets=["<50%","50-60%","60-70%","70-80%","80-90%","90-100%"],bc=[0,0,0,0,0,0];
    hist.forEach(e=>{const p=(e.confidence||0)*100;if(p<50)bc[0]++;else if(p<60)bc[1]++;else if(p<70)bc[2]++;else if(p<80)bc[3]++;else if(p<90)bc[4]++;else bc[5]++;});
    charts.rConf=new Chart(cCtx,{type:"bar",data:{labels:buckets,datasets:[{label:"Scans",data:bc,backgroundColor:["rgba(148,163,184,0.6)","rgba(220,38,38,0.6)","rgba(217,119,6,0.6)","rgba(234,179,8,0.65)","rgba(34,197,94,0.65)","rgba(22,163,74,0.8)"],borderWidth:1,borderRadius:5}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{min:0,beginAtZero:true,ticks:{stepSize:1,font:{size:11}}},x:{ticks:{font:{size:11}},grid:{display:false}}}}});
  }
}

/* ── Admin Panel ── */
async function populateAdmin() {
  if (Auth.role() !== "admin") return;
  const container = document.getElementById("admin-users-list");
  const statsEl   = document.getElementById("admin-stats-row");
  if (!container) return;
  container.innerHTML = `<div style="padding:1rem;color:var(--text-muted)">Loading users…</div>`;
  try {
    const [sRes, uRes] = await Promise.all([apiFetch("/admin/stats"), apiFetch("/admin/users")]);
    if (sRes.ok && statsEl) {
      const s = await sRes.json();
      statsEl.innerHTML = `
        <div class="stat-card"><div class="stat-icon blue">👥</div><div class="stat-body"><div class="stat-value">${s.total_users}</div><div class="stat-label">Total Users</div></div></div>
        <div class="stat-card"><div class="stat-icon blue">🔬</div><div class="stat-body"><div class="stat-value">${s.total_scans}</div><div class="stat-label">Total Scans</div></div></div>
        <div class="stat-card"><div class="stat-icon green">✅</div><div class="stat-body"><div class="stat-value">${s.healthy}</div><div class="stat-label">Healthy</div></div></div>
        <div class="stat-card"><div class="stat-icon red">⚠️</div><div class="stat-body"><div class="stat-value">${s.total_scans - s.healthy}</div><div class="stat-label">Diseased</div></div></div>`;
    } else { await sRes.json(); } // consume
    if (!uRes.ok) { container.innerHTML = `<p style="color:var(--red);padding:1rem">Failed to load users.</p>`; return; }
    const data = await uRes.json();
    container.innerHTML = data.users.map(u => `
      <div class="admin-user-card">
        <div class="admin-user-header">
          <div class="admin-user-avatar">${u.role==="admin"?"👑":"👤"}</div>
          <div class="admin-user-info">
            <div class="admin-user-name">${u.full_name||u.username} <span class="badge ${u.role==="admin"?"badge-blue":"badge-green"}">${u.role}</span></div>
            <div class="admin-user-meta">@${u.username}${u.email?" · "+u.email:""} · Joined ${new Date(u.created_at*1000).toLocaleDateString()}</div>
          </div>
          <div class="admin-user-stats">
            <span class="admin-scan-count">${u.total_scans} scans</span>
            <span class="badge badge-green">${u.healthy} ✅</span>
            <span class="badge badge-red">${u.diseased} ⚠️</span>
          </div>
        </div>
        <div class="admin-user-actions">
          <button class="btn btn-sm btn-outline" onclick="viewUserScans('${u.username}')">📋 View Scans</button>
          ${u.username!=="admin"?`<button class="btn btn-sm btn-danger" onclick="deleteUser('${u.username}')">🗑 Delete</button>`:""}
        </div>
        <div class="admin-user-scans" id="scans-${u.username}" hidden></div>
      </div>`).join("");
  } catch(err) { container.innerHTML = `<p style="color:var(--red);padding:1rem">Error: ${err.message}</p>`; }
}

window.viewUserScans = async function(username) {
  const el = document.getElementById("scans-"+username); if (!el) return;
  if (!el.hidden) { el.hidden=true; return; }
  el.innerHTML = `<p style="color:var(--text-muted);font-size:0.82rem;padding:0.5rem">Loading…</p>`; el.hidden=false;
  try {
    const res = await apiFetch("/admin/users/"+username+"/scans");
    const data = await res.json();
    if (!res.ok) { el.innerHTML=`<p style="color:var(--red);padding:0.5rem">${data.detail}</p>`; return; }
    if (!data.scans.length) { el.innerHTML=`<p style="color:var(--text-muted);font-size:0.82rem;padding:0.5rem">No scans yet.</p>`; return; }
    el.innerHTML=`<table class="data-table" style="margin-top:0.5rem"><thead><tr><th>#</th><th>Date</th><th>Diagnosis</th><th>Confidence</th><th>Severity</th></tr></thead><tbody>${data.scans.map((s,i)=>{const c=getDiseaseContent(s.predicted_class);return `<tr><td>${i+1}</td><td>${new Date(s.timestamp*1000).toLocaleString()}</td><td>${c.label}</td><td>${formatConfidence(s.confidence)}</td><td><span class="severity-badge ${s.severity}">${s.severity}</span></td></tr>`;}).join("")}</tbody></table>`;
  } catch { el.innerHTML=`<p style="color:var(--red);padding:0.5rem">Failed to load scans.</p>`; }
};

window.deleteUser = async function(username) {
  if (!confirm(`Delete user @${username} and ALL their data?`)) return;
  try { const res=await apiFetch("/admin/users/"+username,{method:"DELETE"}); const data=await res.json(); if(res.ok){showToast("User @"+username+" deleted.","red");populateAdmin();}else{showToast(data.detail||"Failed.","red");} } catch{showToast("Network error.","red");}
};

/* ── Export CSV ── */
function exportCSV(data) {
  const hist = data || loadHistory(); if (!hist.length) { alert("No data to export."); return; }
  const header = ["#","Timestamp","Class","Label","Confidence (%)","Severity"];
  const rows = hist.map((e,i)=>[i+1,new Date((e.timestamp*1000||e.timestamp)).toLocaleString(),e.predicted_class,getDiseaseContent(e.predicted_class).label,(e.confidence*100).toFixed(1),e.severity||"none"]);
  const csv = [header,...rows].map(r=>r.map(v=>`"${v}"`).join(",")).join("\n");
  const a=document.createElement("a"); a.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"})); a.download=`aquascan_${new Date().toISOString().slice(0,10)}.csv`; a.click();
}

/* ── PDF Download with jsPDF ── */
let lastResultData = null;
function downloadPDF(data) {
  if (!data) return;
  if (typeof window.jspdf === "undefined") {
    const s=document.createElement("script"); s.src="https://cdn.jsdelivr.net/npm/jspdf@2.5.1/dist/jspdf.umd.min.js"; s.onload=()=>_genPDF(data); document.head.appendChild(s);
  } else { _genPDF(data); }
}
function _genPDF(data) {
  const {jsPDF}=window.jspdf, doc=new jsPDF({unit:"mm",format:"a4"}), c=getDiseaseContent(data.predicted_class), h=isHealthy(data.predicted_class), pw=doc.internal.pageSize.getWidth();
  let y=20;
  doc.setFillColor(30,27,75); doc.rect(0,0,pw,18,"F"); doc.setTextColor(255,255,255); doc.setFontSize(16); doc.setFont("helvetica","bold"); doc.text("AquaScan — Fish Health Report",14,12); doc.setFontSize(9); doc.setFont("helvetica","normal"); doc.text("Generated: "+new Date().toLocaleString(),pw-14,12,{align:"right"});
  y=30; doc.setTextColor(15,23,42);
  doc.setFillColor(h?220:254,h?252:226,h?231:226); doc.roundedRect(14,y,pw-28,28,3,3,"F"); doc.setFontSize(14); doc.setFont("helvetica","bold"); doc.setTextColor(h?22:220,h?163:38,h?74:38); doc.text((h?"✓ ":"⚠ ")+c.label,20,y+10); doc.setFontSize(10); doc.setFont("helvetica","normal"); doc.setTextColor(100,116,139); doc.text("Confidence: "+formatConfidence(data.confidence)+"   |   Severity: "+(data.severity||"none")+"   |   User: "+Auth.username(),20,y+20); y+=36;
  doc.setTextColor(15,23,42); doc.setFontSize(12); doc.setFont("helvetica","bold"); doc.text("Diagnosis Details",14,y); y+=6;
  [["File",data.filename||"—"],["Diagnosis",c.label],["Confidence",formatConfidence(data.confidence)],["Severity",data.severity||"none"],["Timestamp",new Date().toLocaleString()],["User",Auth.username()]].forEach(([lbl,val])=>{ doc.setFillColor(248,250,252); doc.rect(14,y,pw-28,8,"F"); doc.setFont("helvetica","bold"); doc.setFontSize(9); doc.setTextColor(100,116,139); doc.text(lbl,18,y+5.5); doc.setFont("helvetica","normal"); doc.setTextColor(15,23,42); doc.text(String(val),60,y+5.5); y+=9; });
  if (!h && c.description) { y+=6; doc.setFont("helvetica","bold"); doc.setFontSize(11); doc.text("About This Disease",14,y); y+=6; doc.setFont("helvetica","normal"); doc.setFontSize(9); doc.setTextColor(100,116,139); const dl=doc.splitTextToSize(c.description,pw-28); doc.text(dl,14,y); y+=dl.length*5+4; }
  if (!h && c.treatment.length) { y+=4; doc.setFillColor(255,251,235); doc.roundedRect(14,y,pw-28,12+c.treatment.length*8,3,3,"F"); doc.setFont("helvetica","bold"); doc.setFontSize(11); doc.setTextColor(146,64,14); doc.text("Recommended Treatment",20,y+8); y+=14; doc.setFont("helvetica","normal"); doc.setFontSize(9); doc.setTextColor(15,23,42); c.treatment.forEach((s,i)=>{doc.text(`${i+1}. ${s}`,22,y);y+=8;}); y+=4; doc.setTextColor(146,64,14); doc.setFontSize(8); doc.text("⚠ Always consult a licensed aquatic veterinarian before administering treatment.",14,y); }
  const fy=doc.internal.pageSize.getHeight()-10; doc.setDrawColor(226,232,240); doc.line(14,fy-4,pw-14,fy-4); doc.setFont("helvetica","normal"); doc.setFontSize(8); doc.setTextColor(148,163,184); doc.text("AquaScan © 2025 — For informational purposes only.",14,fy);
  doc.save(`AquaScan_Report_${new Date().toISOString().slice(0,10)}.pdf`);
  showToast("📄 PDF downloaded!","blue");
}

/* ── File validation ── */
const ACCEPTED_MIME=["image/jpeg","image/png"], ACCEPTED_EXTS=["jpg","jpeg","png"];
function validateFileType(f) { return ACCEPTED_MIME.includes(f.type) && ACCEPTED_EXTS.includes(f.name.split(".").pop().toLowerCase()); }
function validateFileSize(f) { return f.size <= MAX_FILE_SIZE; }

/* ── Navigation ── */
let currentPage = "dashboard";
function navigateTo(page) {
  document.querySelectorAll(".page").forEach(p=>p.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach(n=>n.classList.remove("active"));
  const pageEl=document.getElementById("page-"+page), navEl=document.querySelector(`[data-page="${page}"]`);
  if (pageEl) pageEl.classList.add("active"); if (navEl) navEl.classList.add("active");
  const titles={dashboard:t("nav_dashboard"),analyze:t("nav_analyze"),batch:t("nav_batch"),history:t("nav_history"),reports:t("nav_reports"),admin:t("nav_admin")};
  setText("topbar-title",titles[page]||page); currentPage=page;
  document.getElementById("sidebar").classList.remove("open"); document.getElementById("sidebar-overlay").classList.remove("open");
  if (page==="dashboard") populateDashboard();
  if (page==="history")   populateHistory();
  if (page==="reports")   populateReports();
  if (page==="admin")     populateAdmin();
}
function wireNavLink(el) { el.addEventListener("click",function(e){e.preventDefault();const target=this.dataset.nav||this.dataset.page;if(target)navigateTo(target);}); }

/* ── API Status ── */
async function checkApiStatus() {
  const dot=document.getElementById("status-dot"), txt=document.getElementById("status-text");
  try { const res=await fetch(API_BASE_URL+"/health",{signal:AbortSignal.timeout(3000)}); const j=await res.json().catch(()=>({})); if(res.ok&&j.model_loaded){dot.className="status-dot online";txt.textContent="API Online";}else{dot.className="status-dot offline";txt.textContent="Model not loaded";} } catch{dot.className="status-dot offline";txt.textContent="API Offline";}
}

/* ── Chatbot ── */
let chatbotContext = {};
function initChatbot() {
  const toggleBtn=document.getElementById("chatbot-toggle"), win=document.getElementById("chatbot-window"), closeBtn=document.getElementById("chatbot-close"), input=document.getElementById("chatbot-input"), sendBtn=document.getElementById("chatbot-send");
  if (!toggleBtn) return;
  appendBotMessage("👋 Hi! I'm AquaScan AI — your fish health assistant.\n\nAsk me about diseases, treatments, water quality, or type **help**!");
  let open=false;
  toggleBtn.addEventListener("click",()=>{open=!open;win.hidden=!open;document.getElementById("chatbot-badge").hidden=true;if(open&&input)input.focus();});
  closeBtn&&closeBtn.addEventListener("click",()=>{open=false;win.hidden=true;});
  document.getElementById("chatbot-chips")&&document.getElementById("chatbot-chips").querySelectorAll(".chatbot-chip").forEach(ch=>{ch.addEventListener("click",()=>sendChatMsg(ch.dataset.msg));});
  sendBtn&&sendBtn.addEventListener("click",()=>{const m=input.value.trim();if(m){sendChatMsg(m);input.value="";}});
  input&&input.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();const m=input.value.trim();if(m){sendChatMsg(m);input.value="";}}});
}
async function sendChatMsg(text) {
  appendUserMessage(text);
  const typing=appendTyping();
  try {
    const res=await fetch(API_BASE_URL+"/chatbot",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:text,context:chatbotContext})});
    const data=await res.json(); typing.remove();
    appendBotMessage(res.ok?data.response:"Sorry, could not process that.");
  } catch{typing.remove();appendBotMessage("⚠️ Can't reach server. Make sure API server is running.");}
  const ls=loadHistory(); if(ls[0]) chatbotContext.last_disease=ls[0].predicted_class;
}
function appendUserMessage(text){const m=document.getElementById("chatbot-messages");if(!m)return;const d=document.createElement("div");d.className="chat-msg chat-user";d.innerHTML=`<div class="chat-bubble chat-bubble-user">${esc(text)}</div>`;m.appendChild(d);m.scrollTop=m.scrollHeight;}
function appendBotMessage(text){const m=document.getElementById("chatbot-messages");if(!m)return;const d=document.createElement("div");d.className="chat-msg chat-bot";d.innerHTML=`<div class="chat-bubble chat-bubble-bot">${esc(text).replace(/\*\*(.*?)\*\*/g,"<strong>$1</strong>").replace(/\n/g,"<br>")}</div>`;m.appendChild(d);m.scrollTop=m.scrollHeight;return d;}
function appendTyping(){const m=document.getElementById("chatbot-messages");const d=document.createElement("div");d.className="chat-msg chat-bot";d.innerHTML=`<div class="chat-bubble chat-bubble-bot chat-typing"><span></span><span></span><span></span></div>`;m&&m.appendChild(d);m&&(m.scrollTop=m.scrollHeight);return d;}
function esc(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");}

/* ── Analyze ── */
let selectedFile = null;
function handleFile(file) {
  if (!file) return;
  hide("upload-error"); hide("result-card"); hide("error-banner"); hide("confidence-warning");
  if (!validateFileType(file)){const e=document.getElementById("upload-error");if(e){e.textContent="⚠️ Unsupported format. Use JPG or PNG.";show(e);}selectedFile=null;hide("preview-section");return;}
  if (!validateFileSize(file)){const e=document.getElementById("upload-error");if(e){e.textContent="⚠️ File too large. Max 10 MB.";show(e);}selectedFile=null;hide("preview-section");return;}
  selectedFile=file;
  const reader=new FileReader(); reader.onload=e=>{document.getElementById("preview-image").src=e.target.result;setText("preview-filename",file.name);const kb=file.size/1024;setText("preview-size",kb<1024?kb.toFixed(1)+" KB":(kb/1024).toFixed(2)+" MB");show("preview-section");};reader.readAsDataURL(file);
}
function renderResult(data) {
  const c=getDiseaseContent(data.predicted_class), h=isHealthy(data.predicted_class), cls=h?"healthy":"disease", pct=(data.confidence*100).toFixed(1);
  addHistoryEntry({predicted_class:data.predicted_class,confidence:data.confidence,severity:data.severity||"none",timestamp:Date.now()/1000,filename:selectedFile?selectedFile.name:""});
  lastResultData={...data,filename:selectedFile?selectedFile.name:""};
  if(data.confidence<CONF_WARN)show("confidence-warning");else hide("confidence-warning");
  document.getElementById("result-image").src=document.getElementById("preview-image").src;
  const badge=document.getElementById("result-badge");badge.textContent=h?t("badge_healthy"):t("badge_disease");badge.className="result-badge "+cls;
  const lbl=document.getElementById("result-label");lbl.textContent=c.label;lbl.className="result-label "+cls;
  const sr=document.getElementById("severity-row"),sb=document.getElementById("severity-badge");
  if(!h&&data.severity){sb.textContent=t("sev_"+data.severity);sb.className="severity-badge "+data.severity;show(sr);}else hide(sr);
  setText("result-confidence",pct+"%");
  const bar=document.getElementById("conf-bar");if(bar){bar.className="conf-bar "+cls;bar.style.width="0%";requestAnimationFrame(()=>requestAnimationFrame(()=>{bar.style.width=pct+"%";}));}
  const allProbs=data.all_probabilities;
  if(allProbs){const grid=document.getElementById("probs-grid");const keys=["healthy","eus","gill","red_spot"];grid.innerHTML=keys.map(k=>{const p=((allProbs[k]||0)*100).toFixed(1);return `<div class="prob-item"><div class="prob-label">${getDiseaseContent(k).label}</div><div class="prob-val">${p}%</div><div class="prob-bar-track"><div class="prob-bar-fill" style="width:${p}%"></div></div></div>`;}).join("");show("all-probs");}
  const descEl=document.getElementById("result-desc");if(!h&&c.description){descEl.textContent=c.description;show(descEl);}else hide(descEl);
  if(data.gradcam){document.getElementById("gradcam-img").src=data.gradcam;show("gradcam-section");}else hide("gradcam-section");
  const tl=document.getElementById("treatment-list"),ts=document.getElementById("treatment-section");
  if(!h&&c.treatment.length){tl.innerHTML=c.treatment.map(tr=>`<li>${tr}</li>`).join("");show(ts);}else hide(ts);
  const card=document.getElementById("result-card");card.className="card result-card "+cls;show(card);
}

/* ── Camera ── */
let cameraStream=null;
async function openCamera(){try{cameraStream=await navigator.mediaDevices.getUserMedia({video:{facingMode:"environment"}});document.getElementById("camera-video").srcObject=cameraStream;show("camera-container");}catch(err){alert("Camera error: "+err.message);}}
function closeCamera(){if(cameraStream){cameraStream.getTracks().forEach(t=>t.stop());cameraStream=null;}hide("camera-container");}
function capturePhoto(){const v=document.getElementById("camera-video"),cv=document.getElementById("capture-canvas");cv.width=v.videoWidth;cv.height=v.videoHeight;cv.getContext("2d").drawImage(v,0,0);cv.toBlob(b=>{if(b){const f=new File([b],"capture.jpg",{type:"image/jpeg"});closeCamera();handleFile(f);}}, "image/jpeg",0.9);}

/* ── Batch ── */
let batchFiles=[];
function updateBatchFileList(){const l=document.getElementById("batch-file-list"),b=document.getElementById("batch-analyze-btn");if(!l)return;if(!batchFiles.length){hide(l);hide(b);return;}l.innerHTML=batchFiles.map(f=>`<div class="batch-file-item"><span>🖼</span>${f.name}</div>`).join("");show(l);show(b);}
async function runBatchAnalysis(){
  if(!batchFiles.length)return;
  show("batch-loading");hide("batch-results");setText("batch-total-count",batchFiles.length);setText("batch-progress","0");
  const btn=document.getElementById("batch-analyze-btn");btn.disabled=true;
  const fd=new FormData();batchFiles.forEach(f=>fd.append("files",f));
  try{
    const res=await apiFetch("/predict-batch",{method:"POST",body:fd});
    const data=await res.json();if(!res.ok){alert(data.detail||"Batch failed.");return;}
    let p=0;const iv=setInterval(()=>{p=Math.min(p+10,100);setText("batch-progress",Math.round((p/100)*batchFiles.length));const pb=document.getElementById("batch-progress-bar");if(pb)pb.style.width=p+"%";if(p>=100)clearInterval(iv);},80);
    data.results.forEach(r=>{if(r.predicted_class)addHistoryEntry({predicted_class:r.predicted_class,confidence:r.confidence,severity:r.severity||"none",timestamp:Date.now()/1000,filename:r.filename});});
    const su=document.getElementById("batch-summary");su.innerHTML=`<div class="batch-stat"><div class="batch-stat-val">${data.total}</div><div class="batch-stat-lbl">Total</div></div><div class="batch-stat" style="border-top:3px solid var(--green)"><div class="batch-stat-val" style="color:var(--green)">${data.healthy_count}</div><div class="batch-stat-lbl">Healthy</div></div><div class="batch-stat" style="border-top:3px solid var(--red)"><div class="batch-stat-val" style="color:var(--red)">${data.diseased_count}</div><div class="batch-stat-lbl">Diseased</div></div><div class="batch-stat" style="border-top:3px solid var(--amber)"><div class="batch-stat-val" style="color:var(--amber)">${Math.round((data.healthy_count/data.total)*100)}%</div><div class="batch-stat-lbl">Health Rate</div></div>`;
    buildPieChart("batchPieChart",data.results.filter(r=>r.predicted_class).map(r=>({predicted_class:r.predicted_class})),"batchPie");
    const tb=document.getElementById("batch-tbody");tb.innerHTML=data.results.map((r,i)=>{if(r.error)return `<tr><td>${i+1}</td><td>${r.filename}</td><td colspan="4" style="color:var(--red)">${r.error}</td></tr>`;const c=getDiseaseContent(r.predicted_class),h=isHealthy(r.predicted_class);return `<tr><td>${i+1}</td><td>${r.filename}</td><td>${c.label}</td><td>${formatConfidence(r.confidence)}</td><td><span class="severity-badge ${r.severity}">${t("sev_"+r.severity)}</span></td><td><span class="badge ${h?"badge-green":"badge-red"}">${h?"✅":"⚠️"}</span></td></tr>`;}).join("");
    show("batch-results");
  }catch(err){alert("Error: "+err.message);}
  finally{hide("batch-loading");btn.disabled=false;}
}

/* ════════════════════════════════════════════
   INIT
════════════════════════════════════════════ */

/* ── Post-login: sync server history, wipe old shared data ── */
async function postLogin() {
  // Wipe old shared-key leftovers from before per-user isolation
  ["aquascan_history", "aquascan_h_guest"].forEach(k => localStorage.removeItem(k));

  // Sync this user's scans from the server into their keyed local cache
  try {
    const res = await apiFetch("/scans/mine");
    if (res.ok) {
      const data = await res.json();
      saveHistory(data.scans || []);
    }
  } catch { /* server may be offline — local cache stays empty */ }

  // Rebuild UI with correct data
  populateDashboard();
}
document.addEventListener("DOMContentLoaded", function () {

  /* Date */
  const de=document.getElementById("topbar-date");if(de)de.textContent=new Date().toLocaleDateString("en-GB",{weekday:"short",year:"numeric",month:"short",day:"numeric"});

  /* Language */
  document.querySelectorAll(".lang-btn").forEach(btn=>{btn.addEventListener("click",function(){currentLang=this.dataset.lang;document.querySelectorAll(".lang-btn").forEach(b=>b.classList.remove("active"));this.classList.add("active");applyI18n();if(currentPage==="dashboard")populateDashboard();if(currentPage==="history")populateHistory();if(currentPage==="reports")populateReports();});});

  /* Sidebar nav */
  document.querySelectorAll(".nav-item").forEach(el=>{el.addEventListener("click",function(e){e.preventDefault();navigateTo(this.dataset.page);});});
  document.querySelectorAll("[data-nav]").forEach(wireNavLink);

  /* Mobile menu */
  const menuBtn=document.getElementById("menu-btn"),sidebar=document.getElementById("sidebar"),overlay=document.getElementById("sidebar-overlay");
  menuBtn&&menuBtn.addEventListener("click",()=>{sidebar.classList.toggle("open");overlay.classList.toggle("open");});
  overlay&&overlay.addEventListener("click",()=>{sidebar.classList.remove("open");overlay.classList.remove("open");});

  /* Auth tabs */
  window.switchTab = window.switchTab || function(tab){};

  /* Login form */
  const loginForm=document.getElementById("login-form");
  loginForm&&loginForm.addEventListener("submit",async function(e){
    e.preventDefault();
    const u=document.getElementById("login-username").value.trim(), p=document.getElementById("login-pin").value.trim(), err=document.getElementById("login-error");
    if(!u||!p){if(err){err.textContent="Enter username and PIN.";err.hidden=false;}return;}
    try{
      const res=await fetch(API_BASE_URL+"/auth/login",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({username:u,pin:p})});
      const data=await res.json();
      if(res.ok){Auth.set(data);hideLoginModal();updateAuthUI();if(err)err.hidden=true;await postLogin();showToast("Welcome back, "+data.username+"! 🐟","blue");}
      else{if(err){err.textContent=data.detail||"Login failed.";err.hidden=false;}}
    }catch{if(err){err.textContent="Network error. Is the server running?";err.hidden=false;}}
  });

  /* Signup form */
  const signupForm=document.getElementById("signup-form");
  signupForm&&signupForm.addEventListener("submit",async function(e){
    e.preventDefault();
    const username=document.getElementById("signup-username").value.trim().toLowerCase(), pin=document.getElementById("signup-pin").value.trim(), pinC=document.getElementById("signup-pin-confirm").value.trim(), email=document.getElementById("signup-email").value.trim(), fullName=document.getElementById("signup-fullname").value.trim(), err=document.getElementById("signup-error");
    const showErr=m=>{if(err){err.textContent=m;err.hidden=false;}};
    if(!username||!pin){showErr("Username and PIN required.");return;}
    if(pin!==pinC){showErr("PINs do not match.");return;}
    if(pin.length<4){showErr("PIN must be at least 4 characters.");return;}
    if(!/^[a-z0-9_]{3,30}$/.test(username)){showErr("Username: 3–30 chars, letters/numbers/underscore only.");return;}
    try{
      const res=await fetch(API_BASE_URL+"/auth/signup",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({username,pin,email,full_name:fullName})});
      const data=await res.json();
      if(res.ok){Auth.set(data);hideLoginModal();updateAuthUI();if(err)err.hidden=true;await postLogin();showToast("🎉 Welcome, "+data.username+"! Account created.","green");}
      else{showErr(data.detail||"Signup failed.");}
    }catch{showErr("Network error. Is the server running?");}
  });

  /* Logout */
  document.getElementById("logout-nav-btn")&&document.getElementById("logout-nav-btn").addEventListener("click",()=>{
    // Clear this user's local cache on logout so next user starts fresh
    clearHistory();
    Auth.clear();
    updateAuthUI();
    // Reset all charts so they don't show previous user's data
    Object.keys(charts).forEach(k => { if(charts[k]){charts[k].destroy();charts[k]=null;} });
    showLoginModal();
  });
  document.getElementById("login-nav-btn")&&document.getElementById("login-nav-btn").addEventListener("click",showLoginModal);

  /* File input */
  const fileInput=document.getElementById("file-input");
  fileInput&&fileInput.addEventListener("change",()=>{if(fileInput.files&&fileInput.files[0])handleFile(fileInput.files[0]);});

  /* Drag & drop */
  const dz=document.getElementById("drop-zone");
  if(dz){
    dz.addEventListener("dragover",e=>{e.preventDefault();dz.classList.add("drag-over");});
    dz.addEventListener("dragleave",e=>{if(!dz.contains(e.relatedTarget))dz.classList.remove("drag-over");});
    dz.addEventListener("drop",e=>{e.preventDefault();dz.classList.remove("drag-over");const f=e.dataTransfer&&e.dataTransfer.files&&e.dataTransfer.files[0];if(f)handleFile(f);});
    dz.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();fileInput&&fileInput.click();}});
    dz.addEventListener("click",e=>{if(e.target===dz||e.target.classList.contains("drop-title")||e.target.classList.contains("drop-inner"))fileInput&&fileInput.click();});
  }

  /* Camera */
  document.getElementById("camera-btn")&&document.getElementById("camera-btn").addEventListener("click",openCamera);
  document.getElementById("capture-btn")&&document.getElementById("capture-btn").addEventListener("click",capturePhoto);
  document.getElementById("close-camera-btn")&&document.getElementById("close-camera-btn").addEventListener("click",closeCamera);

  /* Change / new / retry */
  const resetAnalyze=()=>{selectedFile=null;if(fileInput)fileInput.value="";hide("preview-section");hide("result-card");hide("error-banner");hide("confidence-warning");hide("all-probs");};
  document.getElementById("change-btn")&&document.getElementById("change-btn").addEventListener("click",resetAnalyze);
  document.getElementById("new-analysis-btn")&&document.getElementById("new-analysis-btn").addEventListener("click",resetAnalyze);
  document.getElementById("retry-btn")&&document.getElementById("retry-btn").addEventListener("click",()=>hide("error-banner"));

  /* Analyze */
  const analyzeBtn=document.getElementById("analyze-btn");
  analyzeBtn&&analyzeBtn.addEventListener("click",async function(){
    if(!selectedFile)return;
    show("loading");hide("result-card");hide("error-banner");hide("confidence-warning");analyzeBtn.disabled=true;
    const fd=new FormData();fd.append("file",selectedFile);
    try{
      const res=await apiFetch("/predict",{method:"POST",body:fd});
      let json=null;try{json=await res.json();}catch{}
      if(!json){showAnalyzeError("Unexpected server response.");return;}
      if(res.ok){if(typeof json.predicted_class!=="string"){showAnalyzeError("Invalid response format.");return;}renderResult(json);}
      else{showAnalyzeError(json.detail||json.error||`Server error (${res.status}).`);}
    }catch(err){if(err.message!=="Unauthorized")showAnalyzeError("Network error. Make sure the API server is running.");}
    finally{hide("loading");analyzeBtn.disabled=false;}
  });
  function showAnalyzeError(msg){setText("error-message",msg);show("error-banner");}

  /* PDF Download */
  document.getElementById("pdf-btn")&&document.getElementById("pdf-btn").addEventListener("click",()=>downloadPDF(lastResultData));

  /* Batch */
  const batchInput=document.getElementById("batch-file-input");
  batchInput&&batchInput.addEventListener("change",()=>{batchFiles=Array.from(batchInput.files||[]).filter(f=>validateFileType(f)).slice(0,20);updateBatchFileList();});
  const batchDrop=document.getElementById("batch-drop-zone");
  if(batchDrop){batchDrop.addEventListener("dragover",e=>{e.preventDefault();batchDrop.classList.add("drag-over");});batchDrop.addEventListener("dragleave",e=>{if(!batchDrop.contains(e.relatedTarget))batchDrop.classList.remove("drag-over");});batchDrop.addEventListener("drop",e=>{e.preventDefault();batchDrop.classList.remove("drag-over");batchFiles=Array.from(e.dataTransfer.files||[]).filter(f=>validateFileType(f)).slice(0,20);updateBatchFileList();});batchDrop.addEventListener("click",()=>batchInput&&batchInput.click());}
  document.getElementById("batch-analyze-btn")&&document.getElementById("batch-analyze-btn").addEventListener("click",runBatchAnalysis);
  document.getElementById("batch-export-btn")&&document.getElementById("batch-export-btn").addEventListener("click",()=>exportCSV());

  /* History */
  const hs=document.getElementById("history-search"),hf=document.getElementById("history-filter");
  hs&&hs.addEventListener("input",()=>populateHistory(hf?hf.value:"all",hs.value));
  hf&&hf.addEventListener("change",()=>populateHistory(hf.value,hs?hs.value:""));
  document.getElementById("clear-history-btn")&&document.getElementById("clear-history-btn").addEventListener("click",async()=>{
    if(!confirm("Clear all your scan history?"))return;
    try{await apiFetch("/scans/mine",{method:"DELETE"});}catch{}
    clearHistory();populateHistory();populateDashboard();
  });

  /* Reports */
  document.getElementById("export-csv-btn")&&document.getElementById("export-csv-btn").addEventListener("click",()=>exportCSV());
  /* Remove print-btn — replaced with PDF download only */
  const printBtn=document.getElementById("print-btn");if(printBtn)printBtn.remove();

  /* Init */
  updateAuthUI();
  if (!Auth.isLoggedIn()) showLoginModal();
  applyI18n();
  populateDashboard();
  checkApiStatus();
  setInterval(checkApiStatus,30000);
  initChatbot();
});

if (typeof module !== "undefined") { module.exports = { validateFileType, validateFileSize, formatConfidence, getDiseaseContent }; }
