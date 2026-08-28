import client from "./client";

// ---------- Auth ----------
export const registerUser = (payload) => client.post("/auth/register", payload);
export const loginUser = (payload) => client.post("/auth/login", payload);
export const logoutUser = () => client.post("/auth/logout");
export const getMe = () => client.get("/auth/me");

// ---------- Profile ----------
export const getMyProfile = () => client.get("/users/me");
export const updateMyProfile = (payload) => client.put("/users/me", payload);

// ---------- Prediction ----------
export const getModelStatus = () => client.get("/model-status");
export const predictImage = (formData, onUploadProgress) =>
  client.post("/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress,
  });
export const predictGradcam = (formData) =>
  client.post("/predict/gradcam", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
export const getMyPredictions = (params) => client.get("/predictions", { params });
export const getPredictionDetail = (id) => client.get(`/predictions/${id}`);

// ---------- Model / Research (public) ----------
export const getModelInfo = () => client.get("/model-info");
export const getClasses = () => client.get("/classes");
export const getResearchAll = () => client.get("/research/all");
export const getConfusionMatrix = () => client.get("/research/confusion-matrix");
export const getClassificationReport = () => client.get("/research/classification-report");
export const getRocAuc = () => client.get("/research/roc-auc");
export const getCnnComparison = () => client.get("/research/cnn-comparison");
export const getFusionResults = () => client.get("/research/fusion-results");
export const getPcaComparison = () => client.get("/research/pca-comparison");
export const getKernelComparison = () => client.get("/research/kernel-comparison");
export const getErrorAnalysis = () => client.get("/research/error-analysis");
export const getDatasetStats = () => client.get("/research/dataset-stats");

// ---------- Admin ----------
export const getAdminOverview = () => client.get("/admin/overview");
export const getAdminAnalytics = () => client.get("/admin/analytics");
export const getAdminUsers = (params) => client.get("/admin/users", { params });
export const getAdminUser = (id) => client.get(`/admin/users/${id}`);
export const updateAdminUser = (id, payload) => client.patch(`/admin/users/${id}`, payload);
export const deleteAdminUser = (id) => client.delete(`/admin/users/${id}`);
export const getAdminPredictions = (params) => client.get("/admin/predictions", { params });
export const getAdminPredictionDetail = (id) => client.get(`/admin/predictions/${id}`);
