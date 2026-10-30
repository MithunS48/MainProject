import { Routes, Route } from "react-router-dom";
import { Toaster } from "react-hot-toast";
import PWAInstallPrompt from "./components/PWAInstallPrompt";

import { ProtectedRoute, FarmerRoute, AdminRoute, GuestRoute } from "./components/RouteGuards";
import DashboardLayout from "./layouts/DashboardLayout";

// 3D Public pages
import Landing3D from "./pages/Landing3D";
import LoginPage from "./pages/auth/LoginPage";
import RegisterPage from "./pages/auth/RegisterPage";

// 3D Farmer pages
import FarmerDashboard from "./pages/farmer/FarmerDashboard";
import DetectDisease3D from "./pages/farmer/DetectDisease3D";
import PredictionHistory3D from "./pages/farmer/PredictionHistory3D";
import DiseaseInfo from "./pages/farmer/DiseaseInfo";
import Profile from "./pages/farmer/Profile";
import WaterQuality from "./pages/farmer/WaterQuality";

// Admin pages (unchanged)
import AdminDashboard from "./pages/admin/AdminDashboard";
import UserManagement from "./pages/admin/UserManagement";
import PredictionManagement from "./pages/admin/PredictionManagement";
import Analytics from "./pages/admin/Analytics";
import ModelPerformance from "./pages/admin/ModelPerformance";
import ResearchResults from "./pages/admin/ResearchResults";
import ErrorAnalysis from "./pages/admin/ErrorAnalysis";
import AdminProfile from "./pages/admin/AdminProfile";

function NotFound() {
  return (
    <div className="min-h-screen bg-[#020e1f] flex flex-col items-center justify-center text-white px-5">
      <div className="text-cyan-400 text-8xl font-extrabold mb-4">404</div>
      <p className="text-blue-200/50 text-lg">The page you're looking for doesn't exist.</p>
      <a
        href="/"
        className="mt-8 inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold hover:from-cyan-400 hover:to-blue-500 transition-all hover:scale-105"
      >
        Back to Home
      </a>
    </div>
  );
}

export default function App() {
  return (
    <>
      <PWAInstallPrompt />
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: "rgba(2,14,31,0.95)",
            color: "#fff",
            border: "1px solid rgba(0,212,255,0.2)",
            backdropFilter: "blur(12px)",
          },
        }}
      />

      <Routes>
        {/* Public — 3D Landing */}
        <Route path="/" element={<Landing3D />} />
        <Route
          path="/login"
          element={<GuestRoute><LoginPage /></GuestRoute>}
        />
        <Route
          path="/register"
          element={<GuestRoute><RegisterPage /></GuestRoute>}
        />

        {/* Farmer dashboard — 3D pages */}
        <Route
          path="/dashboard"
          element={
            <FarmerRoute>
              <DashboardLayout variant="farmer">
                <FarmerDashboard />
              </DashboardLayout>
            </FarmerRoute>
          }
        />
        <Route
          path="/dashboard/detect"
          element={
            <FarmerRoute>
              <DashboardLayout variant="farmer" dark>
                <DetectDisease3D />
              </DashboardLayout>
            </FarmerRoute>
          }
        />
        <Route
          path="/dashboard/history"
          element={
            <FarmerRoute>
              <DashboardLayout variant="farmer" dark>
                <PredictionHistory3D />
              </DashboardLayout>
            </FarmerRoute>
          }
        />
        <Route
          path="/dashboard/diseases"
          element={
            <FarmerRoute>
              <DashboardLayout variant="farmer">
                <DiseaseInfo />
              </DashboardLayout>
            </FarmerRoute>
          }
        />
        <Route
          path="/dashboard/profile"
          element={
            <FarmerRoute>
              <DashboardLayout variant="farmer">
                <Profile />
              </DashboardLayout>
            </FarmerRoute>
          }
        />
        <Route
          path="/dashboard/water-quality"
          element={
            <FarmerRoute>
              <DashboardLayout variant="farmer">
                <WaterQuality />
              </DashboardLayout>
            </FarmerRoute>
          }
        />

        {/* Admin dashboard — unchanged */}
        <Route
          path="/admin"
          element={<AdminRoute><DashboardLayout variant="admin"><AdminDashboard /></DashboardLayout></AdminRoute>}
        />
        <Route
          path="/admin/users"
          element={<AdminRoute><DashboardLayout variant="admin"><UserManagement /></DashboardLayout></AdminRoute>}
        />
        <Route
          path="/admin/predictions"
          element={<AdminRoute><DashboardLayout variant="admin"><PredictionManagement /></DashboardLayout></AdminRoute>}
        />
        <Route
          path="/admin/analytics"
          element={<AdminRoute><DashboardLayout variant="admin"><Analytics /></DashboardLayout></AdminRoute>}
        />
        <Route
          path="/admin/model-performance"
          element={<AdminRoute><DashboardLayout variant="admin"><ModelPerformance /></DashboardLayout></AdminRoute>}
        />
        <Route
          path="/admin/research"
          element={<AdminRoute><DashboardLayout variant="admin"><ResearchResults /></DashboardLayout></AdminRoute>}
        />
        <Route
          path="/admin/error-analysis"
          element={<AdminRoute><DashboardLayout variant="admin"><ErrorAnalysis /></DashboardLayout></AdminRoute>}
        />
        <Route
          path="/admin/profile"
          element={<AdminRoute><DashboardLayout variant="admin"><AdminProfile /></DashboardLayout></AdminRoute>}
        />

        {/* Fallback */}
        <Route path="*" element={<NotFound />} />
      </Routes>
    </>
  );
}
