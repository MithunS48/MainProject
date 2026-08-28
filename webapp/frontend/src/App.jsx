import { Routes, Route } from "react-router-dom";
import { Toaster } from "react-hot-toast";

import { ProtectedRoute, FarmerRoute, AdminRoute, GuestRoute } from "./components/RouteGuards";
import DashboardLayout from "./layouts/DashboardLayout";

import LandingPage from "./pages/LandingPage";
import LoginPage from "./pages/auth/LoginPage";
import RegisterPage from "./pages/auth/RegisterPage";

import FarmerDashboard from "./pages/farmer/FarmerDashboard";
import DetectDisease from "./pages/farmer/DetectDisease";
import PredictionHistory from "./pages/farmer/PredictionHistory";
import DiseaseInfo from "./pages/farmer/DiseaseInfo";
import Profile from "./pages/farmer/Profile";

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
    <div className="min-h-screen bg-ocean-950 flex flex-col items-center justify-center text-white px-5">
      <h1 className="font-display text-5xl font-bold mb-3">404</h1>
      <p className="text-white/60">The page you're looking for doesn't exist.</p>
      <a href="/" className="btn-primary mt-6">
        Back to Home
      </a>
    </div>
  );
}

export default function App() {
  return (
    <>
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: "rgba(11, 48, 73, 0.95)",
            color: "#fff",
            border: "1px solid rgba(255,255,255,0.1)",
            backdropFilter: "blur(12px)",
          },
        }}
      />

      <Routes>
        {/* Public */}
        <Route path="/" element={<LandingPage />} />
        <Route
          path="/login"
          element={
            <GuestRoute>
              <LoginPage />
            </GuestRoute>
          }
        />
        <Route
          path="/register"
          element={
            <GuestRoute>
              <RegisterPage />
            </GuestRoute>
          }
        />

        {/* Farmer dashboard */}
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
              <DashboardLayout variant="farmer">
                <DetectDisease />
              </DashboardLayout>
            </FarmerRoute>
          }
        />
        <Route
          path="/dashboard/history"
          element={
            <FarmerRoute>
              <DashboardLayout variant="farmer">
                <PredictionHistory />
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

        {/* Admin dashboard */}
        <Route
          path="/admin"
          element={
            <AdminRoute>
              <DashboardLayout variant="admin">
                <AdminDashboard />
              </DashboardLayout>
            </AdminRoute>
          }
        />
        <Route
          path="/admin/users"
          element={
            <AdminRoute>
              <DashboardLayout variant="admin">
                <UserManagement />
              </DashboardLayout>
            </AdminRoute>
          }
        />
        <Route
          path="/admin/predictions"
          element={
            <AdminRoute>
              <DashboardLayout variant="admin">
                <PredictionManagement />
              </DashboardLayout>
            </AdminRoute>
          }
        />
        <Route
          path="/admin/analytics"
          element={
            <AdminRoute>
              <DashboardLayout variant="admin">
                <Analytics />
              </DashboardLayout>
            </AdminRoute>
          }
        />
        <Route
          path="/admin/model-performance"
          element={
            <AdminRoute>
              <DashboardLayout variant="admin">
                <ModelPerformance />
              </DashboardLayout>
            </AdminRoute>
          }
        />
        <Route
          path="/admin/research"
          element={
            <AdminRoute>
              <DashboardLayout variant="admin">
                <ResearchResults />
              </DashboardLayout>
            </AdminRoute>
          }
        />
        <Route
          path="/admin/error-analysis"
          element={
            <AdminRoute>
              <DashboardLayout variant="admin">
                <ErrorAnalysis />
              </DashboardLayout>
            </AdminRoute>
          }
        />
        <Route
          path="/admin/profile"
          element={
            <AdminRoute>
              <DashboardLayout variant="admin">
                <AdminProfile />
              </DashboardLayout>
            </AdminRoute>
          }
        />

        {/* Fallback */}
        <Route path="*" element={<NotFound />} />
      </Routes>
    </>
  );
}
