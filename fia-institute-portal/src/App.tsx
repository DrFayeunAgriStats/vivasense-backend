import { Toaster } from "@/components/ui/toaster";
import { DomainRedirect } from './components/DomainRedirect';
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { ScrollToTop } from "./components/ScrollToTop";
import { FloatingTutorWidget } from "./components/tutor/FloatingTutorWidget";
import { usePageTracking } from "./hooks/usePageTracking";
import { AuthProvider } from "./contexts/AuthContext";
import { VivaSenseProvider } from "./contexts/VivaSenseContext";
import Index from "./pages/Index";
import Programmes from "./pages/Programmes";
import DataClinic from "./pages/DataClinic";
import Program from "./pages/Program";
import About from "./pages/About";
import Founder from "./pages/Founder";
import AITutors from "./pages/AITutors";
import TechnicalInsights from "./pages/TechnicalInsights";
import ScalingAgriculturalExpertise from "./pages/insights/ScalingAgriculturalExpertise";
import Curriculum from "./pages/Curriculum";
import Faculty from "./pages/Faculty";
import Tools from "./pages/Tools";
import Outcomes from "./pages/Outcomes";
import Testimonials from "./pages/Testimonials";
import Pricing from "./pages/Pricing";
import Apply from "./pages/Apply";
import FAQ from "./pages/FAQ";
import Contact from "./pages/Contact";
import Services from "./pages/Services";
import DataAnalysisServices from "./pages/DataAnalysisServices";
import AgroServices from "./pages/AgroServices";
import AbleFlourishDigitalSystems from "./pages/AbleFlourishDigitalSystems";
import VivaSense from "./pages/VivaSense";
import VivaSenseWorkspace from "./pages/VivaSenseWorkspace";
import VivaSenseAnova from "./pages/VivaSenseAnova";
import VivaSenseGenetics from "./pages/VivaSenseGenetics";
import VivaSenseAdvanced from "./pages/VivaSenseAdvanced";
import VivaSenseAuth from "./pages/VivaSenseAuth";
import VivaSenseAuthGuard from "@/components/vivasense/VivaSenseAuthGuard";
import PrivacyPolicy from "@/pages/PrivacyPolicy";

import CSP409PlantGenetics from "./pages/CSP409PlantGenetics";
import CSP811BiometricalGenetics from "./pages/CSP811BiometricalGenetics";
import JournalHome from "./pages/journal/JournalHome";
import JournalAbout from "./pages/journal/JournalAbout";
import JournalAimsScope from "./pages/journal/JournalAimsScope";
import JournalEditorialBoard from "./pages/journal/JournalEditorialBoard";
import JournalAuthorGuidelines from "./pages/journal/JournalAuthorGuidelines";
import JournalSubmit from "./pages/journal/JournalSubmit";
import JournalCurrentIssue from "./pages/journal/JournalCurrentIssue";
import JournalArchive from "./pages/journal/JournalArchive";
import VerifyCertificate from "./pages/VerifyCertificate";
import CertificateVerify from "./pages/CertificateVerify";
import CertificateAdmin from "./pages/CertificateAdmin";
import ThesisMentor from "./pages/ThesisMentor";
import AdapTutor from "./pages/AdapTutor";
import AdapAdmin from "./pages/AdapAdmin";
import ManuscriptReviewer from "./pages/ManuscriptReviewer";
import ManuscriptReviewerUpload from "./pages/ManuscriptReviewerUpload";
import ManuscriptReviewerResults from "./pages/ManuscriptReviewerResults";

// Research Writing System pages
import RWSLanding from "./pages/rws/RWSLanding";
import RWSSignUp from "./pages/rws/RWSSignUp";
import RWSSignIn from "./pages/rws/RWSSignIn";
import RWSResetPassword from "./pages/rws/RWSResetPassword";
import RWSUpdatePassword from "./pages/rws/RWSUpdatePassword";
import RWSOnboarding from "./pages/rws/RWSOnboarding";
import RWSDiagnostic from "./pages/rws/RWSDiagnostic";
import RWSDashboard from "./pages/rws/RWSDashboard";
import RWSAcademicIntegrity from "./pages/rws/RWSAcademicIntegrity";
import RWSResultsLab from "./pages/rws/RWSResultsLab";
import RWSSupervisorDashboard from "./pages/rws/RWSSupervisorDashboard";
import RWSAdminDashboard from "./pages/rws/RWSAdminDashboard";
import RWSDefenseSimulator from "./pages/rws/RWSDefenseSimulator";
import RWSBookings from "./pages/rws/RWSBookings";
import RWSCertificateGate from "./pages/rws/RWSCertificateGate";
import RWSPortfolio from "./pages/rws/RWSPortfolio";
import RWSDailyDrill from "./pages/rws/RWSDailyDrill";
import Cohort3Registration from "./pages/Cohort3Registration";
import PlantImprovementTutor from "./pages/PlantImprovementTutor";

import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

function AppRoutes() {
  usePageTracking();
  return (
    <>
      <ScrollToTop />
      <DomainRedirect />
      <Routes>
        <Route path="/" element={<Index />} />
        <Route path="/programmes" element={<Programmes />} />
        <Route path="/programmes/data-clinic" element={<DataClinic />} />
        <Route path="/about" element={<About />} />
        <Route path="/founder" element={<Founder />} />
        <Route path="/ai-tutors" element={<AITutors />} />
        <Route path="/technical-insights" element={<TechnicalInsights />} />
        <Route path="/technical-insights/scaling-agricultural-expertise" element={<ScalingAgriculturalExpertise />} />
        <Route path="/program" element={<Program />} />
        <Route path="/curriculum" element={<Curriculum />} />
        <Route path="/faculty" element={<Faculty />} />
        <Route path="/instructor" element={<Faculty />} />
        <Route path="/tools" element={<Tools />} />
        <Route path="/outcomes" element={<Outcomes />} />
        <Route path="/testimonials" element={<Testimonials />} />
        <Route path="/pricing" element={<Pricing />} />
        <Route path="/apply" element={<Apply />} />
        <Route path="/faq" element={<FAQ />} />
        <Route path="/contact" element={<Contact />} />
        <Route path="/services" element={<Services />} />
        <Route path="/services/agro-services" element={<AgroServices />} />
        <Route path="/services/data-analysis" element={<DataAnalysisServices />} />
        <Route path="/services/able-flourish-digital-systems" element={<AbleFlourishDigitalSystems />} />
        <Route path="/agro-services" element={<Navigate to="/services/agro-services" replace />} />
        <Route path="/vivasense" element={<VivaSenseProvider><VivaSense /></VivaSenseProvider>} />
        <Route path="/vivasense/workspace" element={<Navigate to="/vivasense/genetics" replace />} />
        <Route path="/vivasense/auth" element={<VivaSenseAuth />} />
        <Route path="/vivasense/anova" element={<VivaSenseAuthGuard><VivaSenseAnova /></VivaSenseAuthGuard>} />
        <Route path="/vivasense/genetics" element={<VivaSenseAuthGuard><VivaSenseGenetics /></VivaSenseAuthGuard>} />
        <Route path="/vivasense/advanced" element={<VivaSenseAuthGuard><VivaSenseAdvanced /></VivaSenseAuthGuard>} />
        <Route path="/privacy" element={<PrivacyPolicy />} />
        <Route path="/tutor" element={<Navigate to="/adap-tutor" replace />} />
        <Route path="/plant-genetics-mastery-tutor" element={<CSP409PlantGenetics />} />
        <Route path="/csp409-plant-genetics" element={<Navigate to="/plant-genetics-mastery-tutor" replace />} />
        <Route path="/csp409-ai-tutor" element={<Navigate to="/plant-genetics-mastery-tutor" replace />} />
        <Route path="/biometrical-genetics-mastery-tutor" element={<CSP811BiometricalGenetics />} />
        <Route
          path="/csp811-biometrical-genetics"
          element={<Navigate to="/biometrical-genetics-mastery-tutor" replace />}
        />
        <Route path="/journal" element={<JournalHome />} />
        <Route path="/journal/about" element={<JournalAbout />} />
        <Route path="/journal/aims-scope" element={<JournalAimsScope />} />
        <Route path="/journal/editorial-board" element={<JournalEditorialBoard />} />
        <Route path="/journal/author-guidelines" element={<JournalAuthorGuidelines />} />
        <Route path="/journal/submit" element={<JournalSubmit />} />
        <Route path="/journal/current-issue" element={<JournalCurrentIssue />} />
        <Route path="/journal/archive" element={<JournalArchive />} />
        <Route path="/verify-certificate" element={<Navigate to="/verify" replace />} />
        <Route path="/verify" element={<CertificateVerify />} />
        <Route path="/admin/certificates" element={<CertificateAdmin />} />
        <Route path="/thesis-mentor" element={<ThesisMentor />} />
        <Route path="/adap-tutor" element={<AdapTutor />} />
        <Route path="/admin/adap" element={<AdapAdmin />} />
        <Route path="/quiz" element={<Navigate to="/adap-tutor" replace />} />
        <Route path="/manuscript-reviewer" element={<ManuscriptReviewer />} />
        <Route path="/manuscript-reviewer/upload" element={<ManuscriptReviewerUpload />} />
        <Route path="/manuscript-reviewer/results" element={<ManuscriptReviewerResults />} />

        {/* Research Writing System */}
        <Route path="/research-writing" element={<RWSLanding />} />
        <Route path="/research-writing/signup" element={<RWSSignUp />} />
        <Route path="/research-writing/signin" element={<RWSSignIn />} />
        <Route path="/research-writing/reset-password" element={<RWSResetPassword />} />
        <Route path="/research-writing/update-password" element={<RWSUpdatePassword />} />
        <Route path="/research-writing/onboarding" element={<RWSOnboarding />} />
        <Route path="/research-writing/diagnostic" element={<RWSDiagnostic />} />
        <Route path="/research-writing/dashboard" element={<RWSDashboard />} />
        <Route path="/research-writing/academic-integrity" element={<RWSAcademicIntegrity />} />
        <Route path="/research-writing/results-lab" element={<RWSResultsLab />} />
        <Route path="/research-writing/supervisor" element={<RWSSupervisorDashboard />} />
        <Route path="/research-writing/admin" element={<RWSAdminDashboard />} />
        <Route path="/research-writing/defense" element={<RWSDefenseSimulator />} />
        <Route path="/research-writing/bookings" element={<RWSBookings />} />
        <Route path="/research-writing/certificate" element={<RWSCertificateGate />} />
        <Route path="/research-writing/portfolio" element={<RWSPortfolio />} />
        <Route path="/research-writing/daily-drill" element={<RWSDailyDrill />} />

        <Route path="/cohort3-registration" element={<Cohort3Registration />} />
        <Route path="/ai-tutors/plant-improvement" element={<PlantImprovementTutor />} />

        {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
        <Route path="*" element={<NotFound />} />
      </Routes>
    </>
  );
}

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <AuthProvider>
          <AppRoutes />
        </AuthProvider>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
