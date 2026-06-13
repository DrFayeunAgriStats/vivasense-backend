import { useEffect, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { setVivaSenseMode } from "@/lib/vivasenseGating";

interface VivaSenseAuthGuardProps {
  children: React.ReactNode;
}

// Admin users retain access to these paths even if platform_source = 'legacy'
const ADMIN_BYPASS_PREFIXES = ["/admin/", "/vivasense-dashboard"];

function isAdminBypassPath(pathname: string): boolean {
  return ADMIN_BYPASS_PREFIXES.some((prefix) => pathname.startsWith(prefix));
}

export default function VivaSenseAuthGuard({ children }: VivaSenseAuthGuardProps) {
  const [loading, setLoading] = useState(true);
  const [authenticated, setAuthenticated] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  const redirectToAuth = () => {
    navigate(
      `/vivasense/auth?next=${encodeURIComponent(location.pathname)}`,
      { replace: true }
    );
  };

  const redirectToCompleteProfile = () => {
    navigate(
      `/vivasense/complete-profile?next=${encodeURIComponent(location.pathname)}`,
      { replace: true }
    );
  };

  const checkAccess = async (userId: string) => {
    let profile: any = null;

    // Fetch profile with fault tolerance — missing platform_source column must not block login
    try {
      const { data: fetchedProfile, error } = await supabase
        .from("profiles")
        .select("plan, platform_source")
        .eq("id", userId)
        .maybeSingle();

      if (error) {
        console.warn("VivaSenseAuthGuard: profile fetch error (will use defaults)", error.message);
        // If platform_source column doesn't exist or other query error, use safe defaults
        profile = { plan: "free", platform_source: "vivasense" };
      } else {
        profile = fetchedProfile;
      }
    } catch (err) {
      console.warn("VivaSenseAuthGuard: profile fetch exception (will use defaults)", err);
      // Graceful fallback: missing column or network error shouldn't block login
      profile = { plan: "free", platform_source: "vivasense" };
    }

    // Sync Pro/Free mode
    if (profile?.plan === "pro" || profile?.plan === "institutional") {
      setVivaSenseMode("pro");
    } else {
      setVivaSenseMode("free");
    }

    // Admin bypass: fetch roles only when profile is missing or legacy
    if (!profile || profile.platform_source !== "vivasense") {
      const { data: roleRows } = await supabase
        .from("user_roles")
        .select("role")
        .eq("user_id", userId);

      const isAdmin = roleRows?.some((r) => r.role === "admin") ?? false;

      if (isAdmin && isAdminBypassPath(location.pathname)) {
        // Admin accessing an admin path — let them through regardless of profile state
        setAuthenticated(true);
        return;
      }

      // Non-admin, or admin on a normal VivaSense route: require completion
      redirectToCompleteProfile();
      return;
    }

    setAuthenticated(true);
  };

  useEffect(() => {
    supabase.auth.getSession().then(async ({ data: { session } }) => {
      if (session?.user) {
        await checkAccess(session.user.id);
      } else {
        setVivaSenseMode("free");
        redirectToAuth();
      }
      setLoading(false);
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      if (session?.user) {
        void checkAccess(session.user.id).then(() => setLoading(false));
      } else {
        setVivaSenseMode("free");
        setAuthenticated(false);
        redirectToAuth();
      }
    });

    return () => subscription.unsubscribe();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.pathname]);

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0b1d14] flex flex-col items-center justify-center gap-4">
        <div className="w-12 h-12 border-4 border-emerald-400 border-t-transparent rounded-full animate-spin" />
        <p className="text-emerald-300 font-medium">Loading VivaSense...</p>
      </div>
    );
  }

  if (!authenticated) return null;

  return <>{children}</>;
}
