import { useEffect, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";

interface VivaSenseAuthGuardProps {
  children: React.ReactNode;
}

export default function VivaSenseAuthGuard({ children }: VivaSenseAuthGuardProps) {
  const [loading, setLoading] = useState(true);
  const [authenticated, setAuthenticated] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session?.user) {
        setAuthenticated(true);
      } else {
        navigate(
          `/vivasense/auth?next=${encodeURIComponent(location.pathname)}`,
          { replace: true }
        );
      }
      setLoading(false);
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      if (session?.user) {
        setAuthenticated(true);
      } else {
        navigate(
          `/vivasense/auth?next=${encodeURIComponent(location.pathname)}`,
          { replace: true }
        );
      }
    });

    return () => subscription.unsubscribe();
  }, [navigate, location.pathname]);

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
