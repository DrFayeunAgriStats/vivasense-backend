import { useEffect } from "react";
import { useLocation } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";

export function usePageTracking() {
  const location = useLocation();

  useEffect(() => {
    const trackVisit = async () => {
      await supabase.from("page_visits").insert({
        page_path: location.pathname,
        user_agent: navigator.userAgent,
        referrer: document.referrer || null,
      });
    };
    trackVisit();
  }, [location.pathname]);
}
