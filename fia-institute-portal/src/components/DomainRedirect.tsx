import { useEffect } from "react";

/**
 * Redirects standalone VivaSense domains to the canonical VivaSense page.
 *
 * Kept as a client-side fallback in case edge redirects are bypassed.
 */
export function DomainRedirect() {
  useEffect(() => {
    const sourceHosts = new Set(["vivasensestat.com", "www.vivasensestat.com"]);
    const targetUrl = "https://fieldtoinsightacademy.com.ng/vivasense";

    if (sourceHosts.has(window.location.hostname)) {
      window.location.replace(targetUrl);
    }
  }, []);

  return null;
}

export default DomainRedirect;
