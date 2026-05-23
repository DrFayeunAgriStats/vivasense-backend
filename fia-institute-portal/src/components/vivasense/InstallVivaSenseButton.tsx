import { useEffect, useState } from "react";
import { Download } from "lucide-react";
import { Button } from "@/components/ui/button";

interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: "accepted" | "dismissed" }>;
}

declare global {
  interface Window {
    gtag?: (...args: unknown[]) => void;
  }
}

const trackEvent = (event: string, params: Record<string, unknown> = {}) => {
  try {
    window.gtag?.("event", event, { event_category: "pwa", ...params });
  } catch {
    // no-op
  }
};

export function InstallVivaSenseButton({ className }: { className?: string }) {
  const [deferredPrompt, setDeferredPrompt] = useState<BeforeInstallPromptEvent | null>(null);
  const [installed, setInstalled] = useState(false);

  useEffect(() => {
    // Hide if already running standalone
    const standalone =
      window.matchMedia?.("(display-mode: standalone)").matches ||
      // iOS Safari
      (window.navigator as unknown as { standalone?: boolean }).standalone === true;
    if (standalone) {
      setInstalled(true);
      trackEvent("standalone_usage", { page: "/vivasense" });
      return;
    }

    const onBeforeInstall = (e: Event) => {
      e.preventDefault();
      setDeferredPrompt(e as BeforeInstallPromptEvent);
    };
    const onInstalled = () => {
      setInstalled(true);
      setDeferredPrompt(null);
      trackEvent("app_installed", { page: "/vivasense" });
    };

    window.addEventListener("beforeinstallprompt", onBeforeInstall);
    window.addEventListener("appinstalled", onInstalled);
    return () => {
      window.removeEventListener("beforeinstallprompt", onBeforeInstall);
      window.removeEventListener("appinstalled", onInstalled);
    };
  }, []);

  const handleInstall = async () => {
    if (!deferredPrompt) {
      window.alert(
        "To install VivaSense, open your browser menu and choose 'Install app' or 'Add to Home Screen'."
      );
      return;
    }
    trackEvent("install_click", { page: "/vivasense" });
    await deferredPrompt.prompt();
    const choice = await deferredPrompt.userChoice;
    trackEvent("install_prompt_response", { outcome: choice.outcome });
    if (choice.outcome === "accepted") setInstalled(true);
    setDeferredPrompt(null);
  };

  if (installed) return null;

  return (
    <Button onClick={handleInstall} size="sm" className={className} variant={deferredPrompt ? "default" : "outline"}>
      <Download className="w-4 h-4 mr-2" />
      Install VivaSense App
    </Button>
  );
}
