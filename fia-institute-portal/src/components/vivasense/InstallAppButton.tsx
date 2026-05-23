import { useEffect, useState } from "react";
import { Download } from "lucide-react";
import { pwaInstaller } from "@/utils/pwa-install";

export function InstallAppButton() {
  const [canInstall, setCanInstall] = useState(false);
  const [isInstalled, setIsInstalled] = useState(false);

  useEffect(() => {
    // Subscribe to installation state changes
    const unsubscribe = pwaInstaller.onStateChange((installed) => {
      setIsInstalled(installed);
      setCanInstall(pwaInstaller.canInstall());
    });

    // Set initial state
    setCanInstall(pwaInstaller.canInstall());
    setIsInstalled(pwaInstaller.isAppInstalled());

    return unsubscribe;
  }, []);

  const handleInstall = async () => {
    const success = await pwaInstaller.install();
    if (success) {
      setCanInstall(false);
      setIsInstalled(true);
    }
  };

  if (isInstalled) {
    return null;
  }

  if (!canInstall) {
    return null;
  }

  return (
    <button
      onClick={handleInstall}
      className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg font-medium transition-colors"
      title="Install VivaSense as an app on your device"
    >
      <Download className="w-4 h-4" />
      <span className="hidden sm:inline">Install App</span>
      <span className="sm:hidden">Install</span>
    </button>
  );
}
