// PWA Installation Helper
// Manages the install prompt and install-related events

interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: "accepted" | "dismissed" }>;
}

export class PWAInstaller {
  private deferredPrompt: BeforeInstallPromptEvent | null = null;
  private isInstalled: boolean = false;
  private listeners: Set<(installed: boolean) => void> = new Set();

  constructor() {
    this.init();
  }

  private init() {
    // Check if app is already installed
    this.checkInstallation();

    // Listen for beforeinstallprompt event
    window.addEventListener("beforeinstallprompt", (e) => {
      e.preventDefault();
      this.deferredPrompt = e as BeforeInstallPromptEvent;
      this.notifyListeners();
    });

    // Listen for app installation
    window.addEventListener("appinstalled", () => {
      console.log("[PWA] App installed successfully");
      this.isInstalled = true;
      this.deferredPrompt = null;
      this.notifyListeners();
    });

    // Detect if running in standalone mode (already installed)
    if (window.matchMedia("(display-mode: standalone)").matches) {
      this.isInstalled = true;
    }

    // Listen for display mode changes
    window.matchMedia("(display-mode: standalone)").addEventListener("change", (e) => {
      if (e.matches) {
        this.isInstalled = true;
        console.log("[PWA] Entered standalone mode");
      } else {
        this.isInstalled = false;
        console.log("[PWA] Exited standalone mode");
      }
      this.notifyListeners();
    });
  }

  private checkInstallation() {
    // iOS doesn't set display-mode: standalone reliably, so check other indicators
    const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent);
    const isInStandaloneMode = (window.navigator as Navigator & { standalone?: boolean }).standalone === true;
    
    if (isIOS && isInStandaloneMode) {
      this.isInstalled = true;
    } else if (window.matchMedia("(display-mode: standalone)").matches) {
      this.isInstalled = true;
    }
  }

  /**
   * Check if the install prompt is available
   */
  canInstall(): boolean {
    return this.deferredPrompt !== null && !this.isInstalled;
  }

  /**
   * Check if app is already installed
   */
  isAppInstalled(): boolean {
    return this.isInstalled;
  }

  /**
   * Trigger the install prompt
   */
  async install(): Promise<boolean> {
    if (!this.deferredPrompt) {
      console.warn("[PWA] Install prompt not available");
      return false;
    }

    try {
      await this.deferredPrompt.prompt();
      const { outcome } = await this.deferredPrompt.userChoice;
      
      if (outcome === "accepted") {
        console.log("[PWA] User accepted installation");
        this.deferredPrompt = null;
        return true;
      } else {
        console.log("[PWA] User dismissed installation");
        return false;
      }
    } catch (error) {
      console.error("[PWA] Installation failed:", error);
      return false;
    }
  }

  /**
   * Subscribe to installation state changes
   */
  onStateChange(callback: (installed: boolean) => void): () => void {
    this.listeners.add(callback);
    // Call immediately with current state
    callback(this.isInstalled);
    
    // Return unsubscribe function
    return () => {
      this.listeners.delete(callback);
    };
  }

  private notifyListeners() {
    this.listeners.forEach((callback) => callback(this.isInstalled));
  }
}

// Singleton instance
export const pwaInstaller = new PWAInstaller();

/**
 * Register service worker
 */
export async function registerServiceWorker(): Promise<void> {
  if (!("serviceWorker" in navigator)) {
    console.log("[PWA] Service Worker not supported");
    return;
  }

  try {
    const registration = await navigator.serviceWorker.register("/service-worker.js", {
      scope: "/",
    });
    console.log("[PWA] Service Worker registered successfully", registration);
  } catch (error) {
    console.error("[PWA] Service Worker registration failed:", error);
  }
}
