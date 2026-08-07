export type VivaSenseMode = "free" | "pro";

export const VIVASENSE_MODE_KEY = "vivasense_mode";
export const VIVASENSE_DEFAULT_MODE: VivaSenseMode = "free";
export const BOOK_DATA_CLINIC_URL = "https://wa.me/2349022158026?text=Hello%20VivaSense%2C%20I%20want%20to%20book%20a%20Data%20Clinic%20session.";
export const VIVASENSE_MODE_CHANGED_EVENT = "vivasense-mode-changed";

function canUseLocalStorage(): boolean {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

export function initializeVivaSenseMode(): VivaSenseMode {
  if (!canUseLocalStorage()) return VIVASENSE_DEFAULT_MODE;
  const existing = window.localStorage.getItem(VIVASENSE_MODE_KEY);
  if (existing === "free" || existing === "pro") {
    return existing;
  }
  window.localStorage.setItem(VIVASENSE_MODE_KEY, VIVASENSE_DEFAULT_MODE);
  return VIVASENSE_DEFAULT_MODE;
}

export function getVivaSenseMode(): VivaSenseMode {
  return initializeVivaSenseMode();
}

export function setVivaSenseMode(mode: VivaSenseMode): void {
  if (!canUseLocalStorage()) return;
  window.localStorage.setItem(VIVASENSE_MODE_KEY, mode);
  window.dispatchEvent(new Event(VIVASENSE_MODE_CHANGED_EVENT));
}

export function activateProWithCode(code: string): boolean {
  void code;
  return false;
}

export function isProMode(): boolean {
  return getVivaSenseMode() === "pro";
}

export function modeLabel(mode: VivaSenseMode): string {
  return mode === "pro" ? "Pro Mode" : "Free Mode";
}

export class ProFeatureError extends Error {
  code: string;
  featureName: string;

  constructor(featureName: string) {
    super("Upgrade to access this feature");
    this.name = "ProFeatureError";
    this.code = "PRO_FEATURE";
    this.featureName = featureName;
  }
}

export function guardProModule(moduleName: string): void {
  const mode = getVivaSenseMode();
  if (mode !== "pro") {
    console.log(`[PRO GUARD] mode = free, blocked module = ${moduleName}`);
    throw new ProFeatureError(moduleName);
  }
  console.log(`[PRO GUARD] mode = pro, allowed module = ${moduleName}`);
}

export function ensureProAccess(featureName: string): void {
  guardProModule(featureName);
}

/**
 * Identity used by the backend pilot-access gate.
 *
 * This is deliberately separate from VIVASENSE_MODE_KEY: the pilot gate is
 * checked against a server-side allowlist and must not depend on the
 * free/pro mode value, which is currently "pro" for every account.
 */
export const VIVASENSE_USER_KEY = "vivasense_user";

export function setVivaSenseUser(identity: string | null): void {
  if (!canUseLocalStorage()) return;
  const trimmed = identity?.trim();
  if (trimmed) {
    window.localStorage.setItem(VIVASENSE_USER_KEY, trimmed);
  } else {
    window.localStorage.removeItem(VIVASENSE_USER_KEY);
  }
}

export function getVivaSenseUser(): string | null {
  if (!canUseLocalStorage()) return null;
  const stored = window.localStorage.getItem(VIVASENSE_USER_KEY)?.trim();
  return stored ? stored : null;
}

export function buildModeHeaders(baseHeaders?: HeadersInit): Headers {
  const headers = new Headers(baseHeaders);
  headers.set("X-VivaSense-Mode", getVivaSenseMode());
  const identity = getVivaSenseUser();
  if (identity) {
    headers.set("X-VivaSense-User", identity);
  }
  return headers;
}
