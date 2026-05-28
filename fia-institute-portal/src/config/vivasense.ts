/**
 * VivaSense backend URL configuration.
 * Single source of truth for all API base URLs.
 */

/** ANOVA backend — now served by the unified Docker genetics engine */
export const ANOVA_API_BASE = import.meta.env.VITE_API_URL || "https://vivasense-backend-r-production.up.railway.app";

/** Genetics backend — Docker-based genetics engine */
export const GENETICS_API_BASE = import.meta.env.VITE_API_URL || "https://vivasense-backend-r-production.up.railway.app";

/** @deprecated Use ANOVA_API_BASE instead */
export const ANOVA_BASE = ANOVA_API_BASE;

