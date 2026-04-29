/**
 * VivaSense API configuration — single source of truth for backend URL.
 *
 * All genetics API clients import API_BASE from this file.
 * Do NOT read from env vars here: if VITE_GENETICS_ENGINE_BASE is set
 * in Lovable/Vercel to the frontend host it will silently override the
 * fallback and send requests to the wrong server.
 */

export const API_BASE = import.meta.env.VITE_API_URL || "https://vivasense-genetics-docker.onrender.com";
