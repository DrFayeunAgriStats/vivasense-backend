import React, { createContext, useContext, type ReactNode } from "react";
import { GENETICS_API_BASE } from "@/config/vivasense";

const VIVASENSE_BACKENDS = {
  genetics: {
    url: GENETICS_API_BASE,
    label: "Genetics Engine",
  },
} as const;

type ModuleKey = keyof typeof VIVASENSE_BACKENDS;

interface VivaSenseContextValue {
  selectedModule: ModuleKey;
  setSelectedModule: (m: ModuleKey) => void;
  baseURL: string;
  backends: typeof VIVASENSE_BACKENDS;
}

const VivaSenseContext = createContext<VivaSenseContextValue | null>(null);

export function VivaSenseProvider({ children }: { children: ReactNode }) {
  const selectedModule: ModuleKey = "genetics";
  const currentBackend = VIVASENSE_BACKENDS[selectedModule];

  return (
    <VivaSenseContext.Provider
      value={{
        selectedModule,
        setSelectedModule: (_m: ModuleKey) => undefined,
        baseURL: currentBackend.url,
        backends: VIVASENSE_BACKENDS,
      }}
    >
      {children}
    </VivaSenseContext.Provider>
  );
}

export function useVivaSense() {
  const ctx = useContext(VivaSenseContext);
  if (!ctx) throw new Error("useVivaSense must be used within VivaSenseProvider");
  return ctx;
}
