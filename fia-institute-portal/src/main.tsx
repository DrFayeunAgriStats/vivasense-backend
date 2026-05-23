import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";
import { getVivaSenseMode } from "./lib/vivasenseGating";
import { registerServiceWorker } from "./utils/pwa-install";
import { HelmetProvider } from "react-helmet-async";
import React from "react";

getVivaSenseMode();
registerServiceWorker();

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <HelmetProvider>
      <App />
    </HelmetProvider>
  </React.StrictMode>
);
