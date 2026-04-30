/**
 * DataSourceTabs
 * ==============
 * Wraps manual input, file upload, and (optionally) trait relationships.
 *
 * Dataset context sharing
 * -----------------------
 * When `traitRelationshipsContent` is provided, this component holds
 * `datasetContext` in state.  It clones `uploadContent` and injects an
 * `onDatasetReady` prop so that MultiTraitUpload can signal when a dataset
 * is ready — without the parent page needing to manage that state.
 *
 * Usage in VivaSenseGeneticsPage.tsx:
 *
 *   import { DataSourceTabs }     from "./DataSourceTabs";
 *   import { MultiTraitUpload }   from "./MultiTraitUpload";
 *   import { TraitRelationships } from "./TraitRelationships";
 *
 *   // Replace the bare <DynamicInputForm ... /> with:
 *   <DataSourceTabs
 *     manualContent={<DynamicInputForm ... />}
 *     uploadContent={<MultiTraitUpload />}
 *     traitRelationshipsContent={(ctx) => <TraitRelationships datasetContext={ctx} />}
 *   />
 *
 * The `traitRelationshipsContent` prop is optional.  When omitted the third
 * tab does not appear and the component behaves exactly as before.
 */

import React, { useEffect, useState } from "react";
import { UploadDatasetContext } from "@/services/geneticsUploadApi";
import { FileStatusInfo } from "@/components/vivasense-genetics/MultiTraitUpload";
import { FieldLayoutGenerator } from "@/components/vivasense-genetics/FieldLayoutGenerator";
import {
  getVivaSenseMode,
  initializeVivaSenseMode,
  modeLabel,
  VIVASENSE_MODE_CHANGED_EVENT,
  VivaSenseMode,
} from "@/services/featureMode";

type TabId = "field-layout" | "manual" | "upload" | "relationships" | "descriptive";

interface DataSourceTabsProps {
  /** The existing manual input form / component */
  manualContent: React.ReactNode;
  /**
   * The MultiTraitUpload component (or any element that accepts an optional
   * `onDatasetReady` prop).  DataSourceTabs injects `onDatasetReady` via
   * React.cloneElement to capture the dataset context.
   */
  uploadContent: React.ReactElement;
  /**
   * Render-prop for the Trait Relationships tab.
   * Receives the current dataset context (null until the user confirms a
   * column mapping in the Upload File tab).
   * When omitted, the third tab is not rendered.
   */
  traitRelationshipsContent?: (ctx: UploadDatasetContext | null) => React.ReactNode;
  /**
   * Render-prop for the Descriptive Statistics tab.
   * Receives the current dataset context including dataset_token.
   * When omitted, the Descriptive Statistics tab is not rendered.
   */
  descriptiveStatsContent?: (ctx: UploadDatasetContext | null) => React.ReactNode;
}

export function DataSourceTabs({
  manualContent,
  uploadContent,
  traitRelationshipsContent,
  descriptiveStatsContent,
}: DataSourceTabsProps) {
  const [active, setActive] = useState<TabId>("field-layout");
  const [datasetContext, setDatasetContext] = useState<UploadDatasetContext | null>(null);
  const [mode, setMode] = useState<VivaSenseMode>(() => initializeVivaSenseMode());
  const [fileStatus, setFileStatus] = useState<FileStatusInfo>({ state: "none" });

  useEffect(() => {
    // Hard-refresh guard: initialize only when key is missing; keep existing "pro".
    initializeVivaSenseMode();

    const syncMode = () => setMode(getVivaSenseMode());
    window.addEventListener(VIVASENSE_MODE_CHANGED_EVENT, syncMode);
    window.addEventListener("storage", syncMode);
    return () => {
      window.removeEventListener(VIVASENSE_MODE_CHANGED_EVENT, syncMode);
      window.removeEventListener("storage", syncMode);
    };
  }, []);

  const showRelationships = typeof traitRelationshipsContent === "function";
  const showDescriptive = typeof descriptiveStatsContent === "function";

  type TabDef = {
    id: TabId;
    label: string;
    icon: string;
    description: string;
    badge?: "pro" | "free" | "new";
  };

  const tabs: TabDef[] = [
    {
      id: "field-layout",
      label: "Field Layout",
      icon: "🌾",
      description: "Generate CRD / RCBD field plans",
      badge: "free",
    },
    {
      id: "manual",
      label: "Single Trait",
      icon: "✏️",
      description: "Enter data for one trait",
      badge: "free",
    },
    {
      id: "upload",
      label: "Multi-Trait File",
      icon: "📂",
      description: "CSV / Excel — batch analysis",
      badge: "free",
    },
    ...(showRelationships
      ? [
          {
            id: "relationships" as TabId,
            label: "Trait Relationships",
            icon: "🔗",
            description: "Heatmap, correlation, PCA",
            badge: "pro" as const,
          },
        ]
      : []),
    ...(showDescriptive
      ? [
          {
            id: "descriptive" as TabId,
            label: "Descriptive Stats",
            icon: "📊",
            description: "Summary statistics per trait",
            badge: "free" as const,
          },
        ]
      : []),
  ];

  // Inject onDatasetReady and onFileStatus into uploadContent so MultiTraitUpload
  // can signal when the dataset state changes without requiring extra prop drilling.
  const uploadWithCallback = React.cloneElement(uploadContent, {
    onDatasetReady: (ctx: UploadDatasetContext) => setDatasetContext(ctx),
    onFileStatus:   (info: FileStatusInfo) => setFileStatus(info),
  });

  return (
    <div className="w-full">
      {/* ── Top bar: dataset status + mode badge ── */}
      <div className="mb-4 rounded-2xl border border-gray-100 bg-white p-4 shadow-sm">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <DatasetStatusBar status={fileStatus} />
          <span className="shrink-0 inline-flex items-center rounded-full border border-emerald-300 bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-800">
            {modeLabel(mode)}
          </span>
        </div>
      </div>

      {/* ── Main layout: sidebar (lg+) | content ── */}
      <div className="flex flex-col lg:flex-row gap-4">

        {/* Sidebar navigation */}
        <nav
          aria-label="Module navigation"
          className="
            lg:w-52 xl:w-60 shrink-0
            rounded-2xl border border-gray-200 bg-gradient-to-b from-gray-50 to-white
            p-2 flex flex-col gap-1
            /* mobile: horizontal scrollable strip */
            flex-row lg:flex-col overflow-x-auto lg:overflow-x-visible
          "
        >
          {tabs.map((tab) => {
            const isActive = active === tab.id;
            const hasDataDot =
              (tab.id === "relationships" || tab.id === "descriptive") &&
              datasetContext !== null &&
              !isActive;

            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActive(tab.id)}
                className={[
                  "group flex shrink-0 items-center gap-2.5 rounded-xl px-3 py-2.5 text-left transition-all",
                  "lg:w-full",
                  isActive
                    ? "bg-white text-emerald-700 shadow-sm ring-1 ring-emerald-200"
                    : "text-gray-500 hover:bg-white/70 hover:text-gray-700",
                ].join(" ")}
              >
                <span className="text-lg shrink-0">{tab.icon}</span>
                <span className="flex-1 min-w-0">
                  <span className="block text-sm font-medium leading-tight truncate">{tab.label}</span>
                  <span className="hidden lg:block text-xs text-gray-400 leading-snug mt-0.5 truncate">
                    {tab.description}
                  </span>
                </span>
                {/* Badges */}
                <span className="flex items-center gap-1 shrink-0">
                  {hasDataDot && (
                    <span className="h-2 w-2 rounded-full bg-emerald-500" aria-label="Data ready" />
                  )}
                  {tab.badge === "pro" && mode !== "pro" && (
                    <span className="inline-block rounded-full bg-violet-100 px-1.5 py-0.5 text-[10px] font-semibold text-violet-700 leading-none">
                      Pro
                    </span>
                  )}
                  {tab.badge === "pro" && mode === "pro" && (
                    <span className="inline-block rounded-full bg-emerald-100 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-700 leading-none">
                      ✓
                    </span>
                  )}
                </span>
              </button>
            );
          })}
        </nav>

        {/* Content pane */}
        <div className="flex-1 min-w-0">
          <div className={active === "field-layout" ? "block" : "hidden"}>
            <FieldLayoutGenerator />
          </div>
          <div className={active === "manual" ? "block" : "hidden"}>
            {manualContent}
          </div>
          <div className={active === "upload" ? "block" : "hidden"}>
            {uploadWithCallback}
          </div>
          {showRelationships && (
            <div className={active === "relationships" ? "block" : "hidden"}>
              {traitRelationshipsContent(datasetContext)}
            </div>
          )}
          {showDescriptive && (
            <div className={active === "descriptive" ? "block" : "hidden"}>
              {descriptiveStatsContent(datasetContext)}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Dataset status bar
// ─────────────────────────────────────────────────────────────────────────────

function DatasetStatusBar({ status }: { status: FileStatusInfo }) {
  if (status.state === "loaded") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs text-emerald-800">
        <span aria-hidden="true">✓</span>
        <span className="font-medium">{status.filename}</span>
        <span className="text-emerald-500">|</span>
        <span>{status.n_rows?.toLocaleString()} rows</span>
        <span className="text-emerald-500">·</span>
        <span>{status.n_columns} columns</span>
      </span>
    );
  }

  if (status.state === "invalid") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full border border-red-200 bg-red-50 px-3 py-1.5 text-xs text-red-700">
        <span aria-hidden="true">✕</span>
        File could not be read — try another file
      </span>
    );
  }

  // state === "none"
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-200 bg-amber-50 px-3 py-1.5 text-xs text-amber-700">
      <span aria-hidden="true">⚠</span>
      No dataset loaded — Upload to begin
    </span>
  );
}
