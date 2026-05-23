/**
 * DataSourceTabs
 * ==============
 * Wraps the shared upload workflow and downstream analysis modules.
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
 *   // Mount the shared upload workflow with downstream analysis modules:
 *   <DataSourceTabs
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
import { ProFeatureModal } from "@/components/vivasense-genetics/ProFeatureModal";
import {
  getVivaSenseMode,
  initializeVivaSenseMode,
  modeLabel,
  VIVASENSE_MODE_CHANGED_EVENT,
  VivaSenseMode,
} from "@/services/featureMode";

type TabId = "upload" | "field-data" | "anova" | "genetics" | "relationships" | "descriptive" | "advanced";

interface DataSourceTabsProps {
  /**
   * The MultiTraitUpload component (or any element that accepts an optional
   * `onDatasetReady` prop).  DataSourceTabs injects `onDatasetReady` via
   * React.cloneElement to capture the dataset context.
   */
  uploadContent: React.ReactElement;
  /**
    * Render content for the Field Data tab.
    * Free offline plot-level field measurement capture.
    */
    fieldDataContent?: React.ReactNode;
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
  /**
   * Render-prop for the ANOVA tab.
   * Receives the current dataset context after upload/confirmation.
   */
  anovaContent?: (ctx: UploadDatasetContext | null) => React.ReactNode;
  /**
   * Render-prop for the Genetic Parameters tab.
   * Receives the current dataset context after upload/confirmation.
   */
  geneticsContent?: (ctx: UploadDatasetContext | null) => React.ReactNode;
  /**
   * Render-prop for the Advanced Analysis tab.
   * Receives the current dataset context after upload/confirmation.
   */
  advancedContent?: (ctx: UploadDatasetContext | null) => React.ReactNode;
}

export function DataSourceTabs({
  uploadContent,
  fieldDataContent,
  traitRelationshipsContent,
  descriptiveStatsContent,
  anovaContent,
  geneticsContent,
  advancedContent,
}: DataSourceTabsProps) {
  const [active, setActive] = useState<TabId>(() =>
    fieldDataContent !== undefined && fieldDataContent !== null ? "field-data" : "upload"
  );
  const [datasetContext, setDatasetContext] = useState<UploadDatasetContext | null>(null);
  const [mode, setMode] = useState<VivaSenseMode>(() => initializeVivaSenseMode());
  const [fileStatus, setFileStatus] = useState<FileStatusInfo>({ state: "none" });
  const [proModalFeature, setProModalFeature] = useState<string | null>(null);

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
  const showFieldData = fieldDataContent !== undefined && fieldDataContent !== null;
  const showDescriptive = typeof descriptiveStatsContent === "function";
  const showAnova = typeof anovaContent === "function";
  const showGenetics = typeof geneticsContent === "function";
  const showAdvanced = typeof advancedContent === "function";

  type TabDef = {
    id: TabId;
    label: string;
    icon: string;
    description: string;
    methodology?: string[];
    requiresPro: boolean;
    badge?: string;
    badgeColor?: "green" | "violet";
  };

  const tabs: TabDef[] = [
    ...(showFieldData
      ? [
          {
            id: "field-data" as TabId,
            label: "Field Data",
            icon: "📋",
            description: "Plot-level capture and trial setup",
            methodology: ["CRD", "RCBD", "Split-Plot"],
            requiresPro: false,
            badge: "Free",
            badgeColor: "green" as const,
          },
        ]
      : []),
    {
      id: "upload",
      label: "Experimental Dataset",
      icon: "📂",
      description: "CSV / Excel ingestion with structure detection",
      methodology: ["CRD", "RCBD", "MET"],
      requiresPro: false,
      badge: "Free",
      badgeColor: "green",
    },
    ...(showDescriptive
      ? [
          {
            id: "descriptive" as TabId,
            label: "Descriptive Statistics",
            icon: "📊",
            description: "Trait distribution and summary diagnostics",
            methodology: ["QC", "Screening"],
            requiresPro: false,
            badge: "Free",
            badgeColor: "green" as const,
          },
        ]
      : []),
    ...(showAnova
      ? [
          {
            id: "anova" as TabId,
            label: "ANOVA",
            icon: "🧮",
            description: "Design-aware tests for treatment effects",
            methodology: ["CRD", "RCBD", "Split-Plot"],
            requiresPro: false,
            badge: "Free",
            badgeColor: "green" as const,
          },
        ]
      : []),
    ...(showGenetics
      ? [
          {
            id: "genetics" as TabId,
            label: datasetContext?.research_domain === "agronomy"
              ? "Variance Components"
              : "Genetic Parameters",
            icon: "🧬",
            description: "Heritability, GCV, PCV, GAM",
            methodology: ["RCBD", "MET"],
            requiresPro: true,
            badge: "Pro",
            badgeColor: "violet" as const,
          },
        ]
      : []),
    ...(showRelationships
      ? [
          {
            id: "relationships" as TabId,
            label: "Trait Relationships",
            icon: "🔗",
            description: "Correlation matrices, heatmaps, and trend structure",
            methodology: ["Pearson", "Spearman", "PCA"],
            requiresPro: true,
            badge: "Pro",
            badgeColor: "violet" as const,
          },
        ]
      : []),
    ...(showAdvanced
      ? [
          {
            id: "advanced" as TabId,
            label: "Advanced Analysis",
            icon: "🚀",
            description: "GGE biplot, AMMI, BLUP, PCA, and stability",
            methodology: ["GGE", "AMMI", "PCA", "MET"],
            requiresPro: true,
            badge: "Pro",
            badgeColor: "violet" as const,
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

  const handleTabClick = (tab: TabDef) => {
    if (tab.requiresPro && mode !== "pro") {
      setProModalFeature(tab.label);
      return;
    }
    setActive(tab.id);
  };

  return (
    <div className="w-full">
      <div className="mb-6 rounded-3xl border border-gray-200 bg-white px-5 py-4 shadow-sm lg:px-6">
        <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
          <div className="space-y-2">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-gray-500">
              Scientific Workflow Status
            </p>
            <DatasetStatusBar status={fileStatus} datasetContext={datasetContext} />
            <p className="text-sm text-gray-600">
              Recommended sequence: field data, experimental dataset, descriptive statistics, ANOVA, trait relationships, genetic parameters, then advanced analysis.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2 xl:justify-end">
            <span className="shrink-0 inline-flex items-center rounded-full border border-emerald-300 bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-800">
              {modeLabel(mode)}
            </span>
            <span className="inline-flex items-center rounded-full border border-gray-200 bg-gray-50 px-3 py-1 text-xs font-medium text-gray-700">
              Recommended workflow
            </span>
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-6 xl:flex-row">

        <nav
          aria-label="Module navigation"
          className="
            xl:w-[295px] shrink-0
            rounded-3xl border border-gray-200 bg-white
            p-3 flex flex-col gap-2
            flex-row xl:flex-col overflow-x-auto xl:overflow-x-visible
            shadow-sm
          "
        >
          <div className="hidden xl:block rounded-2xl border border-emerald-100 bg-emerald-50/60 px-4 py-3">
            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-emerald-800">
              Analysis Modules
            </p>
            <p className="mt-1 text-xs leading-relaxed text-emerald-700/90">
              Ordered to match agricultural research flow from data intake and screening to inferential and GxE-focused analysis.
            </p>
          </div>

          {tabs.map((tab) => {
            const isActive = active === tab.id;
            const hasDataDot =
              (tab.id === "descriptive" || tab.id === "anova" || tab.id === "genetics" || tab.id === "relationships" || tab.id === "advanced") &&
              datasetContext !== null &&
              !isActive;

            return (
              <button
                key={tab.id}
                type="button"
                title={tab.label}
                onClick={() => handleTabClick(tab)}
                className={[
                  "group flex shrink-0 items-start gap-3 rounded-2xl px-3.5 py-3 text-left transition-all",
                  "xl:w-full min-w-[220px] xl:min-w-0",
                  isActive
                    ? "bg-emerald-50 text-emerald-800 shadow-sm ring-1 ring-emerald-200"
                    : "text-gray-600 hover:bg-gray-50 hover:text-gray-800",
                ].join(" ")}
              >
                <span className={[
                  "mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border text-lg",
                  isActive
                    ? "border-emerald-200 bg-white text-emerald-700"
                    : "border-gray-200 bg-gray-50 text-gray-500 group-hover:border-emerald-200 group-hover:text-emerald-700",
                ].join(" ")}>{tab.icon}</span>
                <span className="flex-1 min-w-0">
                  <span className="block text-sm font-semibold leading-tight">{tab.label}</span>
                  <span className="hidden xl:block text-xs text-gray-500 leading-snug mt-1">
                    {tab.description}
                  </span>
                  {tab.methodology && tab.methodology.length > 0 && (
                    <span className="mt-2 hidden xl:flex flex-wrap gap-1.5">
                      {tab.methodology.map((tag) => (
                        <span
                          key={`${tab.id}-${tag}`}
                          className="rounded-full border border-gray-200 bg-white px-2 py-0.5 text-[10px] font-medium tracking-wide text-gray-600"
                        >
                          {tag}
                        </span>
                      ))}
                    </span>
                  )}
                </span>
                <span className="flex items-center gap-1 shrink-0">
                  {hasDataDot && (
                    <span className="h-2 w-2 rounded-full bg-emerald-500" aria-label="Data ready" />
                  )}
                  {tab.requiresPro && mode === "pro" && (
                    <span className="inline-block rounded-full bg-emerald-100 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-700 leading-none">
                      ✓
                    </span>
                  )}
                  {tab.badge && (
                    <span
                      className={[
                        "inline-block rounded-full px-1.5 py-0.5 text-[10px] font-semibold leading-none",
                        tab.badgeColor === "violet" ? "bg-violet-100 text-violet-700" : "bg-emerald-100 text-emerald-700",
                      ].join(" ")}
                    >
                      {tab.badge}
                    </span>
                  )}
                </span>
              </button>
            );
          })}
        </nav>

        <div className="flex-1 min-w-0">
          <div className={active === "upload" ? "block" : "hidden"}>
            {uploadWithCallback}
          </div>
          {showFieldData && (
            <div className={active === "field-data" ? "block" : "hidden"}>
              {fieldDataContent}
            </div>
          )}
          {showAnova && (
            <div className={active === "anova" ? "block" : "hidden"}>
              {anovaContent(datasetContext)}
            </div>
          )}
          {showGenetics && (
            <div className={active === "genetics" ? "block" : "hidden"}>
              {geneticsContent(datasetContext)}
            </div>
          )}
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
          {showAdvanced && (
            <div className={active === "advanced" ? "block" : "hidden"}>
              {advancedContent(datasetContext)}
            </div>
          )}
        </div>
      </div>

      <ProFeatureModal
        open={proModalFeature !== null}
        featureName={proModalFeature ?? undefined}
        onClose={() => setProModalFeature(null)}
        onActivated={() => {
          if (proModalFeature) {
            const unlocked = tabs.find((tab) => tab.label === proModalFeature);
            if (unlocked) setActive(unlocked.id);
          }
        }}
      />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Dataset status bar
// ─────────────────────────────────────────────────────────────────────────────

function DatasetStatusBar({
  status,
  datasetContext,
}: {
  status: FileStatusInfo;
  datasetContext: UploadDatasetContext | null;
}) {
  if (status.state === "loaded") {
    return (
      <div className="flex flex-wrap items-center gap-2 text-sm text-emerald-900">
        <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-800">
          <span aria-hidden="true">✓</span>
          Dataset loaded
        </span>
        <span className="font-medium text-gray-800">{status.filename}</span>
        <span className="text-gray-500">{status.n_rows?.toLocaleString()} rows</span>
        <span className="text-gray-400">·</span>
        <span className="text-gray-500">{status.n_columns} columns</span>
        {datasetContext?.mode && (
          <>
            <span className="text-gray-400">·</span>
            <span className="text-gray-500">{datasetContext.mode === "multi" ? "MET structure detected" : "Single-environment structure detected"}</span>
          </>
        )}
      </div>
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
