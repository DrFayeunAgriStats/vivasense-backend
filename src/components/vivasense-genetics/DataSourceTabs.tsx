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

import React, { useState } from "react";
import { UploadDatasetContext } from "@/services/geneticsUploadApi";

type TabId = "manual" | "upload" | "relationships" | "descriptive";

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
  const [active, setActive] = useState<TabId>("manual");
  const [datasetContext, setDatasetContext] = useState<UploadDatasetContext | null>(null);

  const showRelationships = typeof traitRelationshipsContent === "function";
  const showDescriptive = typeof descriptiveStatsContent === "function";

  type TabDef = { id: TabId; label: string; icon: string; description: string };

  const tabs: TabDef[] = [
    {
      id: "manual",
      label: "Single Trait",
      icon: "✏️",
      description: "Enter data for one trait",
    },
    {
      id: "upload",
      label: "Multi-Trait File",
      icon: "📂",
      description: "CSV / Excel — analyze all traits at once",
    },
    ...(showRelationships
      ? [
          {
            id: "relationships" as TabId,
            label: "Trait Relationships",
            icon: "🔗",
            description: "Correlations between traits",
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
          },
        ]
      : []),
  ];

  // Inject onDatasetReady into uploadContent so MultiTraitUpload can signal
  // when the user has confirmed a column mapping.
  const uploadWithCallback = React.cloneElement(uploadContent, {
    onDatasetReady: (ctx: UploadDatasetContext) => setDatasetContext(ctx),
  });

  return (
    <div className="w-full">
      {/* Tab bar */}
      <div className="flex rounded-xl border border-gray-200 bg-gray-50 p-1 gap-1 mb-6">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActive(tab.id)}
            className={[
              "flex flex-1 items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium transition-all",
              active === tab.id
                ? "bg-white text-emerald-700 shadow-sm ring-1 ring-gray-200"
                : "text-gray-500 hover:text-gray-700",
            ].join(" ")}
          >
            <span className="text-base">{tab.icon}</span>
            <span className="hidden sm:inline">{tab.label}</span>
            <span className="hidden md:inline text-xs font-normal text-gray-400">
              — {tab.description}
            </span>
            {/* Badge: shows when a dataset is ready and this tab is inactive */}
            {(tab.id === "relationships" || tab.id === "descriptive") &&
              datasetContext !== null &&
              active !== tab.id && (
                <span className="ml-1 h-2 w-2 rounded-full bg-emerald-500 shrink-0" />
              )}
          </button>
        ))}
      </div>

      {/* Content — all panes stay mounted (display:block/hidden) */}
      <div>
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
  );
}
