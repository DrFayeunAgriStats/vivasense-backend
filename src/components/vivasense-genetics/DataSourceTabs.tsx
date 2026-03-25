/**
 * DataSourceTabs
 * ==============
 * Wraps existing manual input form and the new multi-trait upload.
 * Drop this around the existing <DynamicInputForm /> in VivaSenseGeneticsPage.
 *
 * Usage in VivaSenseGeneticsPage.tsx:
 *
 *   import { DataSourceTabs } from "./DataSourceTabs";
 *   import { MultiTraitUpload } from "./MultiTraitUpload";
 *
 *   // Replace the bare <DynamicInputForm ... /> with:
 *   <DataSourceTabs
 *     manualContent={<DynamicInputForm ... />}
 *     uploadContent={<MultiTraitUpload />}
 *   />
 */

import React, { useState } from "react";

type TabId = "manual" | "upload";

interface DataSourceTabsProps {
  /** The existing manual input form / component */
  manualContent: React.ReactNode;
  /** The MultiTraitUpload component */
  uploadContent: React.ReactNode;
}

export function DataSourceTabs({ manualContent, uploadContent }: DataSourceTabsProps) {
  const [active, setActive] = useState<TabId>("manual");

  const tabs: { id: TabId; label: string; icon: string; description: string }[] = [
    {
      id: "manual",
      label: "Manual Input",
      icon: "✏️",
      description: "Enter ANOVA values directly",
    },
    {
      id: "upload",
      label: "Upload File",
      icon: "📂",
      description: "CSV / Excel — analyze all traits at once",
    },
  ];

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
          </button>
        ))}
      </div>

      {/* Content */}
      <div>
        <div className={active === "manual" ? "block" : "hidden"}>
          {manualContent}
        </div>
        <div className={active === "upload" ? "block" : "hidden"}>
          {uploadContent}
        </div>
      </div>
    </div>
  );
}
