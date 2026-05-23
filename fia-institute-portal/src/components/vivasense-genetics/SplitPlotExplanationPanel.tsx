import React, { useState } from "react";

export function SplitPlotExplanationPanel() {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-lg border border-blue-200 bg-blue-50/60">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-3 py-2 text-left"
      >
        <span className="text-sm font-medium text-blue-800">How split-plot designs work</span>
        <span className={`text-xs text-blue-600 transition-transform ${open ? "rotate-90" : ""}`}>▶</span>
      </button>
      {open && (
        <div className="space-y-2 border-t border-blue-200 px-4 py-3 text-xs leading-relaxed text-blue-900">
          <p>
            Split-plot designs have <strong>two sizes of experimental unit</strong>:
          </p>
          <p>
            <strong>Whole plots (main plots)</strong>: larger units randomised within each replication.
            The main plot factor is applied here. These are tested with <em>lower</em> statistical precision.
          </p>
          <p>
            <strong>Sub plots</strong>: smaller units nested within each main plot. The sub-plot factor
            is applied here. These are tested with <em>higher</em> statistical precision.
          </p>
          <p>This means two separate error terms are used:</p>
          <ul className="ml-4 list-disc">
            <li><strong>Error A</strong> — for the main plot factor</li>
            <li><strong>Error B</strong> — for the sub-plot factor and interaction</li>
          </ul>
          <p>
            Applying a single pooled error to both is <strong>incorrect</strong> and will produce wrong
            F-tests and p-values.
          </p>
        </div>
      )}
    </div>
  );
}
