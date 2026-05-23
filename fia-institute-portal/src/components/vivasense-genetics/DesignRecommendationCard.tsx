import React from "react";
import { DesignDetectionResult } from "./designDetection";

interface DesignRecommendationCardProps {
  detection: DesignDetectionResult;
  onAccept: () => void;
  onDismiss: () => void;
}

export function DesignRecommendationCard({
  detection,
  onAccept,
  onDismiss,
}: DesignRecommendationCardProps) {
  const designLabels = {
    crd: "CRD",
    rcbd: "RCBD",
    factorial: "Factorial",
    split_plot_rcbd: "Split-Plot RCBD",
  };

  const designIcons = {
    crd: "🎲",
    rcbd: "🟦",
    factorial: "⚛️",
    split_plot_rcbd: "🔀",
  };

  const confidenceColors = {
    high: "emerald",
    medium: "blue",
    low: "amber",
  };

  const color = confidenceColors[detection.confidence];

  return (
    <div
      className={`rounded-xl border-2 border-${color}-300 bg-${color}-50 p-4 shadow-sm`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <span className="text-3xl" role="img" aria-label="design-icon">
            {designIcons[detection.suggestedDesign]}
          </span>
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <h4 className={`text-sm font-bold text-${color}-900`}>
                Possible {designLabels[detection.suggestedDesign]} Design Detected
              </h4>
              <span
                className={`rounded-full bg-${color}-200 px-2 py-0.5 text-xs font-medium text-${color}-800`}
              >
                {detection.confidence} confidence
              </span>
            </div>
            <ul className={`mt-2 space-y-1 text-xs text-${color}-700`}>
              {detection.reasons.map((reason, idx) => (
                <li key={idx} className="flex items-start gap-1.5">
                  <span className="mt-0.5 text-[10px]">•</span>
                  <span>{reason}</span>
                </li>
              ))}
            </ul>
            {detection.detectedFactors.possibleMainPlot &&
              detection.detectedFactors.possibleSubplot && (
                <div className={`mt-2 rounded-lg bg-${color}-100 px-3 py-2 text-xs`}>
                  <p className={`font-semibold text-${color}-900`}>
                    Split-plot structure identified:
                  </p>
                  <p className={`mt-1 text-${color}-700`}>
                    <span className="font-medium">Main-plot:</span>{" "}
                    {detection.detectedFactors.possibleMainPlot.join(", ")}
                  </p>
                  <p className={`text-${color}-700`}>
                    <span className="font-medium">Subplot:</span>{" "}
                    {detection.detectedFactors.possibleSubplot.join(", ")}
                  </p>
                </div>
              )}
          </div>
        </div>
        <button
          type="button"
          onClick={onDismiss}
          className="text-gray-400 hover:text-gray-600"
          aria-label="Dismiss"
        >
          <svg
            className="h-5 w-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      </div>
      <div className="mt-3 flex gap-2">
        <button
          type="button"
          onClick={onAccept}
          className={`rounded-lg bg-${color}-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-${color}-700 transition-colors`}
        >
          Use {designLabels[detection.suggestedDesign]}
        </button>
        <button
          type="button"
          onClick={onDismiss}
          className="rounded-lg border border-gray-300 bg-white px-3 py-1.5 text-xs font-medium text-gray-700 hover:bg-gray-50 transition-colors"
        >
          Choose manually
        </button>
      </div>
    </div>
  );
}
