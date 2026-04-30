/**
 * VsSpinner
 * =========
 * Branded VivaSense loading spinner.
 * Uses a dual-ring pulse animation in the emerald brand palette.
 *
 * Sizes: sm (h-4 w-4) | md (h-8 w-8, default) | lg (h-12 w-12)
 */

import React from "react";

interface VsSpinnerProps {
  size?: "sm" | "md" | "lg";
  className?: string;
  /** Whether to show "VS" monogram in centre (md and lg only) */
  monogram?: boolean;
}

const sizeMap = {
  sm: { outer: "h-4 w-4", border: "border-2", inner: "h-2 w-2 border", text: "" },
  md: { outer: "h-8 w-8", border: "border-[3px]", inner: "h-4 w-4 border-2", text: "text-[8px]" },
  lg: { outer: "h-12 w-12", border: "border-4", inner: "h-6 w-6 border-2", text: "text-[10px]" },
};

export function VsSpinner({ size = "md", className = "", monogram = false }: VsSpinnerProps) {
  const s = sizeMap[size];

  return (
    <span
      role="status"
      aria-label="Loading…"
      className={`relative inline-flex items-center justify-center ${s.outer} ${className}`}
    >
      {/* Outer ring */}
      <span
        className={`absolute inset-0 rounded-full ${s.border} border-emerald-600 border-t-transparent animate-spin`}
      />
      {/* Inner faint ring — counter-rotates for branded look */}
      <span
        className={`rounded-full ${s.inner} border-emerald-200 border-b-transparent animate-[spin_1.4s_linear_infinite_reverse]`}
      />
      {/* "VS" monogram — visible only md+ when requested */}
      {monogram && size !== "sm" && (
        <span
          className={`absolute font-bold text-emerald-700 leading-none select-none ${s.text}`}
          aria-hidden="true"
        >
          VS
        </span>
      )}
    </span>
  );
}
