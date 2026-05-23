/**
 * Returns significance stars based on p-value (agricultural/breeding journal standard).
 */
export function getSignificanceStars(pValue: unknown): string {
  const p = typeof pValue === "number" ? pValue : parseFloat(String(pValue));
  if (isNaN(p)) return "";
  if (p < 0.001) return "***";
  if (p < 0.01) return "**";
  if (p < 0.05) return "*";
  return "ns";
}

/** Format p-value for journal display: e.g. <0.001, 0.002, 0.047 */
export function formatPValue(pValue: unknown): string {
  const p = typeof pValue === "number" ? pValue : parseFloat(String(pValue));
  if (isNaN(p)) return "—";
  if (p < 0.001) return "<0.001";
  return p.toFixed(3);
}

export function SignificanceLegend() {
  return (
    <p className="text-xs text-muted-foreground mt-3 italic">
      Significance codes: '***' p &lt; 0.001, '**' p &lt; 0.01, '*' p &lt; 0.05, 'ns' not significant (α = 0.05)
    </p>
  );
}
