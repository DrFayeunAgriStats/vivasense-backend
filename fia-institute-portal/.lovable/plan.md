

## Add Visit Trends Line Chart to the Visits Tab

### Overview
Add a Recharts-based line chart to the Visits tab that shows daily visit counts over time, respecting the existing date range filter.

### Changes

**File: `src/components/bgm/BgmAdminDashboard.tsx`**

1. **Add imports** for Recharts components (`LineChart`, `Line`, `XAxis`, `YAxis`, `CartesianGrid`, `Tooltip`, `ResponsiveContainer`) and the existing `ChartContainer` / `ChartTooltip` / `ChartTooltipContent` from `@/components/ui/chart`.

2. **Add a `dailyTrend` computed value** inside the existing `useMemo` block that derives filtered visits:
   - Group filtered visits by date (formatted as `MMM dd`)
   - Sort chronologically
   - Return an array of `{ date: string, visits: number }` objects

3. **Add the line chart card** between the existing two-column grid (Page Visit Summary + Quick Stats) and the full table, inside the Visits `TabsContent`:
   - A full-width `Card` with title "Visit Trends"
   - Uses `ChartContainer` with a `ResponsiveContainer` wrapping a `LineChart`
   - Single `Line` showing daily visit count with a smooth curve
   - X-axis shows dates, Y-axis shows visit count
   - Includes chart tooltip for hover details
   - Fixed height of ~300px

### Technical Details

- Uses the project's existing Recharts dependency and shadcn chart components
- The trend data is derived from the already-filtered `rawVisits` array, so it automatically respects the selected date range filter (7d, 30d, all, custom)
- Uses `date-fns` `format` to group visits by day
- Chart colors use CSS variables (`hsl(var(--primary))`) to match the theme

