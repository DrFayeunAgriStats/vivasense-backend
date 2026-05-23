import { ClipboardList, Database, BarChart3, FileText, Download } from "lucide-react";

const workflowSteps = [
  {
    icon: ClipboardList,
    title: "Design",
    description: "Create randomized experimental layouts, treatment structures, and field plans.",
  },
  {
    icon: Database,
    title: "Upload",
    description: "Import field data with intelligent column mapping and design detection.",
  },
  {
    icon: BarChart3,
    title: "Analyze",
    description: "Run ANOVA, PCA, GGE biplot, AMMI, and methodology-aware statistical workflows.",
  },
  {
    icon: FileText,
    title: "Interpret",
    description: "Surface assumptions, diagnostics, and methodology-aware interpretations.",
  },
  {
    icon: Download,
    title: "Export",
    description: "Publication-ready Word reports with formatted tables and figures.",
  },
];

export function ResearchWorkflow() {
  return (
    <section className="relative py-24 lg:py-32 bg-[#0b1d14] overflow-hidden">
      <div
        className="absolute inset-0 opacity-[0.05] pointer-events-none"
        style={{
          backgroundImage:
            "linear-gradient(rgba(52,211,153,0.6) 1px, transparent 1px), linear-gradient(90deg, rgba(52,211,153,0.6) 1px, transparent 1px)",
          backgroundSize: "64px 64px",
        }}
      />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[400px] rounded-full bg-emerald-500/10 blur-[140px] pointer-events-none" />

      <div className="container-wide relative">
        <div className="text-center mb-20 max-w-3xl mx-auto">
          <div className="inline-flex items-center px-3 py-1 rounded-full bg-emerald-400/10 border border-emerald-400/20 text-[11px] uppercase tracking-[0.2em] text-emerald-300 font-medium mb-5">
            Research Workflow
          </div>
          <h2 className="font-serif text-4xl lg:text-6xl font-bold text-white mb-5 tracking-tight">
            Upload → Analyze → Interpret → Export
          </h2>
          <p className="text-white/60 text-lg lg:text-xl leading-relaxed">
            An end-to-end methodology-aware environment for designing, analyzing,
            and reporting agricultural research.
          </p>
        </div>

        <div className="relative max-w-7xl mx-auto">
          <div className="hidden lg:block absolute left-[8%] right-[8%] top-[72px] border-t border-dashed border-emerald-400/25" />
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
            {workflowSteps.map((step, index) => (
              <div
                key={index}
                className="relative group rounded-2xl border border-emerald-400/10 bg-gradient-to-br from-emerald-500/[0.04] to-transparent backdrop-blur-sm p-8 hover:border-emerald-400/30 hover:from-emerald-500/[0.08] transition-all duration-500 hover:-translate-y-1"
              >
                <div className="relative w-[72px] h-[72px] rounded-2xl bg-gradient-to-br from-emerald-400/20 to-emerald-600/5 border border-emerald-400/20 flex items-center justify-center mb-6 mx-auto shadow-[0_0_30px_-5px_rgba(52,211,153,0.3)] group-hover:shadow-[0_0_40px_-5px_rgba(52,211,153,0.5)] transition-shadow">
                  <step.icon className="w-8 h-8 text-emerald-300" strokeWidth={1.75} />
                  <span className="absolute -top-2 -right-2 w-6 h-6 rounded-full bg-emerald-400 text-emerald-950 text-xs font-bold flex items-center justify-center">
                    {index + 1}
                  </span>
                </div>
                <h3 className="font-serif font-semibold text-xl text-white mb-3 text-center">
                  {step.title}
                </h3>
                <p className="text-sm text-white/60 leading-relaxed text-center">
                  {step.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
