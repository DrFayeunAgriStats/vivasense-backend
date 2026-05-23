import {
  CheckCircle,
  FileText,
  BarChart3,
  Table,
  Download,
  MessageSquare,
  Code,
  AlertTriangle,
} from "lucide-react";

const features = [
  {
    icon: BarChart3,
    title: "Statistical Analysis Suite",
    description: "ANOVA, descriptive statistics, correlation, regression, MANOVA, and nonparametric tests with assumption diagnostics.",
    beta: false,
  },
  {
    icon: CheckCircle,
    title: "Genetic Parameters",
    description: "H², GCV, PCV, genetic advance, and variance components for multi-environment trial data.",
    beta: true,
  },
  {
    icon: AlertTriangle,
    title: "Stability Analysis",
    description: "GGE biplot, AMMI, and Eberhart-Russell — auto-triggered for multi-environment trial data.",
    beta: false,
  },
  {
    icon: Table,
    title: "Advanced Multivariate",
    description: "PCA, cluster analysis, BLUP, path analysis, and selection index across breeding programs.",
    beta: true,
  },
  {
    icon: MessageSquare,
    title: "Trait Relationships",
    description: "Pearson, Spearman, and genotypic correlations with publication-grade heatmap visualisation.",
    beta: false,
  },
  {
    icon: Download,
    title: "Publication-Ready Reports",
    description: "Formatted Word reports with tables, figures, interpretation text, and breeding recommendations.",
    beta: false,
  },
  {
    icon: FileText,
    title: "Methodology Interpretation",
    description: "Discipline-aware explanations with assumption flags and examiner-style checkpoints.",
    beta: true,
  },
  {
    icon: Code,
    title: "Breeding Decision Support",
    description: "Rule-based synthesis across traits with GxE-aware genotype recommendations.",
    beta: false,
  },
];

export function VivaSenseFeatures() {
  return (
    <section className="relative py-24 lg:py-32 bg-[#0a1a11] overflow-hidden">
      <div
        className="absolute inset-0 opacity-[0.04] pointer-events-none"
        style={{
          backgroundImage:
            "linear-gradient(rgba(52,211,153,0.6) 1px, transparent 1px), linear-gradient(90deg, rgba(52,211,153,0.6) 1px, transparent 1px)",
          backgroundSize: "80px 80px",
        }}
      />
      <div className="absolute top-1/3 right-0 w-[600px] h-[600px] rounded-full bg-emerald-500/8 blur-[150px] pointer-events-none" />

      <div className="container-wide relative">
        <div className="text-center mb-20 max-w-3xl mx-auto">
          <div className="inline-flex items-center px-3 py-1 rounded-full bg-emerald-400/10 border border-emerald-400/20 text-[11px] uppercase tracking-[0.2em] text-emerald-300 font-medium mb-5">
            Platform Modules
          </div>
          <h2 className="font-serif text-4xl lg:text-6xl font-bold text-white mb-5 tracking-tight">
            Built for serious agricultural science
          </h2>
          <p className="text-lg lg:text-xl text-white/60 leading-relaxed">
            Statistical analysis, multi-environment trial evaluation, genetic parameter
            estimation, and publication-ready reporting from one integrated workspace.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-6xl mx-auto">
          {features.map((feature, index) => (
            <div
              key={index}
              className="group relative rounded-2xl border border-emerald-400/10 bg-gradient-to-br from-emerald-500/[0.04] to-transparent backdrop-blur-sm p-8 hover:border-emerald-400/30 hover:from-emerald-500/[0.08] transition-all duration-500 hover:-translate-y-0.5"
            >
              <div className="flex items-start gap-5">
                <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-emerald-400/20 to-emerald-600/5 border border-emerald-400/20 flex items-center justify-center shrink-0 shadow-[0_0_25px_-8px_rgba(52,211,153,0.4)] group-hover:shadow-[0_0_35px_-8px_rgba(52,211,153,0.6)] transition-shadow">
                  <feature.icon className="w-7 h-7 text-emerald-300" strokeWidth={1.75} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2.5 flex-wrap">
                    <h3 className="font-serif font-semibold text-white text-xl leading-tight">
                      {feature.title}
                    </h3>
                    {feature.beta && (
                      <span className="inline-flex items-center px-2 py-0.5 rounded-md text-[10px] font-semibold uppercase tracking-wider bg-amber-400/10 text-amber-300 border border-amber-400/20">
                        Beta
                      </span>
                    )}
                  </div>
                  <p className="text-sm lg:text-[15px] text-white/60 leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-16 max-w-3xl mx-auto">
          <div className="rounded-2xl border border-emerald-400/15 bg-emerald-500/[0.03] px-8 py-5 backdrop-blur-sm">
            <p className="text-sm lg:text-base text-white/55 text-center leading-relaxed">
              Developed for agricultural researchers, students, and consultants. Advanced
              workflows are under continuous scientific validation and methodology review.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
