import { Button } from "@/components/ui/button";
import { BarChart3, ArrowRight, Sparkles } from "lucide-react";
import { useNavigate } from "react-router-dom";

interface VivaSenseHeroProps {
  onStart?: () => void;
}

export function VivaSenseHero({ onStart }: VivaSenseHeroProps) {
  const navigate = useNavigate();

  const handleStart = () => {
    if (onStart) {
      onStart();
      return;
    }
    navigate("/vivasense/workspace");
  };

  return (
    <section
      id="platform"
      className="relative overflow-hidden pt-28 pb-32 lg:pt-40 lg:pb-44 bg-[#0b1d14]"
    >
      {/* Grid background */}
      <div
        className="absolute inset-0 opacity-[0.09] pointer-events-none"
        style={{
          backgroundImage:
            "linear-gradient(rgba(52,211,153,0.6) 1px, transparent 1px), linear-gradient(90deg, rgba(52,211,153,0.6) 1px, transparent 1px)",
          backgroundSize: "64px 64px",
          maskImage: "radial-gradient(ellipse at center, black 35%, transparent 80%)",
        }}
      />
      {/* Glow layers */}
      <div className="absolute top-[-10%] left-1/2 -translate-x-1/2 w-[1100px] h-[1100px] rounded-full bg-emerald-500/15 blur-[160px] pointer-events-none" />
      <div className="absolute bottom-0 left-1/4 w-[500px] h-[500px] rounded-full bg-emerald-400/10 blur-[120px] pointer-events-none" />
      <div className="absolute bottom-0 right-1/4 w-[500px] h-[500px] rounded-full bg-teal-400/10 blur-[120px] pointer-events-none" />

      <div className="container-wide relative">
        <div className="max-w-5xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-emerald-400/10 text-emerald-300 text-xs font-medium mb-8 border border-emerald-400/25 backdrop-blur-sm">
            <Sparkles className="w-3.5 h-3.5" />
            <span className="tracking-[0.18em] uppercase">Scientific Analytics Platform</span>
          </div>

          <h1 className="font-serif text-6xl md:text-7xl lg:text-8xl xl:text-9xl font-bold text-white mb-8 tracking-tight leading-[0.95]">
            <span className="bg-gradient-to-br from-white via-emerald-50 to-emerald-200/80 bg-clip-text text-transparent">
              VivaSense
            </span>
          </h1>

          <p className="text-2xl lg:text-4xl text-emerald-100/95 font-medium mb-6 max-w-4xl mx-auto leading-tight tracking-tight">
            Methodology-Aware Analytics for Agricultural Research
          </p>

          <p className="text-base lg:text-xl text-white/65 max-w-3xl mx-auto mb-12 leading-relaxed">
            Design experiments, analyze multi-environment trials, and generate
            publication-ready outputs from one integrated, reproducible scientific workspace.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Button
              size="lg"
              onClick={handleStart}
              className="gap-2 bg-emerald-400 text-emerald-950 hover:bg-emerald-300 font-semibold px-9 py-7 text-base shadow-[0_0_60px_-5px_rgba(52,211,153,0.6)]"
            >
              <BarChart3 className="w-5 h-5" />
              Launch Workspace
              <ArrowRight className="w-5 h-5" />
            </Button>
            <Button
              size="lg"
              variant="ghost"
              asChild
              className="text-white/85 hover:text-white hover:bg-white/5 px-8 py-7 text-base border border-white/10"
            >
              <a href="#modules">Explore Modules</a>
            </Button>
          </div>

          <p className="mt-14 text-[11px] uppercase tracking-[0.25em] text-white/35">
            Built by Field-to-Insight Academy
          </p>
        </div>
      </div>
    </section>
  );
}
