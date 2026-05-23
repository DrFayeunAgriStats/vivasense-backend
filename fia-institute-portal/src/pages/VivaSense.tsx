import { Button } from "@/components/ui/button";
import { VivaSenseLayout } from "@/components/vivasense/layout/VivaSenseLayout";
import { VivaSenseHero } from "@/components/vivasense/VivaSenseHero";
import { ResearchWorkflow } from "@/components/vivasense/ResearchWorkflow";
import { VivaSenseFeatures } from "@/components/vivasense/VivaSenseFeatures";
import { OutputShowcase } from "@/components/vivasense/OutputShowcase";
import { VivaSenseAudience } from "@/components/vivasense/VivaSenseAudience";
import { VivaSenseSampleData } from "@/components/vivasense/VivaSenseSampleData";
import { InstallVivaSenseButton } from "@/components/vivasense/InstallVivaSenseButton";
import { VivaSensePlanBadge } from "@/components/vivasense/VivaSensePlanBadge";
import { ProFeatureModal } from "@/components/vivasense/ProFeatureModal";
import { BarChart3, ArrowRight, Sparkles } from "lucide-react";
import { Link } from "react-router-dom";
import { useState } from "react";
import { type ProGuardInfo } from "@/lib/vivasenseGating";

const VivaSense = () => {
  const [proGuard, setProGuard] = useState<ProGuardInfo | null>(null);
  const closeProModal = () => setProGuard(null);

  return (
    <VivaSenseLayout>
      <VivaSenseHero />

      <div className="container-wide flex items-center justify-end gap-3 pt-4">
        <VivaSensePlanBadge />
        <InstallVivaSenseButton />
      </div>

      <div id="workflow">
        <ResearchWorkflow />
      </div>

      <section id="assistant" className="relative py-24 lg:py-32 bg-[#0b1d14] overflow-hidden">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[400px] rounded-full bg-emerald-500/10 blur-[140px] pointer-events-none" />
        <div className="container-wide relative">
          <div className="max-w-4xl mx-auto">
            <div className="rounded-3xl border border-emerald-400/20 bg-gradient-to-br from-emerald-500/[0.08] via-emerald-500/[0.03] to-transparent p-10 lg:p-16 backdrop-blur-sm text-center shadow-[0_0_80px_-20px_rgba(52,211,153,0.3)]">
              <div className="flex justify-center mb-6">
                <span className="rounded-2xl bg-gradient-to-br from-emerald-400/20 to-emerald-600/5 border border-emerald-400/25 p-4 shadow-[0_0_40px_-10px_rgba(52,211,153,0.5)]">
                  <Sparkles className="h-8 w-8 text-emerald-300" />
                </span>
              </div>
              <div className="inline-flex items-center gap-1.5 mb-4 px-3 py-1 rounded-full bg-amber-400/10 border border-amber-400/20 text-[10px] uppercase tracking-[0.2em] text-amber-300 font-semibold">
                Beta
              </div>
              <h2 className="font-serif text-3xl lg:text-5xl font-bold text-white mb-5 tracking-tight">
                Methodology Assistant
              </h2>
              <p className="text-base lg:text-xl text-white/65 mb-10 leading-relaxed max-w-2xl mx-auto">
                Methodology-aware guidance to help frame the right analysis, surface
                assumptions, and structure interpretation — workflow support, not
                autonomous analysis.
              </p>
              <Button
                asChild
                size="lg"
                className="gap-2 bg-emerald-400 text-emerald-950 hover:bg-emerald-300 font-semibold px-9 py-7 text-base shadow-[0_0_50px_-10px_rgba(52,211,153,0.6)]"
              >
                <Link to="/vivasense/genetics">
                  <BarChart3 className="h-5 w-5" />
                  Open Statistical Workspace
                  <ArrowRight className="h-5 w-5" />
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </section>

      <div id="modules">
        <VivaSenseFeatures />
      </div>

      <div id="output">
        <OutputShowcase />
      </div>

      <div id="institutions">
        <VivaSenseAudience />
      </div>

      <div id="documentation" className="container-wide pt-4 -mb-4 text-center">
        <p className="text-xs text-white/50 leading-relaxed">
          Download sample datasets to validate workflows across standard agricultural experimental designs.
        </p>
      </div>
      <VivaSenseSampleData />

      <section className="py-10 lg:py-14 border-t border-emerald-400/10">
        <div className="container-wide max-w-3xl text-center">
          <h2 className="font-serif text-xl lg:text-2xl font-semibold text-white mb-3">
            Scientific Integrity
          </h2>
          <p className="text-sm lg:text-base text-white/60 leading-relaxed">
            VivaSense is under continuous scientific refinement and validation. Workflows
            are designed to support transparent, reproducible, methodology-aware research.
          </p>
        </div>
      </section>

      <ProFeatureModal open={!!proGuard} onOpenChange={(o) => !o && closeProModal()} guard={proGuard} />
    </VivaSenseLayout>
  );
};

export default VivaSense;
