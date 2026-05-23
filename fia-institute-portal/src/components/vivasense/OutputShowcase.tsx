import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { ZoomIn, ZoomOut, Maximize2 } from "lucide-react";
import { useMemo, useState } from "react";

export function OutputShowcase() {
  const imgSrc = "/images/gge-mean-stability-demo.png";
  const [zoomLevel, setZoomLevel] = useState(1);

  const zoomPercent = useMemo(() => `${Math.round(zoomLevel * 100)}%`, [zoomLevel]);

  const decreaseZoom = () => setZoomLevel((prev) => Math.max(1, Number((prev - 0.2).toFixed(1))));
  const increaseZoom = () => setZoomLevel((prev) => Math.min(2.6, Number((prev + 0.2).toFixed(1))));

  return (
    <section className="py-24 bg-muted/30" aria-labelledby="vivasense-output-showcase-title">
      <div className="container-wide">
        <div className="text-center mb-14">
          <h2
            id="vivasense-output-showcase-title"
            className="font-serif text-4xl lg:text-5xl font-bold text-foreground mb-3"
          >
            Research Outputs from VivaSense
          </h2>
        </div>

        <figure className="max-w-[1000px] mx-auto">
          <Card className="bg-background border-border shadow-md transition-all duration-300 hover:shadow-xl hover:shadow-emerald-900/10 hover:scale-[1.01]">
            <CardContent className="p-5 sm:p-6 md:p-7 lg:p-8">
              <div className="rounded-xl border border-border bg-white/90 overflow-hidden">
                <img
                  src={imgSrc}
                  alt="VivaSense GGE biplot showing mean performance and stability analysis"
                  className="w-full h-auto max-w-full object-contain"
                  loading="lazy"
                  onError={(e) => {
                    e.currentTarget.src = "/images/gge-mean-stability-demo.png";
                  }}
                />
              </div>
              <figcaption className="mt-5 sm:mt-6 text-center">
                <div className="flex items-center justify-center mb-2">
                  <Badge className="bg-emerald-100 text-emerald-800 border border-emerald-200 hover:bg-emerald-100">
                    Live VivaSense Output
                  </Badge>
                </div>
                <p className="text-sm sm:text-[15px] text-muted-foreground leading-relaxed px-1">
                  GGE Biplot Mean vs Stability View generated in VivaSense showing genotype performance, stability,
                  and ideal genotype ranking across environments.
                </p>
                <p className="text-xs text-muted-foreground/85 mt-2">
                  Generated directly in VivaSense using live analytical workflows.
                </p>
                <Dialog>
                  <DialogTrigger asChild>
                    <Button
                      type="button"
                      variant="outline"
                      className="mt-4 gap-2"
                      onClick={() => setZoomLevel(1)}
                    >
                      <Maximize2 className="h-4 w-4" />
                      Click to expand
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="max-w-6xl w-[95vw] p-4 sm:p-6">
                    <div className="sr-only">
                      <DialogTitle>VivaSense GGE Biplot Mean vs Stability</DialogTitle>
                      <DialogDescription>
                        Enlarged view of the scientifically generated GGE Mean vs Stability output.
                      </DialogDescription>
                    </div>

                    <div className="flex items-center justify-between gap-3 pb-2 border-b border-border">
                      <p className="text-sm font-medium text-foreground">Interactive chart preview</p>
                      <div className="flex items-center gap-2">
                        <Button type="button" variant="outline" size="sm" onClick={decreaseZoom}>
                          <ZoomOut className="h-4 w-4" />
                        </Button>
                        <span className="text-xs text-muted-foreground min-w-[42px] text-center">{zoomPercent}</span>
                        <Button type="button" variant="outline" size="sm" onClick={increaseZoom}>
                          <ZoomIn className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>

                    <div className="max-h-[78vh] overflow-auto rounded-lg border border-border bg-white">
                      <img
                        src={imgSrc}
                        alt="Expanded VivaSense GGE biplot showing mean performance and stability analysis"
                        className="origin-top-left transition-transform duration-200"
                        style={{
                          transform: `scale(${zoomLevel})`,
                          width: `${100 / zoomLevel}%`,
                          height: "auto",
                        }}
                      />
                    </div>
                  </DialogContent>
                </Dialog>
              </figcaption>
            </CardContent>
          </Card>
        </figure>
      </div>
    </section>
  );
}
