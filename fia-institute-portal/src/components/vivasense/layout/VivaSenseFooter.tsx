import { Link } from "react-router-dom";
import { FlaskConical, Mail } from "lucide-react";

const cols = {
  Platform: [
    { name: "Overview", href: "#platform" },
    { name: "Launch Workspace", href: "/vivasense/workspace" },
    { name: "Advanced Analysis", href: "/vivasense/advanced" },
  ],
  Modules: [
    { name: "ANOVA & Field Designs", href: "#modules" },
    { name: "Genetic Parameters", href: "#modules" },
    { name: "GGE Biplot · AMMI · PCA", href: "#modules" },
  ],
  Workflow: [
    { name: "Research Workflow", href: "#workflow" },
    { name: "Methodology Assistant", href: "#assistant" },
    { name: "Publication-Ready Output", href: "#output" },
  ],
  Institutions: [
    { name: "Institutional Licensing", href: "/contact" },
    { name: "Documentation", href: "#documentation" },
    { name: "Contact", href: "/contact" },
  ],
};

export function VivaSenseFooter() {
  return (
    <footer className="bg-[#07150f] border-t border-emerald-400/10 text-white/70">
      <div className="container-wide py-12">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-8 pb-10 border-b border-white/5">
          <div className="col-span-2 md:col-span-1">
            <div className="flex items-center gap-2.5 mb-3">
              <div className="w-9 h-9 rounded-md bg-gradient-to-br from-emerald-400/20 to-emerald-600/10 border border-emerald-400/20 flex items-center justify-center">
                <FlaskConical className="w-4.5 h-4.5 text-emerald-300" />
              </div>
              <span className="font-serif font-semibold text-base text-white">VivaSense</span>
            </div>
            <p className="text-xs leading-relaxed text-white/50">
              Methodology-aware research platform for agricultural science.
            </p>
            <a
              href="mailto:info@fieldtoinsightacademy.com.ng"
              className="inline-flex items-center gap-1.5 mt-3 text-xs text-white/60 hover:text-emerald-300"
            >
              <Mail className="w-3 h-3" />
              info@fieldtoinsightacademy.com.ng
            </a>
          </div>

          {Object.entries(cols).map(([title, links]) => (
            <div key={title}>
              <h4 className="text-xs font-semibold uppercase tracking-wider text-white/90 mb-3">
                {title}
              </h4>
              <ul className="space-y-2">
                {links.map((l) => (
                  <li key={l.name}>
                    {l.href.startsWith("/") ? (
                      <Link to={l.href} className="text-xs text-white/55 hover:text-emerald-300 transition-colors">
                        {l.name}
                      </Link>
                    ) : (
                      <a href={l.href} className="text-xs text-white/55 hover:text-emerald-300 transition-colors">
                        {l.name}
                      </a>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="pt-6 flex flex-col sm:flex-row justify-between items-center gap-2 text-[11px] text-white/40">
          <p>VivaSense — Built by Field-to-Insight Academy</p>
          <p>© {new Date().getFullYear()} · All rights reserved</p>
        </div>
      </div>
    </footer>
  );
}
