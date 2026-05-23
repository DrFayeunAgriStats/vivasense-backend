import { useState } from "react";
import { Link } from "react-router-dom";
import { Menu, X, FlaskConical, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";

const nav = [
  { name: "Platform", href: "#platform" },
  { name: "Modules", href: "#modules" },
  { name: "Workflow", href: "#workflow" },
  { name: "Institutions", href: "#institutions" },
  { name: "Documentation", href: "#documentation" },
];

export function VivaSenseHeader() {
  const [open, setOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 bg-[#0b1d14]/85 backdrop-blur-xl border-b border-emerald-400/10">
      <nav className="container-wide flex items-center justify-between py-3.5">
        <Link to="/vivasense" className="flex items-center gap-2.5 group">
          <div className="w-9 h-9 rounded-md bg-gradient-to-br from-emerald-400/20 to-emerald-600/10 border border-emerald-400/20 flex items-center justify-center">
            <FlaskConical className="w-4.5 h-4.5 text-emerald-300" />
          </div>
          <div className="flex flex-col leading-tight">
            <span className="font-serif font-semibold text-[17px] text-white tracking-tight">
              VivaSense
            </span>
            <span className="text-[10px] text-emerald-200/50 font-medium tracking-wider uppercase">
              by Field-to-Insight Academy
            </span>
          </div>
        </Link>

        <div className="hidden lg:flex items-center gap-1">
          {nav.map((item) => (
            <a
              key={item.name}
              href={item.href}
              className="px-3 py-2 text-sm font-medium text-white/65 hover:text-white rounded-md hover:bg-white/5 transition-colors"
            >
              {item.name}
            </a>
          ))}
        </div>

        <div className="hidden lg:flex items-center gap-2">
          <Button
            asChild
            size="sm"
            className="bg-emerald-400 text-emerald-950 hover:bg-emerald-300 font-semibold gap-1.5 shadow-[0_0_20px_-5px_rgba(52,211,153,0.5)]"
          >
            <Link to="/vivasense/workspace">
              Launch Workspace
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </Button>
        </div>

        <button
          className="lg:hidden p-2 rounded-md text-white/70 hover:text-white hover:bg-white/5"
          onClick={() => setOpen(!open)}
          aria-label="Toggle menu"
        >
          {open ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </nav>

      {open && (
        <div className="lg:hidden border-t border-emerald-400/10 bg-[#0b1d14]/95">
          <div className="container-wide py-3 space-y-1">
            {nav.map((item) => (
              <a
                key={item.name}
                href={item.href}
                onClick={() => setOpen(false)}
                className="block px-3 py-2.5 text-sm font-medium text-white/70 hover:text-white hover:bg-white/5 rounded-md"
              >
                {item.name}
              </a>
            ))}
            <Button
              asChild
              size="sm"
              className="w-full mt-2 bg-emerald-400 text-emerald-950 hover:bg-emerald-300 font-semibold"
            >
              <Link to="/vivasense/workspace" onClick={() => setOpen(false)}>
                Launch Workspace
              </Link>
            </Button>
          </div>
        </div>
      )}
    </header>
  );
}
