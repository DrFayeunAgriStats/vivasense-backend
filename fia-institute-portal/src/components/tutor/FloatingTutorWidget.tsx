import { useState } from "react";
import { MessageCircle, X } from "lucide-react";
import { TutorChat } from "./TutorChat";

export function FloatingTutorWidget() {
  const [open, setOpen] = useState(false);

  return (
    <>
      {/* Chat panel */}
      {open && (
        <div className="fixed bottom-20 right-4 z-50 w-[360px] h-[500px] rounded-2xl border border-border bg-card shadow-lg flex flex-col overflow-hidden animate-fade-in-up sm:w-[400px]">
          <div className="flex items-center justify-between px-4 py-3 bg-primary text-primary-foreground">
            <div>
              <p className="font-semibold text-sm">Dr. Fayeun AI Tutor</p>
              <p className="text-[10px] text-primary-foreground/70">FIA–ADAP Week 0–1</p>
            </div>
            <button onClick={() => setOpen(false)} className="hover:bg-primary-foreground/10 rounded-full p-1 transition">
              <X className="w-4 h-4" />
            </button>
          </div>
          <TutorChat className="flex-1 min-h-0" compact />
        </div>
      )}

      {/* FAB */}
      <button
        onClick={() => setOpen((o) => !o)}
        className="fixed bottom-4 right-4 z-50 w-14 h-14 rounded-full bg-primary text-primary-foreground shadow-lg flex items-center justify-center hover:scale-105 transition-transform"
        aria-label="Open AI Tutor"
      >
        {open ? <X className="w-6 h-6" /> : <MessageCircle className="w-6 h-6" />}
      </button>
    </>
  );
}
