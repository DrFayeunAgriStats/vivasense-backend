import { useState, useEffect, useRef, ReactNode } from "react";
import { Layout } from "@/components/layout/Layout";

const REGISTRATION_URL = "https://forms.gle/E5omnbhtvBuCcXfV6";
const QR_URL =
  "https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=https%3A%2F%2Fforms.gle%2FE5omnbhtvBuCcXfV6&color=0F6E56&bgcolor=ffffff&margin=8";

function useFadeIn() {
  const ref = useRef<HTMLDivElement | null>(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const obs = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) setVisible(true);
      },
      { threshold: 0.1 },
    );
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, []);
  return [ref, visible] as const;
}

function FadeSection({ children, className = "" }: { children: ReactNode; className?: string }) {
  const [ref, visible] = useFadeIn();
  return (
    <div
      ref={ref}
      className={`${className} transition-all duration-700 ease-out ${
        visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-6"
      }`}
    >
      {children}
    </div>
  );
}

const learnItems = [
  { icon: "📊", text: "ANOVA, G×E & stability analysis made simple" },
  { icon: "🧬", text: "Heritability, GCV, PCV & genetic advance" },
  { icon: "🤖", text: "Automatic AI interpretation of results" },
  { icon: "📄", text: "Publication-ready Word reports & tables" },
  { icon: "🔗", text: "Trait correlations & heatmap visualisation" },
  { icon: "🏗️", text: "Structure clean datasets for any trial design" },
];

const receiveLeft = ["Free VivaSense Pro access", "Data collection templates", "Workshop recording"];
const receiveRight = ["Ready-to-use datasets", "VivaSense sample reports", "Certificate of participation*"];

export default function DataClinic() {
  const [pwaPrompt, setPwaPrompt] = useState(false);

  return (
    <Layout>
      {/* HERO */}
      <section className="relative overflow-hidden bg-primary text-primary-foreground">
        {/* decorative circles */}
        <div className="absolute -top-24 -left-24 w-80 h-80 rounded-full bg-accent/10 blur-3xl" />
        <div className="absolute -bottom-32 -right-24 w-96 h-96 rounded-full bg-white/5 blur-3xl" />

        <div className="container-wide relative py-20 md:py-28">
          {/* badge */}
          <div className="inline-flex items-center gap-2 bg-white/10 border border-white/20 px-4 py-1.5 rounded-full text-sm font-medium mb-6">
            <span className="w-2 h-2 rounded-full bg-accent animate-pulse" />
            Free Online Workshop
          </div>

          <h1 className="font-serif text-4xl md:text-6xl font-bold leading-tight mb-4">
            VivaSense
            <span className="block text-accent">Data Clinic</span>
          </h1>

          <p className="text-lg md:text-xl text-primary-foreground/85 max-w-2xl mb-8">
            Turn your field trial data into publication-ready results in minutes — not weeks.
          </p>

          {/* event pills */}
          <div className="flex flex-wrap gap-3 mb-10">
            {[
              { icon: "📅", text: "8–9 May 2026" },
              { icon: "🕒", text: "7:30–9:00 PM WAT" },
              { icon: "💻", text: "Online · Zoom" },
              { icon: "🌱", text: "2 Days · Free" },
            ].map((p) => (
              <span
                key={p.text}
                className="inline-flex items-center gap-2 bg-white/10 border border-white/15 px-4 py-2 rounded-full text-sm"
              >
                <span>{p.icon}</span>
                {p.text}
              </span>
            ))}
          </div>

          {/* CTA buttons */}
          <div className="flex flex-wrap gap-3">
            <a
              href={REGISTRATION_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center justify-center bg-accent text-accent-foreground hover:opacity-90 font-semibold px-6 py-3.5 rounded-2xl transition-all"
            >
              Register Free — Limited Slots
            </a>
            <a
              href="/vivasense"
              className="inline-flex items-center justify-center bg-white text-primary hover:bg-white/90 font-semibold px-6 py-3.5 rounded-2xl transition-all"
            >
              Start Free Analysis
            </a>
            <button
              onClick={() => setPwaPrompt(true)}
              className="text-primary-foreground/70 hover:text-primary-foreground font-medium px-6 py-3.5 rounded-2xl transition-all hover:bg-white/5 text-sm border border-white/10"
            >
              Install VivaSense App
            </button>
          </div>

          {pwaPrompt && (
            <div className="mt-6 inline-flex items-center bg-white/10 border border-white/15 rounded-xl px-4 py-3 text-sm">
              📱 Open VivaSense in your browser → tap Share → Add to Home Screen
              <button
                onClick={() => setPwaPrompt(false)}
                className="ml-3 text-primary-foreground/50 hover:text-primary-foreground"
                aria-label="Dismiss"
              >
                ✕
              </button>
            </div>
          )}
        </div>
      </section>

      {/* STATS STRIP */}
      <section className="bg-secondary border-y border-border">
        <div className="container-wide py-10">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
            {[
              { num: "2", label: "Live Sessions" },
              { num: "5+", label: "Modules Covered" },
              { num: "₦0", label: "Registration Fee" },
              { num: "AI", label: "Powered Reports" },
            ].map((s) => (
              <div key={s.label}>
                <p className="font-serif text-3xl md:text-4xl font-bold text-primary">{s.num}</p>
                <p className="text-sm text-muted-foreground mt-1">{s.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* WHY THIS MATTERS */}
      <section className="section-padding">
        <div className="container-wide">
          <FadeSection>
            <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground text-center mb-12">
              Why This Workshop Matters
            </h2>
            <div className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto">
              {[
                {
                  icon: "⏳",
                  text: "Most researchers waste weeks analysing data manually — VivaSense does it in minutes",
                },
                {
                  icon: "❌",
                  text: "Many results are rejected due to poor statistical interpretation — VivaSense generates publication-ready language automatically",
                },
                {
                  icon: "🚀",
                  text: "VivaSense automates both analysis and interpretation — so you focus on your science, not your software",
                },
              ].map((item, i) => (
                <div key={i} className="card-elevated p-6">
                  <div className="text-3xl mb-3">{item.icon}</div>
                  <p className="text-muted-foreground leading-relaxed">{item.text}</p>
                </div>
              ))}
            </div>
          </FadeSection>
        </div>
      </section>

      {/* WHAT YOU WILL LEARN */}
      <section className="section-padding bg-muted">
        <div className="container-wide">
          <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground text-center mb-12">
            What You Will Learn
          </h2>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5 max-w-5xl mx-auto">
            {learnItems.map((item) => (
              <div
                key={item.text}
                className="flex items-start gap-4 bg-card p-5 rounded-xl border border-border hover:border-primary transition-colors"
              >
                <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center text-2xl flex-shrink-0">
                  {item.icon}
                </div>
                <p className="text-foreground font-medium">{item.text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* FEATURE HIGHLIGHT */}
      <section className="section-padding">
        <div className="container-wide">
          <FadeSection className="max-w-4xl mx-auto">
            <div className="card-elevated p-8 md:p-12 bg-gradient-to-br from-primary/5 to-accent/5">
              <div className="flex flex-col md:flex-row items-start gap-6">
                <div className="text-6xl">🗺️</div>
                <div>
                  <p className="text-sm font-semibold text-accent uppercase tracking-wide mb-1">
                    Special Feature
                  </p>
                  <h3 className="font-serif text-2xl md:text-3xl font-bold text-foreground mb-4">
                    NEW: Field Layout Generator
                  </h3>
                  <ul className="space-y-2">
                    {[
                      "Design CRD & RCBD field layouts instantly",
                      "Export as Excel, PNG or Word",
                      "Ready for real field deployment — before data collection",
                    ].map((pt) => (
                      <li key={pt} className="flex items-start gap-2 text-muted-foreground">
                        <span className="text-primary font-bold mt-0.5">✓</span>
                        <span>{pt}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </FadeSection>
        </div>
      </section>

      {/* FACILITATORS */}
      <section className="section-padding bg-secondary">
        <div className="container-wide">
          <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground text-center mb-12">
            Your Facilitators
          </h2>
          <div className="grid md:grid-cols-2 gap-6 max-w-4xl mx-auto">
            {[
              {
                initial: "F",
                name: "Dr. Lawrence Stephen Fayeun",
                role: "Lead Facilitator",
                desc: "Plant Breeder · Data Scientist · Research Consultant · Founder, Field-to-Insight Academy (FIA) · Associate Professor, FUTA",
              },
              {
                initial: "H",
                name: "Mr. Haruna Isola Aremu",
                role: "Co-Facilitator",
                desc: "Agricultural Data Analyst · Field Trial Specialist · Co-instructor, Field-to-Insight Academy · Supports hands-on training and real dataset walkthroughs",
              },
            ].map((f) => (
              <div key={f.name} className="card-elevated p-6 flex gap-5">
                <div className="w-16 h-16 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-serif text-2xl font-bold flex-shrink-0">
                  {f.initial}
                </div>
                <div>
                  <p className="text-xs uppercase tracking-wide text-accent font-semibold mb-1">{f.role}</p>
                  <h3 className="font-serif text-lg font-semibold text-foreground mb-2">{f.name}</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">{f.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* WHAT YOU RECEIVE */}
      <section className="section-padding">
        <div className="container-wide">
          <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground text-center mb-12">
            What You Receive
          </h2>
          <div className="grid sm:grid-cols-2 gap-4 max-w-3xl mx-auto">
            {[...receiveLeft, ...receiveRight].map((item) => (
              <div key={item} className="flex items-center gap-3 bg-card border border-border p-4 rounded-xl">
                <span className="w-8 h-8 rounded-full bg-primary/10 text-primary flex items-center justify-center font-bold flex-shrink-0">
                  ✓
                </span>
                <span className="text-foreground">{item}</span>
              </div>
            ))}
          </div>
          <p className="text-center text-sm text-muted-foreground mt-6">
            *Certificate available at a small administrative fee
          </p>
        </div>
      </section>

      {/* CTA / REGISTRATION */}
      <section className="section-padding bg-primary text-primary-foreground">
        <div className="container-wide">
          <div className="max-w-2xl mx-auto text-center">
            <p className="text-accent font-semibold uppercase tracking-wide mb-3">Free Registration</p>
            <h2 className="font-serif text-3xl md:text-4xl font-bold mb-4">
              ⚡ Limited to 50 participants — secure your place now
            </h2>

            <div className="bg-white p-4 rounded-2xl inline-block mt-8 mb-4">
              <img
                src={QR_URL}
                alt="QR code to register"
                className="w-48 h-48"
                onError={(e) => {
                  (e.target as HTMLImageElement).style.display = "none";
                }}
              />
            </div>
            <p className="text-primary-foreground/80 mb-8">Scan to register</p>

            <div>
              <a
                href={REGISTRATION_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center bg-accent text-accent-foreground hover:opacity-90 font-semibold px-8 py-4 rounded-2xl transition-all"
              >
                Register Now →
              </a>
            </div>

            <div className="mt-10 flex flex-col sm:flex-row gap-4 justify-center text-sm text-primary-foreground/85">
              <span>📧 info@fieldtoinsightacademy.com.ng</span>
              <span>📱 +234 902 215 8026</span>
            </div>
          </div>
        </div>
      </section>
    </Layout>
  );
}
