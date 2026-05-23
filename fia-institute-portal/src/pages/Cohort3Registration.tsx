import { Helmet } from "react-helmet-async";
import { useEffect, useState } from "react";
import {
  ArrowRight,
  Sparkles,
  Brain,
  LineChart,
  FlaskConical,
  Database,
  Compass,
  Bot,
  ShieldCheck,
  GraduationCap,
  Microscope,
  BookOpen,
  Briefcase,
  CalendarDays,
  Clock,
  Users,
  Wifi,
  Award,
  MessagesSquare,
  PencilRuler,
  CheckCircle2,
  CreditCard,
  ClipboardCheck,
  Phone,
  Mail,
  Globe,
  QrCode,
  Leaf,
} from "lucide-react";
import heroImg from "@/assets/cohort3-hero.jpg";
import qrImg from "@/assets/cohort3-qr.png";

const STUDENT_URL = "https://paystack.shop/pay/9gwxt0qe17";
const PRO_URL = "https://paystack.shop/pay/e39uuj7a96";

const palette = {
  forest: "#0d3b2e",
  forestSoft: "#13513f",
  gold: "#c89b3c",
  goldSoft: "#e1c07a",
  cream: "#faf6ee",
  creamDeep: "#f1ead9",
  ink: "#0f1f1a",
};

function Section({
  id,
  className = "",
  children,
}: {
  id?: string;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className={`px-6 md:px-10 lg:px-16 ${className}`}>
      <div className="mx-auto max-w-6xl">{children}</div>
    </section>
  );
}

function Eyebrow({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="inline-flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.22em]"
      style={{ color: palette.gold }}
    >
      <span className="h-px w-8" style={{ background: palette.gold }} />
      {children}
    </div>
  );
}

function SectionTitle({
  eyebrow,
  title,
  intro,
  align = "left",
}: {
  eyebrow?: string;
  title: string;
  intro?: string;
  align?: "left" | "center";
}) {
  return (
    <div className={align === "center" ? "text-center" : ""}>
      {eyebrow && <Eyebrow>{eyebrow}</Eyebrow>}
      <h2
        className="mt-4 font-serif text-3xl md:text-4xl lg:text-[2.65rem] leading-[1.1] tracking-tight"
        style={{ color: palette.forest }}
      >
        {title}
      </h2>
      {intro && (
        <p
          className={`mt-4 max-w-2xl text-base md:text-lg leading-relaxed ${align === "center" ? "mx-auto" : ""}`}
          style={{ color: "#3b4a44" }}
        >
          {intro}
        </p>
      )}
    </div>
  );
}

const highlights = [
  { icon: Compass, title: "Foundation in Agricultural Data Thinking", desc: "Build the conceptual scaffolding behind every credible agricultural dataset." },
  { icon: FlaskConical, title: "Experimental Design Principles", desc: "Understand CRD, RCBD, factorial and split-plot logic before touching software." },
  { icon: LineChart, title: "Statistical Reasoning & Interpretation", desc: "Read results like a researcher — not a software output reader." },
  { icon: Database, title: "Data Organisation & Analytical Discipline", desc: "Learn the unglamorous habits that distinguish defensible analyses." },
  { icon: Brain, title: "Research-Ready Mindset", desc: "Move from coursework thinking to investigator-grade reasoning." },
  { icon: Bot, title: "AI-Assisted Agricultural Analytics", desc: "Use AI as a thinking partner — for hypotheses, design checks and interpretation." },
  { icon: ShieldCheck, title: "Responsible Use of AI in Research", desc: "Apply AI with academic integrity, transparency and methodological care." },
];

const audience = [
  { icon: GraduationCap, label: "Final Year Students" },
  { icon: BookOpen, label: "MSc Students" },
  { icon: Microscope, label: "PhD Researchers" },
  { icon: Briefcase, label: "Early-Career Scientists" },
];

const details = [
  { icon: CalendarDays, label: "Duration", value: "6 Weeks" },
  { icon: Sparkles, label: "Starts", value: "Friday, 29 May 2026" },
  { icon: CalendarDays, label: "Days", value: "Fridays & Saturdays" },
  { icon: Clock, label: "Time", value: "7:00 – 9:00 PM (WAT)" },
  { icon: Wifi, label: "Mode", value: "Online — Live Zoom" },
  { icon: Users, label: "Cohort Size", value: "25 Participants Max" },
];

const experience = [
  { icon: MessagesSquare, title: "Live Interactive Sessions", desc: "Weekly faculty-led sessions designed for dialogue, not lecture." },
  { icon: PencilRuler, title: "Practical Exercises", desc: "Structured problems from real agricultural research contexts." },
  { icon: ClipboardCheck, title: "Quizzes & Assessments", desc: "Brief, formative checks that consolidate reasoning each week." },
  { icon: Bot, title: "AI Tutor Support", desc: "Guided AI mentoring between sessions for concept clarification." },
  { icon: Users, title: "Learning Community", desc: "Access to a curated cohort of peers and faculty." },
  { icon: Award, title: "Digital Certificate", desc: "Awarded on completion and assessment-based competence." },
];

export default function Cohort3Registration() {
  const [showStickyCTA, setShowStickyCTA] = useState(false);

  useEffect(() => {
    const onScroll = () => setShowStickyCTA(window.scrollY > 600);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <div
      className="min-h-screen font-sans antialiased"
      style={{ background: palette.cream, color: palette.ink }}
    >
      <Helmet>
        <title>FIA–ADAP Foundations Cohort 3 — Field-to-Insight Academy</title>
        <meta
          name="description"
          content="6-week foundation programme in agricultural data analytics, statistical reasoning and AI-assisted research. Cohort 3 — May to July 2026."
        />
        <link rel="canonical" href="https://fieldtoinsightacademy.com.ng/cohort3-registration" />
        <meta property="og:title" content="FIA–ADAP Foundations Cohort 3" />
        <meta
          property="og:description"
          content="From Field Data to Defensible Insight. A 6-week academic programme by Field-to-Insight Academy."
        />
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://fieldtoinsightacademy.com.ng/cohort3-registration" />
      </Helmet>

      {/* TOP BAR */}
      <header
        className="border-b"
        style={{ borderColor: "rgba(13,59,46,0.12)", background: "rgba(250,246,238,0.85)", backdropFilter: "blur(8px)" }}
      >
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 md:px-10 lg:px-16 py-4">
          <div className="flex items-center gap-2.5">
            <div
              className="grid h-9 w-9 place-items-center rounded-md"
              style={{ background: palette.forest, color: palette.gold }}
            >
              <Leaf size={18} />
            </div>
            <div className="leading-tight">
              <div className="font-serif text-[15px] font-semibold" style={{ color: palette.forest }}>
                Field-to-Insight Academy
              </div>
              <div className="text-[11px] uppercase tracking-[0.18em]" style={{ color: palette.gold }}>
                FIA · Cohort 3
              </div>
            </div>
          </div>
          <a
            href="#pricing"
            className="hidden md:inline-flex items-center gap-2 rounded-full px-5 py-2.5 text-sm font-semibold transition-all hover:-translate-y-0.5"
            style={{ background: palette.forest, color: palette.cream }}
          >
            Register <ArrowRight size={16} />
          </a>
        </div>
      </header>

      {/* HERO */}
      <Section className="pt-16 md:pt-24 pb-20 md:pb-28">
        <div className="grid gap-12 lg:grid-cols-12 lg:gap-16 items-center">
          <div className="lg:col-span-7">
            <div
              className="inline-flex items-center gap-2 rounded-full border px-4 py-1.5 text-[11px] font-semibold uppercase tracking-[0.2em]"
              style={{
                borderColor: "rgba(200,155,60,0.45)",
                color: palette.gold,
                background: "rgba(200,155,60,0.08)",
              }}
            >
              <Sparkles size={13} /> Cohort 3 · May – July 2026
            </div>
            <h1
              className="mt-6 font-serif text-[2.6rem] sm:text-5xl lg:text-[4.2rem] leading-[1.02] tracking-tight"
              style={{ color: palette.forest }}
            >
              FIA–ADAP <span style={{ color: palette.gold }}>Foundations</span>
            </h1>
            <p className="mt-6 max-w-xl text-lg md:text-xl leading-relaxed" style={{ color: "#2f3d37" }}>
              A 6-week foundation programme in Agricultural Data Analytics, Research Thinking,
              Statistical Reasoning, and AI-Assisted Agricultural Research.
            </p>
            <div
              className="mt-6 inline-block border-l-2 pl-4 font-serif italic text-lg md:text-xl"
              style={{ borderColor: palette.gold, color: palette.forestSoft }}
            >
              From Field Data to Defensible Insight.
            </div>

            <div className="mt-10 flex flex-col sm:flex-row gap-3.5">
              <a
                href={STUDENT_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="group inline-flex items-center justify-center gap-2 rounded-full px-7 py-4 text-sm font-semibold tracking-wide shadow-lg transition-all hover:-translate-y-0.5 hover:shadow-xl"
                style={{ background: palette.forest, color: palette.cream }}
              >
                Register as Student
                <ArrowRight size={16} className="transition-transform group-hover:translate-x-1" />
              </a>
              <a
                href={PRO_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="group inline-flex items-center justify-center gap-2 rounded-full border-2 px-7 py-4 text-sm font-semibold tracking-wide transition-all hover:-translate-y-0.5"
                style={{ borderColor: palette.forest, color: palette.forest, background: "transparent" }}
              >
                Register as Professional
                <ArrowRight size={16} className="transition-transform group-hover:translate-x-1" />
              </a>
            </div>

            <div className="mt-10 flex flex-wrap gap-x-8 gap-y-3 text-sm" style={{ color: "#4a5853" }}>
              <span className="inline-flex items-center gap-2"><CheckCircle2 size={15} style={{ color: palette.gold }} /> Live online sessions</span>
              <span className="inline-flex items-center gap-2"><CheckCircle2 size={15} style={{ color: palette.gold }} /> 25-participant cohort</span>
              <span className="inline-flex items-center gap-2"><CheckCircle2 size={15} style={{ color: palette.gold }} /> Digital certificate</span>
            </div>
          </div>

          <div className="lg:col-span-5">
            <div className="relative">
              <div
                className="absolute -inset-4 -z-10 rounded-[2rem] opacity-70"
                style={{ background: `linear-gradient(135deg, ${palette.gold}33, ${palette.forest}22)` }}
              />
              <div
                className="overflow-hidden rounded-[1.5rem] shadow-2xl ring-1"
                style={{ boxShadow: "0 30px 80px -30px rgba(13,59,46,0.45)", borderColor: "rgba(13,59,46,0.1)" }}
              >
                <img
                  src={heroImg}
                  alt="Aerial view of an agricultural research field with overlaid analytics"
                  width={1536}
                  height={1024}
                  className="h-full w-full object-cover"
                />
              </div>
              <div
                className="absolute -bottom-6 -left-6 hidden md:block rounded-2xl px-5 py-4 shadow-xl"
                style={{ background: palette.forest, color: palette.cream }}
              >
                <div className="text-[10px] uppercase tracking-[0.22em]" style={{ color: palette.gold }}>
                  Programme
                </div>
                <div className="mt-1 font-serif text-lg leading-tight">6-Week Foundations</div>
                <div className="text-xs opacity-80">Cohort 3 · 2026</div>
              </div>
            </div>
          </div>
        </div>
      </Section>

      {/* HIGHLIGHTS */}
      <Section className="py-20 md:py-28">
        <SectionTitle
          eyebrow="Programme Outcomes"
          title="What You Will Gain"
          intro="Foundations designed for researchers who want to reason — not just run software."
        />
        <div className="mt-14 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {highlights.map(({ icon: Icon, title, desc }, i) => (
            <article
              key={title}
              className="group relative rounded-2xl border p-7 transition-all hover:-translate-y-1"
              style={{
                background: "white",
                borderColor: "rgba(13,59,46,0.1)",
                boxShadow: "0 6px 24px -12px rgba(13,59,46,0.18)",
              }}
            >
              <div
                className="grid h-11 w-11 place-items-center rounded-xl"
                style={{ background: `${palette.forest}10`, color: palette.forest }}
              >
                <Icon size={20} />
              </div>
              <h3 className="mt-5 font-serif text-lg leading-snug" style={{ color: palette.forest }}>
                {title}
              </h3>
              <p className="mt-2.5 text-sm leading-relaxed" style={{ color: "#4a5853" }}>
                {desc}
              </p>
              <div
                className="absolute left-7 right-7 bottom-0 h-px scale-x-0 origin-left transition-transform duration-500 group-hover:scale-x-100"
                style={{ background: palette.gold }}
              />
              <div className="mt-5 text-[11px] uppercase tracking-[0.2em]" style={{ color: palette.gold }}>
                {String(i + 1).padStart(2, "0")}
              </div>
            </article>
          ))}
        </div>
      </Section>

      {/* AUDIENCE */}
      <Section className="py-20 md:py-24" >
        <div className="rounded-3xl px-8 md:px-12 py-14 md:py-16" style={{ background: palette.creamDeep }}>
          <SectionTitle eyebrow="Who This Is For" title="Designed for serious learners." />
          <div className="mt-12 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
            {audience.map(({ icon: Icon, label }) => (
              <div
                key={label}
                className="flex items-center gap-4 rounded-xl border bg-white px-5 py-5 transition-all hover:-translate-y-0.5"
                style={{ borderColor: "rgba(13,59,46,0.1)" }}
              >
                <div
                  className="grid h-11 w-11 shrink-0 place-items-center rounded-lg"
                  style={{ background: palette.forest, color: palette.gold }}
                >
                  <Icon size={20} />
                </div>
                <div className="font-serif text-[15px]" style={{ color: palette.forest }}>
                  {label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* DETAILS */}
      <Section className="py-20 md:py-28">
        <SectionTitle eyebrow="Programme Details" title="Structure & Schedule" />
        <div className="mt-14 grid gap-px rounded-2xl overflow-hidden border sm:grid-cols-2 lg:grid-cols-3"
             style={{ borderColor: "rgba(13,59,46,0.12)", background: "rgba(13,59,46,0.12)" }}>
          {details.map(({ icon: Icon, label, value }) => (
            <div key={label} className="flex items-start gap-4 bg-white p-7">
              <div
                className="grid h-10 w-10 shrink-0 place-items-center rounded-lg"
                style={{ background: `${palette.gold}1f`, color: palette.gold }}
              >
                <Icon size={18} />
              </div>
              <div>
                <div className="text-[11px] uppercase tracking-[0.2em]" style={{ color: palette.gold }}>
                  {label}
                </div>
                <div className="mt-1.5 font-serif text-lg" style={{ color: palette.forest }}>
                  {value}
                </div>
              </div>
            </div>
          ))}
        </div>
      </Section>

      {/* LEARNING EXPERIENCE */}
      <Section className="py-20 md:py-28">
        <SectionTitle
          eyebrow="The Learning Experience"
          title="Considered. Practical. Mentored."
          intro="Every component is designed to build genuine analytical maturity."
        />
        <div className="mt-14 grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {experience.map(({ icon: Icon, title, desc }) => (
            <div
              key={title}
              className="rounded-2xl border bg-white p-7 transition-all hover:-translate-y-1"
              style={{ borderColor: "rgba(13,59,46,0.1)", boxShadow: "0 6px 24px -14px rgba(13,59,46,0.2)" }}
            >
              <div
                className="grid h-11 w-11 place-items-center rounded-xl"
                style={{ background: `${palette.forest}10`, color: palette.forest }}
              >
                <Icon size={20} />
              </div>
              <h3 className="mt-5 font-serif text-lg" style={{ color: palette.forest }}>
                {title}
              </h3>
              <p className="mt-2 text-sm leading-relaxed" style={{ color: "#4a5853" }}>
                {desc}
              </p>
            </div>
          ))}
        </div>
      </Section>

      {/* PRICING */}
      <Section id="pricing" className="py-20 md:py-28">
        <SectionTitle
          eyebrow="Investment"
          title="Choose your category."
          intro="Early-bird pricing is available until 22 May 2026."
          align="center"
        />
        <div className="mt-14 grid gap-7 md:grid-cols-2 max-w-4xl mx-auto">
          {/* Student */}
          <div
            className="group relative rounded-3xl border p-8 md:p-10 bg-white transition-all hover:-translate-y-1"
            style={{ borderColor: "rgba(13,59,46,0.12)", boxShadow: "0 12px 40px -20px rgba(13,59,46,0.25)" }}
          >
            <div className="text-[11px] uppercase tracking-[0.22em]" style={{ color: palette.gold }}>
              Student
            </div>
            <h3 className="mt-3 font-serif text-2xl" style={{ color: palette.forest }}>
              Final-year & Graduate Students
            </h3>
            <div className="mt-7 flex items-end gap-3">
              <div className="font-serif text-5xl tracking-tight" style={{ color: palette.forest }}>
                ₦30,000
              </div>
              <div className="pb-2 text-sm line-through" style={{ color: "#94a09a" }}>
                ₦35,000
              </div>
            </div>
            <div className="mt-1 text-xs uppercase tracking-[0.18em]" style={{ color: palette.gold }}>
              Early-bird
            </div>
            <ul className="mt-7 space-y-3 text-sm" style={{ color: "#3b4a44" }}>
              {["Live weekly sessions", "Practical exercises", "AI tutor access", "Digital certificate"].map((f) => (
                <li key={f} className="flex items-start gap-2.5">
                  <CheckCircle2 size={16} style={{ color: palette.gold }} className="mt-0.5 shrink-0" />
                  {f}
                </li>
              ))}
            </ul>
            <a
              href={STUDENT_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-9 inline-flex w-full items-center justify-center gap-2 rounded-full px-6 py-3.5 text-sm font-semibold transition-all hover:-translate-y-0.5"
              style={{ background: palette.forest, color: palette.cream }}
            >
              Register as Student <ArrowRight size={15} />
            </a>
          </div>

          {/* Professional */}
          <div
            className="group relative rounded-3xl p-8 md:p-10 transition-all hover:-translate-y-1 overflow-hidden"
            style={{
              background: `linear-gradient(160deg, ${palette.forest} 0%, ${palette.forestSoft} 100%)`,
              color: palette.cream,
              boxShadow: "0 24px 60px -24px rgba(13,59,46,0.55)",
            }}
          >
            <div
              className="absolute -top-12 -right-12 h-48 w-48 rounded-full opacity-20 blur-2xl"
              style={{ background: palette.gold }}
            />
            <div
              className="inline-flex items-center gap-2 rounded-full border px-3 py-1 text-[10px] uppercase tracking-[0.22em]"
              style={{ borderColor: "rgba(200,155,60,0.5)", color: palette.gold }}
            >
              Professional · Recommended
            </div>
            <h3 className="mt-3 font-serif text-2xl">Researchers & Practitioners</h3>
            <div className="mt-7 flex items-end gap-3">
              <div className="font-serif text-5xl tracking-tight">₦55,000</div>
              <div className="pb-2 text-sm line-through opacity-60">₦60,000</div>
            </div>
            <div className="mt-1 text-xs uppercase tracking-[0.18em]" style={{ color: palette.gold }}>
              Early-bird
            </div>
            <ul className="mt-7 space-y-3 text-sm opacity-95">
              {["All student benefits", "Priority cohort placement", "Professional certificate", "Extended AI tutor access"].map((f) => (
                <li key={f} className="flex items-start gap-2.5">
                  <CheckCircle2 size={16} style={{ color: palette.gold }} className="mt-0.5 shrink-0" />
                  {f}
                </li>
              ))}
            </ul>
            <a
              href={PRO_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-9 inline-flex w-full items-center justify-center gap-2 rounded-full px-6 py-3.5 text-sm font-semibold transition-all hover:-translate-y-0.5"
              style={{ background: palette.gold, color: palette.forest }}
            >
              Register as Professional <ArrowRight size={15} />
            </a>
          </div>
        </div>
        <p className="mt-8 text-center text-sm" style={{ color: "#5a6862" }}>
          Early-bird discount ends <strong style={{ color: palette.forest }}>22 May 2026</strong>.
        </p>
      </Section>

      {/* REGISTRATION FLOW */}
      <Section className="py-20 md:py-28">
        <SectionTitle eyebrow="Registration" title="A simple three-step process." align="center" />
        <div className="mt-14 grid gap-6 md:grid-cols-3">
          {[
            { icon: Compass, step: "01", title: "Choose your category", desc: "Select Student or Professional based on your stage." },
            { icon: CreditCard, step: "02", title: "Pay securely with Paystack", desc: "Encrypted, verified payment via Paystack." },
            { icon: ClipboardCheck, step: "03", title: "Complete the registration form", desc: "You will be redirected automatically after payment." },
          ].map(({ icon: Icon, step, title, desc }) => (
            <div
              key={step}
              className="relative rounded-2xl border bg-white p-8"
              style={{ borderColor: "rgba(13,59,46,0.1)" }}
            >
              <div
                className="absolute -top-4 left-7 rounded-full px-3 py-1 text-[10px] font-bold tracking-[0.22em]"
                style={{ background: palette.gold, color: palette.forest }}
              >
                STEP {step}
              </div>
              <div
                className="grid h-11 w-11 place-items-center rounded-xl"
                style={{ background: `${palette.forest}10`, color: palette.forest }}
              >
                <Icon size={20} />
              </div>
              <h3 className="mt-5 font-serif text-lg" style={{ color: palette.forest }}>
                {title}
              </h3>
              <p className="mt-2 text-sm leading-relaxed" style={{ color: "#4a5853" }}>
                {desc}
              </p>
            </div>
          ))}
        </div>
      </Section>

      {/* QR CODE */}
      <Section className="py-20 md:py-24">
        <div
          className="grid gap-10 md:gap-14 md:grid-cols-2 items-center rounded-3xl px-8 md:px-12 py-14 md:py-16"
          style={{ background: palette.creamDeep }}
        >
          <div>
            <Eyebrow>Quick Access</Eyebrow>
            <h3 className="mt-4 font-serif text-3xl md:text-4xl tracking-tight" style={{ color: palette.forest }}>
              Scan to Register
            </h3>
            <p className="mt-4 max-w-md text-base leading-relaxed" style={{ color: "#3b4a44" }}>
              Open your camera and scan to access the FIA–ADAP Cohort 3 registration page on any device.
            </p>
            <div className="mt-6 inline-flex items-center gap-2 text-sm" style={{ color: palette.forestSoft }}>
              <QrCode size={16} /> fieldtoinsightacademy.com.ng/cohort3-registration
            </div>
          </div>
          <div className="flex md:justify-end">
            <div
              className="rounded-2xl bg-white p-6 shadow-xl"
              style={{ boxShadow: "0 20px 50px -20px rgba(13,59,46,0.3)" }}
            >
              <img src={qrImg} alt="QR code linking to the registration page" width={260} height={260} className="h-[240px] w-[240px]" />
              <div className="mt-3 text-center text-[11px] uppercase tracking-[0.2em]" style={{ color: palette.gold }}>
                Cohort 3 · 2026
              </div>
            </div>
          </div>
        </div>
      </Section>

      {/* PHILOSOPHY */}
      <Section className="py-24 md:py-32">
        <div
          className="relative overflow-hidden rounded-3xl px-8 md:px-14 py-16 md:py-24 text-center"
          style={{
            background: `linear-gradient(135deg, ${palette.forest} 0%, #082822 100%)`,
            color: palette.cream,
          }}
        >
          <div
            className="absolute inset-0 opacity-[0.07]"
            style={{
              backgroundImage:
                "radial-gradient(circle at 25% 30%, rgba(200,155,60,0.6) 0, transparent 40%), radial-gradient(circle at 75% 70%, rgba(200,155,60,0.4) 0, transparent 45%)",
            }}
          />
          <div className="relative">
            <Eyebrow>Core Philosophy</Eyebrow>
            <div className="mt-8 flex flex-wrap items-center justify-center gap-3 md:gap-6 font-serif text-xl md:text-3xl">
              {["Field", "Analysis", "Insight", "Action"].map((word, i, arr) => (
                <span key={word} className="inline-flex items-center gap-3 md:gap-6">
                  <span style={{ color: i === arr.length - 1 ? palette.gold : palette.cream }}>{word}</span>
                  {i < arr.length - 1 && (
                    <ArrowRight size={20} style={{ color: palette.gold }} className="opacity-80" />
                  )}
                </span>
              ))}
            </div>
            <p className="mt-10 mx-auto max-w-2xl font-serif italic text-xl md:text-2xl leading-snug">
              “From Field Data to Defensible Insight.”
            </p>
            <div className="mt-3 text-[11px] uppercase tracking-[0.25em]" style={{ color: palette.gold }}>
              Field-to-Insight Academy
            </div>
          </div>
        </div>
      </Section>

      {/* FOOTER */}
      <footer
        className="border-t px-6 md:px-10 lg:px-16 py-14"
        style={{ borderColor: "rgba(13,59,46,0.15)", background: palette.cream }}
      >
        <div className="mx-auto max-w-6xl grid gap-10 md:grid-cols-3">
          <div>
            <div className="flex items-center gap-2.5">
              <div
                className="grid h-9 w-9 place-items-center rounded-md"
                style={{ background: palette.forest, color: palette.gold }}
              >
                <Leaf size={18} />
              </div>
              <div className="font-serif text-base font-semibold" style={{ color: palette.forest }}>
                Field-to-Insight Academy
              </div>
            </div>
            <p className="mt-4 text-sm leading-relaxed" style={{ color: "#4a5853" }}>
              An AI-powered ecosystem for agricultural research, training, and data-driven insight.
            </p>
          </div>
          <div>
            <div className="text-[11px] uppercase tracking-[0.22em]" style={{ color: palette.gold }}>
              Contact
            </div>
            <ul className="mt-4 space-y-3 text-sm" style={{ color: "#3b4a44" }}>
              <li className="flex items-center gap-2.5"><Phone size={15} style={{ color: palette.forest }} /> <a href="tel:+2349022158026" className="hover:underline">+234 902 215 8026</a></li>
              <li className="flex items-center gap-2.5"><Mail size={15} style={{ color: palette.forest }} /> fieldtoinsightacademy@gmail.com</li>
              <li className="flex items-center gap-2.5"><Globe size={15} style={{ color: palette.forest }} /> fieldtoinsightacademy.com.ng</li>
            </ul>
          </div>
          <div>
            <div className="text-[11px] uppercase tracking-[0.22em]" style={{ color: palette.gold }}>
              Cohort 3
            </div>
            <p className="mt-4 text-sm" style={{ color: "#3b4a44" }}>
              Registration opens for the May – July 2026 cohort. 25 participant maximum.
            </p>
            <a
              href={PRO_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-5 inline-flex items-center gap-2 text-sm font-semibold"
              style={{ color: palette.forest }}
            >
              Reserve your place <ArrowRight size={14} />
            </a>
          </div>
        </div>
        <div
          className="mx-auto mt-12 max-w-6xl border-t pt-6 text-xs"
          style={{ borderColor: "rgba(13,59,46,0.12)", color: "#6b766f" }}
        >
          © {new Date().getFullYear()} Field-to-Insight Academy. All rights reserved.
        </div>
      </footer>

      {/* STICKY MOBILE CTA */}
      <div
        className={`fixed bottom-0 left-0 right-0 z-50 md:hidden border-t px-4 py-3 transition-transform duration-300 ${
          showStickyCTA ? "translate-y-0" : "translate-y-full"
        }`}
        style={{ background: "rgba(250,246,238,0.96)", backdropFilter: "blur(10px)", borderColor: "rgba(13,59,46,0.15)" }}
      >
        <div className="flex gap-2">
          <a
            href={STUDENT_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="flex-1 inline-flex items-center justify-center rounded-full border-2 px-4 py-3 text-xs font-semibold"
            style={{ borderColor: palette.forest, color: palette.forest }}
          >
            Student
          </a>
          <a
            href={PRO_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="flex-1 inline-flex items-center justify-center rounded-full px-4 py-3 text-xs font-semibold"
            style={{ background: palette.forest, color: palette.cream }}
          >
            Professional
          </a>
        </div>
      </div>
    </div>
  );
}
