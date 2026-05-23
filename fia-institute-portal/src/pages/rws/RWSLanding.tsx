import { Link } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  ArrowRight, BookOpen, Brain, Shield, GraduationCap, Target,
  FlaskConical, MessageSquare, Award, AlertTriangle, CheckCircle,
  Users, FileText, Microscope, HelpCircle, ChevronDown, ClipboardCheck,
  Route, PenTool, BarChart3, UserCheck, Swords, BadgeCheck,
} from "lucide-react";
import { useState } from "react";

const TRACKS = [
  { title: "Undergraduate Final-Year Project", duration: "Flexible", desc: "Develop a well-structured project report with clear evidence, proper academic formatting, and defensible conclusions.", icon: FileText },
  { title: "MSc Thesis Development", duration: "1 month", desc: "Build research reasoning skills for thesis chapter writing, results presentation, and evidence-based discussion.", icon: BookOpen },
  { title: "PhD Research Writing and Defense", duration: "2 months", desc: "Advanced methodology, publication-quality academic writing, critical analysis, and comprehensive viva voce preparation.", icon: Microscope },
  { title: "Research Paper Writing", duration: "2 weeks", desc: "Craft publication-ready manuscripts with structured results, rigorous discussion, and clear scientific narrative.", icon: PenTool },
];

const MODULES = [
  { title: "AI Research Writing Mentor", desc: "Structured guidance for developing each chapter of a thesis, dissertation, or project report.", icon: Brain, available: true },
  { title: "Guided Results Interpretation Lab", desc: "Learn to interpret ANOVA tables, correlation matrices, PCA, AMMI, GGE biplots, and other statistical outputs.", icon: FlaskConical, available: false },
  { title: "Research Journey Tracker", desc: "Track research progress from proposal development to thesis defense with milestone-based accountability.", icon: Target, available: false },
  { title: "Defense Simulator", desc: "Prepare for viva voce examination using AI-generated practice questions and structured feedback.", icon: Shield, available: false },
  { title: "Supervisor Guidance System", desc: "Structured supervisor-student milestone interaction and progress review tools.", icon: Users, available: false },
  { title: "Certification", desc: "Earn competence-based certificates after demonstrating research mastery across core modules.", icon: Award, available: false },
];

const MISTAKES = [
  "Presenting results without numerical evidence or units",
  "Confusing the Results section with the Discussion section",
  "Reporting p-values without explaining what they mean for the research question",
  "Writing vague conclusions without specific reference to findings",
  "Omitting measures of variability (SE, LSD, CV%) from tables",
  "Copy-pasting statistical output without interpreting it",
  "Failing to explain what statistical results mean in biological or practical terms",
];

const JOURNEY_STEPS = [
  { step: 1, title: "Diagnostic Assessment", desc: "Identify knowledge gaps and determine your starting level through a structured evaluation.", icon: ClipboardCheck },
  { step: 2, title: "Guided Learning Pathway", desc: "Receive a personalised research learning path matched to your academic track and current stage.", icon: Route },
  { step: 3, title: "AI Research Writing Mentor", desc: "Develop proposal and thesis chapters step-by-step with structured AI guidance.", icon: Brain },
  { step: 4, title: "Results Interpretation Lab", desc: "Learn to interpret statistical outputs and translate data into meaningful research conclusions.", icon: BarChart3 },
  { step: 5, title: "Supervisor Interaction", desc: "Structured milestone-based feedback and progress discussions with your supervisor.", icon: UserCheck },
  { step: 6, title: "Defense Simulation", desc: "Prepare for viva voce with realistic examination questions and guided response strategies.", icon: Swords },
  { step: 7, title: "Certification of Competence", desc: "Receive certification after demonstrating mastery across research writing and analysis modules.", icon: BadgeCheck },
];

const FAQS = [
  { q: "Does FIA write my thesis for me?", a: "No. FIA teaches you to think, analyse, and write your own research. All AI tools provide guidance and prompts — never ready-made text." },
  { q: "What academic levels are supported?", a: "Final-year undergraduate projects, MSc theses, PhD dissertations, and research papers across agricultural and biological sciences." },
  { q: "Is this suitable for non-agricultural disciplines?", a: "The core methodology is applicable to all quantitative research, though examples are drawn primarily from agricultural and biological sciences." },
  { q: "How is academic integrity maintained?", a: "All AI-generated content is clearly labelled as guidance notes. Students must write in their own words. PDF exports include watermarks indicating 'NOT THESIS TEXT'." },
  { q: "Can my supervisor see my progress?", a: "Supervisor dashboards are planned for a future phase. Currently, students can share PDF guidance notes with their supervisors." },
  { q: "Can undergraduate final-year students use FIA?", a: "Yes. FIA includes a dedicated track for final-year research projects in addition to postgraduate research training." },
  { q: "Does FIA replace my academic supervisor?", a: "No. FIA supports students between supervision meetings but does not replace university supervision. It complements the guidance you receive from your academic supervisor." },
];

export default function RWSLanding() {
  const [openFaq, setOpenFaq] = useState<number | null>(null);

  return (
    <Layout>
      {/* Hero */}
      <section className="relative overflow-hidden" style={{ background: "var(--gradient-hero)" }}>
        <div className="container max-w-5xl py-20 md:py-28 lg:py-32 text-primary-foreground relative z-10">
          <p className="text-primary-foreground/60 text-xs uppercase tracking-[0.2em] mb-4 font-medium">
            Field-to-Insight Academy — Research Writing System
          </p>
          <h1 className="font-serif text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold leading-tight max-w-3xl">
            Learn to Interpret Data and Write Research Like a Scientist
          </h1>
          <p className="text-primary-foreground/80 text-base sm:text-lg md:text-xl mt-6 max-w-2xl leading-relaxed">
            A professor-led, AI-assisted academic platform that teaches postgraduate and undergraduate researchers to reason through their data, interpret statistical results, and produce defensible research writing.
          </p>
          <div className="flex flex-wrap gap-4 mt-8">
            <Link to="/research-writing/signup">
              <Button size="lg" className="bg-accent text-accent-foreground hover:bg-accent/90 gap-2 font-semibold">
                Register Now <ArrowRight className="w-4 h-4" />
              </Button>
            </Link>
            <Link to="/research-writing/signin">
              <Button size="lg" variant="outline" className="border-primary-foreground/30 text-primary-foreground hover:bg-primary-foreground/10">
                Sign In
              </Button>
            </Link>
          </div>
        </div>
        <div className="absolute inset-0 opacity-5" style={{ backgroundImage: "url(\"data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='1'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E\")" }} />
      </section>

      {/* The Problem */}
      <section className="py-16 md:py-20 bg-background">
        <div className="container max-w-4xl">
          <div className="text-center mb-12">
            <AlertTriangle className="w-8 h-8 text-accent mx-auto mb-4" />
            <h2 className="font-serif text-2xl sm:text-3xl font-bold text-foreground">The Problem in Postgraduate Research Writing</h2>
            <p className="text-muted-foreground mt-3 max-w-2xl mx-auto text-sm sm:text-base">
              Many MSc and PhD students reach the writing stage without the skills to communicate their findings effectively.
            </p>
          </div>
          <div className="grid md:grid-cols-2 gap-6">
            <Card className="border-destructive/20">
              <CardContent className="pt-6 space-y-3">
                <p className="font-medium text-foreground">Students often cannot…</p>
                <ul className="space-y-2.5 text-sm text-muted-foreground">
                  <li className="flex gap-2"><span className="text-destructive font-bold">✗</span> Distinguish between results and discussion</li>
                  <li className="flex gap-2"><span className="text-destructive font-bold">✗</span> Interpret their own statistical output</li>
                  <li className="flex gap-2"><span className="text-destructive font-bold">✗</span> Present numerical evidence with proper context</li>
                  <li className="flex gap-2"><span className="text-destructive font-bold">✗</span> Explain what statistical results mean in biological or practical terms</li>
                  <li className="flex gap-2"><span className="text-destructive font-bold">✗</span> Defend findings under examination</li>
                </ul>
              </CardContent>
            </Card>
            <Card className="border-primary/20">
              <CardContent className="pt-6 space-y-3">
                <p className="font-medium text-foreground">FIA teaches students to…</p>
                <ul className="space-y-2.5 text-sm text-muted-foreground">
                  <li className="flex gap-2"><CheckCircle className="w-4 h-4 text-primary shrink-0 mt-0.5" /> Think statistically about their data</li>
                  <li className="flex gap-2"><CheckCircle className="w-4 h-4 text-primary shrink-0 mt-0.5" /> Write evidence-based results sections</li>
                  <li className="flex gap-2"><CheckCircle className="w-4 h-4 text-primary shrink-0 mt-0.5" /> Interpret tables and figures critically</li>
                  <li className="flex gap-2"><CheckCircle className="w-4 h-4 text-primary shrink-0 mt-0.5" /> Translate statistical findings into meaningful conclusions</li>
                  <li className="flex gap-2"><CheckCircle className="w-4 h-4 text-primary shrink-0 mt-0.5" /> Prepare for viva voce examination</li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Tracks */}
      <section className="py-16 md:py-20 bg-secondary/30">
        <div className="container max-w-5xl">
          <div className="text-center mb-12">
            <h2 className="font-serif text-2xl sm:text-3xl font-bold text-foreground">Programme Tracks</h2>
            <p className="text-muted-foreground mt-2 text-sm sm:text-base">Structured pathways matched to your academic level and research requirements</p>
          </div>
          <div className="grid sm:grid-cols-2 gap-6">
            {TRACKS.map((t) => (
              <Card key={t.title} className="hover:border-primary/30 transition-colors">
                <CardContent className="pt-6">
                  <div className="flex items-start gap-4">
                    <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                      <t.icon className="w-5 h-5 text-primary" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-start justify-between gap-2 mb-2">
                        <h3 className="font-serif font-semibold text-foreground text-sm sm:text-base">{t.title}</h3>
                        <span className="text-xs px-2 py-1 rounded-full bg-primary/10 text-primary font-medium shrink-0">{t.duration}</span>
                      </div>
                      <p className="text-sm text-muted-foreground leading-relaxed">{t.desc}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Learning Modules */}
      <section className="py-16 md:py-20 bg-background">
        <div className="container max-w-5xl">
          <div className="text-center mb-12">
            <h2 className="font-serif text-2xl sm:text-3xl font-bold text-foreground">Core Learning Modules</h2>
            <p className="text-muted-foreground mt-2 text-sm sm:text-base">A comprehensive suite of tools designed to build research competence at every stage</p>
          </div>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {MODULES.map((m, i) => (
              <Card key={m.title} className={`transition-colors ${m.available ? "border-primary/30 shadow-md" : "opacity-70"}`}>
                <CardContent className="pt-6">
                  <div className="w-11 h-11 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                    <m.icon className="w-5 h-5 text-primary" />
                  </div>
                  <div className="flex items-center gap-2 mb-2">
                    <h3 className="font-serif font-semibold text-foreground text-sm sm:text-base">{m.title}</h3>
                  </div>
                  {!m.available && (
                    <span className="inline-block text-[10px] px-2 py-0.5 rounded-full bg-muted text-muted-foreground font-medium mb-2">Coming Soon</span>
                  )}
                  {m.available && (
                    <span className="inline-block text-[10px] px-2 py-0.5 rounded-full bg-primary/10 text-primary font-medium mb-2">Available Now</span>
                  )}
                  <p className="text-sm text-muted-foreground leading-relaxed">{m.desc}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Common Mistakes */}
      <section className="py-16 md:py-20 bg-secondary/30">
        <div className="container max-w-4xl">
          <div className="text-center mb-12">
            <h2 className="font-serif text-2xl sm:text-3xl font-bold text-foreground">Common Research Writing Mistakes FIA Addresses</h2>
            <p className="text-muted-foreground mt-2 text-sm sm:text-base">These are the errors examiners frequently identify in theses and dissertations</p>
          </div>
          <div className="grid sm:grid-cols-2 gap-4">
            {MISTAKES.map((m, i) => (
              <div key={i} className="flex items-start gap-3 p-4 rounded-lg bg-card border border-border">
                <AlertTriangle className="w-4 h-4 text-accent shrink-0 mt-0.5" />
                <p className="text-sm text-foreground leading-relaxed">{m}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Why FIA */}
      <section className="py-16 md:py-20 bg-background">
        <div className="container max-w-4xl">
          <div className="text-center mb-12">
            <h2 className="font-serif text-2xl sm:text-3xl font-bold text-foreground">Why FIA Is Different</h2>
            <p className="text-muted-foreground mt-2 text-sm sm:text-base">Three principles that define our approach to research training</p>
          </div>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-14 h-14 rounded-full bg-primary/10 flex items-center justify-center mx-auto mb-4">
                <GraduationCap className="w-7 h-7 text-primary" />
              </div>
              <h3 className="font-serif font-semibold text-foreground mb-2">Professor-Led Curriculum</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">Designed by experienced academics who understand how research is evaluated and what examiners expect.</p>
            </div>
            <div className="text-center">
              <div className="w-14 h-14 rounded-full bg-primary/10 flex items-center justify-center mx-auto mb-4">
                <Brain className="w-7 h-7 text-primary" />
              </div>
              <h3 className="font-serif font-semibold text-foreground mb-2">AI-Assisted, Not AI-Written</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">AI provides guidance, prompts, and feedback while students produce their own academic writing.</p>
            </div>
            <div className="text-center">
              <div className="w-14 h-14 rounded-full bg-primary/10 flex items-center justify-center mx-auto mb-4">
                <Shield className="w-7 h-7 text-primary" />
              </div>
              <h3 className="font-serif font-semibold text-foreground mb-2">Academic Integrity First</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">The platform reinforces independent learning and responsible research conduct at every step.</p>
            </div>
          </div>
        </div>
      </section>

      {/* For Students & Supervisors */}
      <section className="py-16 md:py-20 bg-secondary/30">
        <div className="container max-w-5xl">
          <div className="grid md:grid-cols-2 gap-8">
            <Card>
              <CardContent className="pt-6">
                <GraduationCap className="w-8 h-8 text-primary mb-4" />
                <h3 className="font-serif text-xl font-bold text-foreground mb-4">For Students</h3>
                <ul className="space-y-3 text-sm text-muted-foreground">
                  <li className="flex gap-2"><CheckCircle className="w-4 h-4 text-primary shrink-0 mt-0.5" /> Diagnostic assessment to identify knowledge gaps</li>
                  <li className="flex gap-2"><CheckCircle className="w-4 h-4 text-primary shrink-0 mt-0.5" /> Personalised learning pathway by academic level</li>
                  <li className="flex gap-2"><CheckCircle className="w-4 h-4 text-primary shrink-0 mt-0.5" /> AI-guided writing prompts — not ready-made text</li>
                  <li className="flex gap-2"><CheckCircle className="w-4 h-4 text-primary shrink-0 mt-0.5" /> Progress tracking from proposal to defense</li>
                </ul>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <Users className="w-8 h-8 text-primary mb-4" />
                <h3 className="font-serif text-xl font-bold text-foreground mb-4">For Supervisors</h3>
                <ul className="space-y-3 text-sm text-muted-foreground">
                  <li className="flex gap-2"><CheckCircle className="w-4 h-4 text-primary shrink-0 mt-0.5" /> Structured student progress monitoring</li>
                  <li className="flex gap-2"><CheckCircle className="w-4 h-4 text-primary shrink-0 mt-0.5" /> Milestone-based accountability framework</li>
                  <li className="flex gap-2"><CheckCircle className="w-4 h-4 text-primary shrink-0 mt-0.5" /> Structured summaries to support supervisor-student discussions</li>
                  <li className="flex gap-2"><CheckCircle className="w-4 h-4 text-primary shrink-0 mt-0.5" /> Coming soon: Supervisor dashboard</li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* The FIA Research Journey */}
      <section className="py-16 md:py-20 bg-background">
        <div className="container max-w-4xl">
          <div className="text-center mb-12">
            <h2 className="font-serif text-2xl sm:text-3xl font-bold text-foreground">The FIA Research Journey</h2>
            <p className="text-muted-foreground mt-2 text-sm sm:text-base">A structured pathway from diagnostic assessment to certification of competence</p>
          </div>
          <div className="relative">
            {/* Vertical connector line */}
            <div className="absolute left-5 md:left-6 top-6 bottom-6 w-px bg-primary/20 hidden sm:block" />
            <div className="space-y-6">
              {JOURNEY_STEPS.map((s) => (
                <div key={s.step} className="flex gap-4 sm:gap-6 items-start relative">
                  <div className="w-10 h-10 md:w-12 md:h-12 rounded-full bg-primary/10 border-2 border-primary/30 flex items-center justify-center shrink-0 z-10 bg-background">
                    <s.icon className="w-5 h-5 text-primary" />
                  </div>
                  <div className="flex-1 pb-2">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs font-semibold text-primary">Step {s.step}</span>
                    </div>
                    <h3 className="font-serif font-semibold text-foreground text-sm sm:text-base">{s.title}</h3>
                    <p className="text-sm text-muted-foreground leading-relaxed mt-1">{s.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* FAQ */}
      <section className="py-16 md:py-20 bg-secondary/30">
        <div className="container max-w-3xl">
          <div className="text-center mb-12">
            <h2 className="font-serif text-2xl sm:text-3xl font-bold text-foreground">Frequently Asked Questions</h2>
          </div>
          <div className="space-y-3">
            {FAQS.map((faq, i) => (
              <button
                key={i}
                onClick={() => setOpenFaq(openFaq === i ? null : i)}
                className="w-full text-left p-4 sm:p-5 rounded-lg border border-border bg-card hover:border-primary/30 transition-colors"
              >
                <div className="flex items-center justify-between gap-3">
                  <span className="font-medium text-sm sm:text-base text-foreground">{faq.q}</span>
                  <ChevronDown className={`w-4 h-4 text-muted-foreground transition-transform shrink-0 ${openFaq === i ? "rotate-180" : ""}`} />
                </div>
                {openFaq === i && (
                  <p className="text-sm text-muted-foreground mt-3 leading-relaxed">{faq.a}</p>
                )}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-16 md:py-20" style={{ background: "var(--gradient-hero)" }}>
        <div className="container max-w-3xl text-center text-primary-foreground">
          <h2 className="font-serif text-2xl sm:text-3xl font-bold">Begin Your Research Journey</h2>
          <p className="text-primary-foreground/80 mt-4 max-w-xl mx-auto text-sm sm:text-base leading-relaxed">
            Join researchers who are learning to think critically, analyse rigorously, and write with confidence.
          </p>
          <div className="flex flex-wrap gap-4 justify-center mt-8">
            <Link to="/research-writing/signup">
              <Button size="lg" className="bg-accent text-accent-foreground hover:bg-accent/90 gap-2 font-semibold">
                Register Now <ArrowRight className="w-4 h-4" />
              </Button>
            </Link>
            <Link to="/research-writing/signin">
              <Button size="lg" variant="outline" className="border-primary-foreground/30 text-primary-foreground hover:bg-primary-foreground/10">
                Sign In
              </Button>
            </Link>
          </div>
        </div>
      </section>
    </Layout>
  );
}
