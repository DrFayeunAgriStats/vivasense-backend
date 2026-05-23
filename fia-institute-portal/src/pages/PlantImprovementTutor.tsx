import { useState } from "react";
import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Sheet, SheetContent, SheetTrigger, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { useToast } from "@/hooks/use-toast";
import { streamPlantImprovementChat, type ChatMessage } from "@/lib/plantImprovementStreamChat";
import { SmartQuiz } from "@/components/tutor/SmartQuiz";
import {
  BookOpen, Globe2, Sprout, MapPin, Wheat, Dna, Ship, Flag, HelpCircle,
  Sparkles, Send, Download, MessageCircle, ChevronRight, Bot, User as UserIcon,
  Menu, Lightbulb,
} from "lucide-react";


// ============================================================
// Data
// ============================================================
const modules = [
  { id: "intro", icon: BookOpen, title: "Introduction", desc: "Why crop origin and domestication matter in plant improvement." },
  { id: "origin", icon: Globe2, title: "Centre of Origin", desc: "Geographic regions where crops first evolved from wild ancestors." },
  { id: "vavilov", icon: Sprout, title: "Vavilov's Theory", desc: "The eight centres of crop diversity proposed by N.I. Vavilov." },
  { id: "diversity", icon: MapPin, title: "Centres of Diversity", desc: "Primary vs secondary centres and their breeding value." },
  { id: "domestication", icon: Wheat, title: "Domestication", desc: "From wild plants to cultivated crops — the human-plant journey." },
  { id: "syndrome", icon: Dna, title: "Domestication Syndrome", desc: "The suite of traits separating crops from their wild relatives." },
  { id: "introduction", icon: Ship, title: "Plant Introduction", desc: "Movement of germplasm across regions for crop improvement." },
  { id: "nigeria", icon: Flag, title: "Nigerian Case Studies", desc: "Cassava, yam, cowpea — origin and introduction in Nigeria." },
  { id: "quiz", icon: HelpCircle, title: "Revision Quiz", desc: "Test your mastery with interactive multiple-choice questions." },
];

const wildVsDomesticated = [
  { trait: "Seed shattering", wild: "Shatters easily (dispersal)", domesticated: "Non-shattering (retained for harvest)" },
  { trait: "Seed dormancy", wild: "Strong dormancy", domesticated: "Reduced/uniform germination" },
  { trait: "Plant architecture", wild: "Branched, spreading", domesticated: "Compact, erect" },
  { trait: "Seed/fruit size", wild: "Small", domesticated: "Large, uniform" },
  { trait: "Pigmentation", wild: "Often dark, bitter", domesticated: "Lighter, palatable" },
  { trait: "Photoperiod", wild: "Strong sensitivity", domesticated: "Day-neutral varieties" },
];

const cropOrigins = [
  { crop: "Wheat", origin: "Near East (Fertile Crescent)", year: "~10,000 BP" },
  { crop: "Rice", origin: "Indo-Chinese / Yangtze Valley", year: "~9,000 BP" },
  { crop: "Maize", origin: "Mexico (Mesoamerica)", year: "~9,000 BP" },
  { crop: "Cassava", origin: "South America (Brazil/Amazon)", year: "~8,000 BP" },
  { crop: "Yam", origin: "West Africa", year: "~7,000 BP" },
  { crop: "Cowpea", origin: "West/Central Africa", year: "~6,000 BP" },
  { crop: "Sorghum", origin: "Ethiopia / NE Africa", year: "~6,000 BP" },
  { crop: "Potato", origin: "Andes (Peru/Bolivia)", year: "~8,000 BP" },
];

const nigerianCrops = [
  { crop: "Cassava", note: "Introduced from Brazil by Portuguese (16th c.). Now Nigeria leads global production." },
  { crop: "Yam", note: "Indigenous West African origin. Nigeria within the 'yam belt' produces ~70% of world output." },
  { crop: "Cowpea", note: "Domesticated in West Africa. Nigeria is the world's largest producer." },
  { crop: "Maize", note: "Introduced from the Americas via trans-Atlantic exchange (15th–16th c.)." },
  { crop: "Rice", note: "Two species: African rice (Oryza glaberrima, indigenous) & Asian rice (O. sativa, introduced)." },
];

const timeline = [
  { year: "~10,000 BP", title: "Fertile Crescent", note: "Wheat, barley, lentil, pea — earliest known agriculture." },
  { year: "~9,000 BP", title: "Mesoamerica", note: "Maize domesticated from teosinte in southern Mexico." },
  { year: "~9,000 BP", title: "Yangtze Valley", note: "Rice (Oryza sativa) domesticated in China." },
  { year: "~8,000 BP", title: "Andes", note: "Potato, quinoa domesticated in highland South America." },
  { year: "~7,000 BP", title: "West Africa", note: "Yam (Dioscorea) and African rice (O. glaberrima) domesticated." },
  { year: "~6,000 BP", title: "Ethiopian Plateau", note: "Sorghum, teff, finger millet emerge." },
  { year: "1492 CE", title: "Columbian Exchange", note: "Maize, cassava, cocoa move from Americas to Africa and beyond." },
  { year: "1926 CE", title: "Vavilov Centres", note: "N.I. Vavilov publishes the 8 (later 11) centres of origin." },
];

const didYouKnow = [
  "Vavilov led 115 expeditions across 5 continents and assembled >250,000 plant accessions before his death in 1943.",
  "A single Tb1 gene change largely explains why maize grows as one tall stalk while wild teosinte is bushy.",
  "Sweet cassava varieties carry up to 10× less cyanogenic glycosides than bitter wild cassava.",
  "Nigeria produces over 60% of the world's yam — sitting at the heart of the West African yam belt.",
  "Modern wheat is hexaploid (AABBDD), the result of two natural hybridisation events with wild grasses.",
];

// Mock AI responses keyed by keyword
const WELCOME_TEXT =
  "Hello! I'm your **Plant Improvement AI Tutor** for CSP 502.\n\nAsk me about **Centres of Origin**, **Vavilov's theory**, **Domestication & the Domestication Syndrome**, **Plant Introduction**, or **Nigerian crop case studies** (cassava, yam, cowpea, maize, rice, oil palm).";

// ============================================================
// Component
// ============================================================
export default function PlantImprovementTutor() {
  const { toast } = useToast();
  const [activeModule, setActiveModule] = useState("intro");
  const [chat, setChat] = useState<{ role: "user" | "ai"; text: string }[]>([
    { role: "ai", text: WELCOME_TEXT },
  ]);
  const [input, setInput] = useState("");
  const [typing, setTyping] = useState(false);

  // Quiz progress (lifted from SmartQuiz)
  const [quizProgress, setQuizProgress] = useState({ answered: 0, total: 0 });
  const answeredCount = quizProgress.answered;
  const totalQuestions = quizProgress.total || 8;
  const progress = totalQuestions ? (answeredCount / totalQuestions) * 100 : 0;

  const sendMessage = async (text?: string) => {
    const q = (text ?? input).trim();
    if (!q || typing) return;

    const nextChat: { role: "user" | "ai"; text: string }[] = [
      ...chat,
      { role: "user", text: q },
    ];
    setChat(nextChat);
    setInput("");
    setTyping(true);

    // Build history (exclude welcome) for the AI
    const history: ChatMessage[] = nextChat
      .slice(1)
      .map((m) => ({ role: m.role === "ai" ? "assistant" : "user", content: m.text }));

    let assistantSoFar = "";
    let started = false;

    await streamPlantImprovementChat({
      messages: history,
      onDelta: (chunk) => {
        assistantSoFar += chunk;
        setChat((prev) => {
          if (!started) {
            started = true;
            return [...prev, { role: "ai", text: assistantSoFar }];
          }
          const copy = [...prev];
          copy[copy.length - 1] = { role: "ai", text: assistantSoFar };
          return copy;
        });
      },
      onDone: () => setTyping(false),
      onError: (err) => {
        setTyping(false);
        toast({ title: "Tutor unavailable", description: err, variant: "destructive" });
      },
    });
  };

  const scrollTo = (id: string) => {
    setActiveModule(id);
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };


  return (
    <Layout>
      {/* ============ HERO ============ */}
      <section className="relative overflow-hidden bg-gradient-to-br from-emerald-900 via-primary to-emerald-950 text-white">
        <div
          className="absolute inset-0 opacity-20"
          style={{
            backgroundImage:
              "radial-gradient(circle at 20% 30%, rgba(255,255,255,0.15) 0, transparent 40%), radial-gradient(circle at 80% 70%, rgba(255,255,255,0.1) 0, transparent 40%)",
          }}
        />
        <div className="container-wide relative py-20 md:py-28">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="max-w-3xl"
          >
            <Badge className="mb-5 bg-white/15 text-white border-white/20 hover:bg-white/20">
              <Sparkles className="w-3.5 h-3.5 mr-1.5" /> AI Tutor · Plant Improvement
            </Badge>
            <h1 className="font-serif text-4xl md:text-6xl font-bold mb-5 leading-tight">
              Plant Improvement <span className="text-emerald-300">AI Tutor</span>
            </h1>
            <p className="text-lg md:text-xl text-white/85 leading-relaxed mb-8">
              Interactive AI-assisted learning for crop improvement students — explore
              centres of origin, domestication, and plant introduction with guided modules,
              an AI mentor, and self-check quizzes.
            </p>
            <div className="flex flex-wrap gap-3">
              <Button size="lg" onClick={() => scrollTo("intro")} className="bg-white text-emerald-900 hover:bg-emerald-50">
                Start Learning <ChevronRight className="w-4 h-4 ml-1" />
              </Button>
              <Button size="lg" variant="outline" onClick={() => scrollTo("ai-panel")} className="border-white/40 text-white hover:bg-white/10 bg-transparent">
                <MessageCircle className="w-4 h-4 mr-1" /> Ask AI Tutor
              </Button>
              <Button size="lg" variant="outline" className="border-white/40 text-white hover:bg-white/10 bg-transparent" asChild>
                <a href="/CSP502_Centre_of_Origin_Domestication_PlantIntroduction.pptx" download>
                  <Download className="w-4 h-4 mr-1" /> Download Slides
                </a>
              </Button>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ============ MAIN GRID ============ */}
      <section className="bg-background py-12 md:py-16">
        {/* MOBILE MODULE DRAWER */}
        <div className="container-wide lg:hidden mb-6 flex items-center justify-between gap-3">
          <Sheet>
            <SheetTrigger asChild>
              <Button variant="outline" className="border-emerald-300 text-emerald-800">
                <Menu className="w-4 h-4 mr-2" /> Modules
              </Button>
            </SheetTrigger>
            <SheetContent side="left" className="w-72">
              <SheetHeader>
                <SheetTitle className="font-serif">Course Modules</SheetTitle>
              </SheetHeader>
              <nav className="mt-4 space-y-1">
                {modules.map((m) => {
                  const Icon = m.icon;
                  return (
                    <button
                      key={m.id}
                      onClick={() => scrollTo(m.id)}
                      className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-left text-foreground hover:bg-emerald-100/60 transition"
                    >
                      <Icon className="w-4 h-4 flex-shrink-0 text-emerald-700" />
                      <span className="truncate">{m.title}</span>
                    </button>
                  );
                })}
              </nav>
              <div className="mt-5 pt-5 border-t">
                <div className="flex items-center justify-between mb-2 text-xs text-muted-foreground">
                  <span>Quiz Progress</span>
                  <span className="font-semibold text-emerald-700">{answeredCount}/{totalQuestions}</span>
                </div>
                <Progress value={progress} className="h-2" />
              </div>
            </SheetContent>
          </Sheet>
          <Badge className="bg-emerald-100 text-emerald-800 hover:bg-emerald-100">
            {answeredCount}/{totalQuestions} answered
          </Badge>
        </div>

        <div className="container-wide grid lg:grid-cols-[260px_1fr] gap-10">
          {/* SIDEBAR */}
          <aside className="hidden lg:block">
            <div className="sticky top-24">
              <Card className="p-5 border-emerald-100 bg-gradient-to-b from-emerald-50/50 to-white">
                <h3 className="font-serif text-sm uppercase tracking-wider text-emerald-800 mb-4">
                  Course Modules
                </h3>
                <nav className="space-y-1">
                  {modules.map((m) => {
                    const Icon = m.icon;
                    const active = activeModule === m.id;
                    return (
                      <button
                        key={m.id}
                        onClick={() => scrollTo(m.id)}
                        className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-left transition-all ${
                          active
                            ? "bg-primary text-primary-foreground shadow-sm"
                            : "text-foreground hover:bg-emerald-100/60"
                        }`}
                      >
                        <Icon className="w-4 h-4 flex-shrink-0" />
                        <span className="truncate">{m.title}</span>
                      </button>
                    );
                  })}
                </nav>
                <div className="mt-5 pt-5 border-t border-emerald-100">
                  <div className="flex items-center justify-between mb-2 text-xs text-muted-foreground">
                    <span>Quiz Progress</span>
                    <span className="font-semibold text-emerald-700">{answeredCount}/{totalQuestions}</span>
                  </div>
                  <Progress value={progress} className="h-2" />
                </div>
              </Card>
            </div>
          </aside>

          {/* CONTENT */}
          <div className="space-y-16">
            {/* MODULE CARDS */}
            <div id="intro">
              <h2 className="font-serif text-3xl font-bold text-foreground mb-2">Learning Modules</h2>
              <p className="text-muted-foreground mb-8">Nine guided modules covering crop origin to domestication.</p>
              <div className="grid sm:grid-cols-2 xl:grid-cols-3 gap-5">
                {modules.map((m, i) => {
                  const Icon = m.icon;
                  return (
                    <motion.div
                      key={m.id}
                      initial={{ opacity: 0, y: 12 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.35, delay: i * 0.04 }}
                      whileHover={{ y: -4 }}
                    >
                      <Card className="p-6 h-full border-emerald-100 hover:border-emerald-300 hover:shadow-lg hover:shadow-emerald-100/50 transition-all group">
                        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-emerald-100 to-emerald-50 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                          <Icon className="w-6 h-6 text-emerald-700" />
                        </div>
                        <h3 className="font-serif text-lg font-bold mb-2">{m.title}</h3>
                        <p className="text-sm text-muted-foreground mb-4 leading-relaxed">{m.desc}</p>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => scrollTo(m.id)}
                          className="text-emerald-700 hover:text-emerald-800 hover:bg-emerald-50 px-0"
                        >
                          Open Module <ChevronRight className="w-4 h-4 ml-1" />
                        </Button>
                      </Card>
                    </motion.div>
                  );
                })}
              </div>
            </div>

            {/* AI TUTOR PANEL */}
            <div id="ai-panel">
              <h2 className="font-serif text-3xl font-bold text-foreground mb-2">AI Tutor Panel</h2>
              <p className="text-muted-foreground mb-6">Chat with your AI mentor for instant academic guidance.</p>
              <Card className="overflow-hidden border-emerald-200 shadow-xl shadow-emerald-100/40">
                <div className="bg-gradient-to-r from-emerald-700 to-primary text-white px-5 py-4 flex items-center gap-3">
                  <div className="relative">
                    <div className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center backdrop-blur">
                      <Bot className="w-5 h-5" />
                    </div>
                    <span className="absolute -bottom-0.5 -right-0.5 w-3 h-3 bg-emerald-300 rounded-full ring-2 ring-emerald-700 animate-pulse" />
                  </div>
                  <div>
                    <p className="font-semibold">Plant Improvement Tutor</p>
                    <p className="text-xs text-white/75">AI · Online · Educational use</p>
                  </div>
                </div>

                <div className="bg-gradient-to-b from-emerald-50/30 to-white p-5 max-h-[420px] overflow-y-auto space-y-4">
                  {chat.map((m, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={`flex gap-3 ${m.role === "user" ? "justify-end" : ""}`}
                    >
                      {m.role === "ai" && (
                        <div className="w-8 h-8 rounded-full bg-emerald-100 flex items-center justify-center flex-shrink-0">
                          <Bot className="w-4 h-4 text-emerald-700" />
                        </div>
                      )}
                      <div
                        className={`max-w-[78%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                          m.role === "user"
                            ? "bg-primary text-primary-foreground rounded-br-sm"
                            : "bg-white border border-emerald-100 shadow-sm rounded-bl-sm"
                        }`}
                      >
                        <div className="prose prose-sm max-w-none prose-p:my-2 prose-headings:mt-3 prose-headings:mb-1 prose-strong:text-inherit">
                          <ReactMarkdown>{m.text}</ReactMarkdown>
                        </div>
                      </div>
                      {m.role === "user" && (
                        <div className="w-8 h-8 rounded-full bg-emerald-700 flex items-center justify-center flex-shrink-0">
                          <UserIcon className="w-4 h-4 text-white" />
                        </div>
                      )}
                    </motion.div>
                  ))}
                  {typing && (
                    <div className="flex gap-3">
                      <div className="w-8 h-8 rounded-full bg-emerald-100 flex items-center justify-center">
                        <Bot className="w-4 h-4 text-emerald-700" />
                      </div>
                      <div className="bg-white border border-emerald-100 rounded-2xl px-4 py-3 flex gap-1">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-bounce" />
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-bounce [animation-delay:0.15s]" />
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-bounce [animation-delay:0.3s]" />
                      </div>
                    </div>
                  )}
                </div>

                <div className="border-t border-emerald-100 bg-white p-4">
                  <div className="flex flex-wrap gap-2 mb-3">
                    {[
                      "Why are centres of origin important?",
                      "Explain Vavilov's theory",
                      "What is the domestication syndrome?",
                    ].map((s) => (
                      <button
                        key={s}
                        onClick={() => sendMessage(s)}
                        className="text-xs px-3 py-1.5 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-800 hover:bg-emerald-100 transition"
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                  <div className="flex gap-2">
                    <Input
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && sendMessage()}
                      placeholder="Ask the AI tutor a question..."
                      className="border-emerald-200 focus-visible:ring-emerald-500"
                    />
                    <Button onClick={() => sendMessage()} className="bg-emerald-700 hover:bg-emerald-800">
                      <Send className="w-4 h-4" />
                    </Button>
                  </div>
                  <p className="text-[10px] text-muted-foreground mt-2 text-center">
                    Powered by FIA AI · Subject scope: CSP 502 — Centre of Origin, Domestication & Plant Introduction · © Dr. Fayeun Lawrence Stephen
                  </p>
                </div>
              </Card>
            </div>

            {/* DOMESTICATION SYNDROME TABLE */}
            <div id="syndrome">
              <h2 className="font-serif text-3xl font-bold mb-2">Domestication Syndrome</h2>
              <p className="text-muted-foreground mb-6">Comparison of wild vs domesticated trait expression.</p>
              <Card className="overflow-hidden border-emerald-100">
                <table className="w-full text-sm">
                  <thead className="bg-emerald-700 text-white">
                    <tr>
                      <th className="text-left px-4 py-3 font-semibold">Trait</th>
                      <th className="text-left px-4 py-3 font-semibold">Wild Ancestor</th>
                      <th className="text-left px-4 py-3 font-semibold">Domesticated Crop</th>
                    </tr>
                  </thead>
                  <tbody>
                    {wildVsDomesticated.map((r, i) => (
                      <tr key={r.trait} className={i % 2 ? "bg-emerald-50/40" : "bg-white"}>
                        <td className="px-4 py-3 font-medium">{r.trait}</td>
                        <td className="px-4 py-3 text-muted-foreground">{r.wild}</td>
                        <td className="px-4 py-3 text-emerald-800">{r.domesticated}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            </div>

            {/* CROP ORIGIN CARDS */}
            <div id="origin">
              <h2 className="font-serif text-3xl font-bold mb-2">Crop Origin Cards</h2>
              <p className="text-muted-foreground mb-6">Major crops and their evolutionary homelands.</p>
              <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {cropOrigins.map((c, i) => (
                  <motion.div
                    key={c.crop}
                    initial={{ opacity: 0, scale: 0.95 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.3, delay: i * 0.04 }}
                  >
                    <Card className="p-5 border-emerald-100 hover:border-emerald-300 hover:-translate-y-1 transition-all">
                      <Wheat className="w-6 h-6 text-emerald-600 mb-3" />
                      <h4 className="font-serif font-bold text-lg">{c.crop}</h4>
                      <p className="text-xs text-emerald-700 font-medium mt-1">{c.year}</p>
                      <p className="text-sm text-muted-foreground mt-2">{c.origin}</p>
                    </Card>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* NIGERIAN CROPS */}
            <div id="nigeria">
              <h2 className="font-serif text-3xl font-bold mb-2">Nigerian Case Studies</h2>
              <p className="text-muted-foreground mb-6">Indigenous and introduced crops shaping Nigerian agriculture.</p>
              <div className="grid md:grid-cols-2 gap-4">
                {nigerianCrops.map((c) => (
                  <Card key={c.crop} className="p-5 border-l-4 border-l-emerald-600 hover:shadow-md transition">
                    <div className="flex items-start gap-3">
                      <Flag className="w-5 h-5 text-emerald-700 mt-0.5" />
                      <div>
                        <h4 className="font-serif font-bold text-lg">{c.crop}</h4>
                        <p className="text-sm text-muted-foreground mt-1 leading-relaxed">{c.note}</p>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            </div>

            {/* DOMESTICATION TIMELINE */}
            <div id="timeline">
              <h2 className="font-serif text-3xl font-bold mb-2">Domestication Timeline</h2>
              <p className="text-muted-foreground mb-6">A walk through the great milestones of crop evolution.</p>
              <div className="relative pl-6 sm:pl-10 border-l-2 border-emerald-200 space-y-6">
                {timeline.map((t, i) => (
                  <motion.div
                    key={t.year + t.title}
                    initial={{ opacity: 0, x: -10 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.35, delay: i * 0.05 }}
                    className="relative"
                  >
                    <span className="absolute -left-[34px] sm:-left-[46px] top-1.5 w-4 h-4 rounded-full bg-emerald-600 ring-4 ring-emerald-100" />
                    <p className="text-xs font-bold text-emerald-700 uppercase tracking-wider">{t.year}</p>
                    <h4 className="font-serif text-lg font-bold mt-0.5">{t.title}</h4>
                    <p className="text-sm text-muted-foreground mt-1">{t.note}</p>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* DID YOU KNOW */}
            <div>
              <h2 className="font-serif text-3xl font-bold mb-2">Did You Know?</h2>
              <p className="text-muted-foreground mb-6">Quick facts to deepen your intuition.</p>
              <div className="grid sm:grid-cols-2 gap-4">
                {didYouKnow.map((f, i) => (
                  <Card key={i} className="p-5 border-emerald-100 bg-gradient-to-br from-amber-50/50 to-white hover:shadow-md transition">
                    <div className="flex items-start gap-3">
                      <div className="w-9 h-9 rounded-lg bg-amber-100 flex items-center justify-center flex-shrink-0">
                        <Lightbulb className="w-4 h-4 text-amber-700" />
                      </div>
                      <p className="text-sm leading-relaxed">{f}</p>
                    </div>
                  </Card>
                ))}
              </div>
            </div>

            {/* SMART QUIZ */}
            <div id="quiz">
              <div className="flex items-end justify-between mb-6 flex-wrap gap-3">
                <div>
                  <h2 className="font-serif text-3xl font-bold mb-2">Smart Revision Quiz</h2>
                  <p className="text-muted-foreground">
                    Adaptive engine — MCQ, True/False, fill-in-the-gap and scenario questions across
                    Vavilov, domestication, plant introduction, molecular evidence and Nigerian crops.
                  </p>
                </div>
              </div>
              <SmartQuiz onAnswered={(answered, total) => setQuizProgress({ answered, total })} />
            </div>
          </div>
        </div>
      </section>
    </Layout>
  );
}
