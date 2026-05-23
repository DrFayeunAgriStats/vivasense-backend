import { useState, useEffect, useRef, useCallback } from "react";
import { Layout } from "@/components/layout/Layout";
import {
  Leaf, BookOpen, Globe, FlaskConical, Sprout, Dna,
  Plane, MapPin, HelpCircle, ChevronRight, Download,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

// ── Section definitions ──────────────────────────────────────────────────────

const SECTIONS = [
  {
    id: "introduction",
    label: "Introduction",
    icon: BookOpen,
    color: "text-emerald-600",
    bg: "bg-emerald-50",
    border: "border-emerald-200",
  },
  {
    id: "centre-of-origin",
    label: "Centre of Origin",
    icon: Globe,
    color: "text-blue-600",
    bg: "bg-blue-50",
    border: "border-blue-200",
  },
  {
    id: "vavilovs-theory",
    label: "Vavilov's Theory",
    icon: FlaskConical,
    color: "text-violet-600",
    bg: "bg-violet-50",
    border: "border-violet-200",
  },
  {
    id: "centres-of-diversity",
    label: "Centres of Diversity",
    icon: MapPin,
    color: "text-amber-600",
    bg: "bg-amber-50",
    border: "border-amber-200",
  },
  {
    id: "domestication",
    label: "Domestication",
    icon: Sprout,
    color: "text-lime-600",
    bg: "bg-lime-50",
    border: "border-lime-200",
  },
  {
    id: "domestication-syndrome",
    label: "Domestication Syndrome",
    icon: Dna,
    color: "text-rose-600",
    bg: "bg-rose-50",
    border: "border-rose-200",
  },
  {
    id: "plant-introduction",
    label: "Plant Introduction",
    icon: Plane,
    color: "text-sky-600",
    bg: "bg-sky-50",
    border: "border-sky-200",
  },
  {
    id: "nigerian-case-studies",
    label: "Nigerian Case Studies",
    icon: MapPin,
    color: "text-orange-600",
    bg: "bg-orange-50",
    border: "border-orange-200",
  },
  {
    id: "revision-quiz",
    label: "Revision Quiz",
    icon: HelpCircle,
    color: "text-indigo-600",
    bg: "bg-indigo-50",
    border: "border-indigo-200",
  },
] as const;

type SectionId = (typeof SECTIONS)[number]["id"];

// ── Quiz data ─────────────────────────────────────────────────────────────────

const QUIZ_QUESTIONS = [
  {
    q: "Who developed the theory of Centres of Origin for cultivated plants?",
    options: ["Charles Darwin", "Nikolai Vavilov", "Gregor Mendel", "Alphonse de Candolle"],
    answer: 1,
  },
  {
    q: "Which of the following is a trait associated with the Domestication Syndrome in cereals?",
    options: ["Non-shattering rachis", "Increased seed dormancy", "Smaller seed size", "Bitter taste"],
    answer: 0,
  },
  {
    q: "The centre of origin of maize (Zea mays) is:",
    options: ["South America (Andes)", "Mesoamerica (Mexico/Guatemala)", "West Africa", "East Asia"],
    answer: 1,
  },
  {
    q: "Plant introduction refers to:",
    options: [
      "The genetic modification of a plant species",
      "The transfer of plant material from one region to another",
      "Crossbreeding of two species",
      "Inducing mutations using chemicals",
    ],
    answer: 1,
  },
  {
    q: "Which crop is considered to have originated in Nigeria/West Africa?",
    options: ["Rice (Oryza sativa)", "African yam (Dioscorea rotundata)", "Wheat", "Soybean"],
    answer: 1,
  },
];

// ── Quiz component ────────────────────────────────────────────────────────────

function RevisionQuiz() {
  const [answers, setAnswers] = useState<Record<number, number>>({});
  const [submitted, setSubmitted] = useState(false);
  const score = submitted
    ? QUIZ_QUESTIONS.filter((q, i) => answers[i] === q.answer).length
    : 0;

  return (
    <div className="space-y-6">
      {QUIZ_QUESTIONS.map((q, qi) => (
        <div key={qi} className="rounded-xl border border-border bg-card p-5 space-y-3">
          <p className="font-medium text-sm">
            {qi + 1}. {q.q}
          </p>
          <div className="grid gap-2">
            {q.options.map((opt, oi) => {
              const selected = answers[qi] === oi;
              const isCorrect = oi === q.answer;
              let optClass =
                "flex items-center gap-2 rounded-lg border px-4 py-2.5 text-sm cursor-pointer transition-colors";
              if (!submitted) {
                optClass += selected
                  ? " border-primary bg-primary/10 text-primary font-medium"
                  : " border-border hover:bg-muted/60";
              } else {
                if (isCorrect) optClass += " border-emerald-500 bg-emerald-50 text-emerald-800 font-medium";
                else if (selected && !isCorrect) optClass += " border-rose-400 bg-rose-50 text-rose-700 line-through";
                else optClass += " border-border text-muted-foreground";
              }
              return (
                <label key={oi} className={optClass}>
                  <input
                    type="radio"
                    name={`q${qi}`}
                    className="sr-only"
                    disabled={submitted}
                    checked={selected}
                    onChange={() => setAnswers((prev) => ({ ...prev, [qi]: oi }))}
                  />
                  <span className="w-5 h-5 rounded-full border-2 flex items-center justify-center shrink-0"
                    style={{ borderColor: "currentColor" }}>
                    {selected && <span className="w-2.5 h-2.5 rounded-full bg-current" />}
                  </span>
                  {opt}
                </label>
              );
            })}
          </div>
        </div>
      ))}
      <div className="flex items-center gap-4">
        {!submitted ? (
          <button
            onClick={() => setSubmitted(true)}
            disabled={Object.keys(answers).length < QUIZ_QUESTIONS.length}
            className="px-6 py-2.5 rounded-lg bg-primary text-primary-foreground text-sm font-medium disabled:opacity-50 hover:bg-primary/90 transition-colors"
          >
            Submit Answers
          </button>
        ) : (
          <>
            <div className="rounded-lg border border-emerald-300 bg-emerald-50 px-5 py-2.5 text-sm font-medium text-emerald-800">
              Score: {score}/{QUIZ_QUESTIONS.length}
              {score === QUIZ_QUESTIONS.length && " 🎉 Perfect!"}
            </div>
            <button
              onClick={() => { setAnswers({}); setSubmitted(false); }}
              className="px-5 py-2.5 rounded-lg border border-border text-sm hover:bg-muted transition-colors"
            >
              Retry
            </button>
          </>
        )}
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function CSP409PlantGenetics() {
  const [activeSection, setActiveSection] = useState<SectionId>("introduction");
  const sectionRefs = useRef<Partial<Record<SectionId, HTMLElement>>>({});
  const observerRef = useRef<IntersectionObserver | null>(null);

  // Register section refs
  const setRef = useCallback((id: SectionId) => (el: HTMLElement | null) => {
    if (el) sectionRefs.current[id] = el;
  }, []);

  // Intersection Observer for active section highlighting
  useEffect(() => {
    observerRef.current = new IntersectionObserver(
      (entries) => {
        // Pick the entry with the highest intersection ratio that is intersecting
        const visible = entries.filter((e) => e.isIntersecting);
        if (visible.length === 0) return;
        const top = visible.reduce((a, b) =>
          a.intersectionRatio > b.intersectionRatio ? a : b
        );
        setActiveSection(top.target.id as SectionId);
      },
      { threshold: [0.2, 0.4, 0.6], rootMargin: "-80px 0px -30% 0px" }
    );

    SECTIONS.forEach(({ id }) => {
      const el = sectionRefs.current[id];
      if (el) observerRef.current!.observe(el);
    });

    return () => observerRef.current?.disconnect();
  }, []);

  const scrollTo = (id: SectionId) => {
    const el = sectionRefs.current[id];
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <Layout>
      {/* Hero */}
      <section className="bg-primary text-primary-foreground py-10 md:py-14">
        <div className="container-wide">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-12 h-12 rounded-full bg-primary-foreground/10 flex items-center justify-center">
              <Leaf className="w-6 h-6" />
            </div>
            <div>
              <Badge className="mb-1 bg-primary-foreground/10 text-primary-foreground border-primary-foreground/20 text-[11px] uppercase tracking-wider">
                CSP 409
              </Badge>
              <h1 className="font-serif text-2xl md:text-3xl font-bold">
                Plant Genetics &amp; Crop Improvement
              </h1>
              <p className="text-primary-foreground/70 text-sm mt-1">
                Module 1 · Origins, Domestication &amp; Introduction of Crop Plants
              </p>
            </div>
          </div>
          {/* Download & module strip */}
          <div className="mt-4 flex items-center gap-3">
            <a
              href="/CSP502_Centre_of_Origin_Domestication_PlantIntroduction.pptx"
              download
              className="inline-flex items-center gap-2 rounded-lg bg-primary-foreground/15 hover:bg-primary-foreground/25 text-primary-foreground text-sm font-medium px-4 py-2 transition-colors"
            >
              <Download className="w-4 h-4" /> Download Slides (.pptx)
            </a>
          </div>
          <div className="mt-4 grid grid-cols-3 sm:grid-cols-5 md:grid-cols-9 gap-2">
            {SECTIONS.map(({ id, label, icon: Icon, bg, color }) => (
              <button
                key={id}
                onClick={() => scrollTo(id)}
                className={cn(
                  "flex flex-col items-center gap-1 rounded-xl p-3 text-center transition-all",
                  "bg-primary-foreground/10 hover:bg-primary-foreground/20 text-primary-foreground",
                  activeSection === id && "bg-primary-foreground/25 ring-2 ring-primary-foreground/40"
                )}
              >
                <Icon className="w-4 h-4 shrink-0" />
                <span className="text-[10px] leading-tight font-medium line-clamp-2">{label}</span>
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Body: sidebar + content */}
      <div className="container-wide py-8">
        <div className="flex gap-8 items-start">
          {/* ── Sticky sidebar ── */}
          <aside className="hidden lg:block w-60 shrink-0 sticky top-[89px]">
            <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-3 px-1">
              Module Contents
            </p>
            <nav className="space-y-1">
              {SECTIONS.map(({ id, label, icon: Icon, color, bg, border }) => {
                const isActive = activeSection === id;
                return (
                  <button
                    key={id}
                    onClick={() => scrollTo(id)}
                    className={cn(
                      "w-full flex items-center gap-2.5 rounded-lg px-3 py-2.5 text-sm text-left transition-all",
                      isActive
                        ? cn("font-semibold border", bg, border, color)
                        : "text-muted-foreground hover:bg-muted hover:text-foreground"
                    )}
                  >
                    <Icon className={cn("w-4 h-4 shrink-0", isActive ? color : "text-muted-foreground")} />
                    <span className="flex-1 leading-tight">{label}</span>
                    {isActive && <ChevronRight className={cn("w-3.5 h-3.5 shrink-0", color)} />}
                  </button>
                );
              })}
            </nav>
          </aside>

          {/* ── Content sections ── */}
          <main className="flex-1 min-w-0 space-y-14">

            {/* 1. Introduction */}
            <section
              id="introduction"
              ref={setRef("introduction")}
              className="scroll-mt-24"
            >
              <SectionHeading icon={BookOpen} color="text-emerald-600" bg="bg-emerald-50" border="border-emerald-200">
                Introduction
              </SectionHeading>
              <div className="prose prose-sm max-w-none mt-4 text-foreground/90">
                <p>
                  Understanding where our crop plants came from — and how they became the food sources we depend on today — is fundamental to modern plant science and breeding. This module explores the origins of cultivated plants, the development of genetic diversity, the process of domestication, and how plant material has been intentionally moved and introduced around the world.
                </p>
                <p>
                  Crop plants have not always looked or behaved as they do today. Thousands of years of human selection, migration, and agricultural practice have shaped them from wild ancestors into the productive varieties we now recognise. The science of these transformations underpins our ability to develop new, improved cultivars for a changing climate and growing population.
                </p>
                <ul>
                  <li><strong>Key themes:</strong> geographical origins, genetic diversity, domestication, plant introductions.</li>
                  <li><strong>Why it matters:</strong> identifying centres of diversity is essential for conservation and breeding programme design.</li>
                  <li><strong>Nigerian relevance:</strong> West Africa is a primary centre for yam, sorghum, and several other staple crops.</li>
                </ul>
              </div>
            </section>

            {/* 2. Centre of Origin */}
            <section
              id="centre-of-origin"
              ref={setRef("centre-of-origin")}
              className="scroll-mt-24"
            >
              <SectionHeading icon={Globe} color="text-blue-600" bg="bg-blue-50" border="border-blue-200">
                Centre of Origin
              </SectionHeading>
              <div className="prose prose-sm max-w-none mt-4 text-foreground/90">
                <p>
                  The <strong>centre of origin</strong> of a cultivated plant is the geographical region where it was first domesticated from its wild progenitor. It is the area where the wild ancestor still grows naturally, and where the earliest archaeological evidence of cultivation has been found.
                </p>
                <p>
                  The concept was first systematically developed by Swiss botanist <strong>Alphonse de Candolle</strong> (1882) in his work <em>Origin of Cultivated Plants</em>, which used historical, linguistic, and botanical evidence to propose regions of origin for more than 200 crop species.
                </p>
                <InfoBox title="Distinguishing Terms">
                  <ul>
                    <li><strong>Centre of Origin:</strong> where domestication first occurred.</li>
                    <li><strong>Centre of Diversity:</strong> where the greatest genetic variation in a crop is found (may or may not coincide with the centre of origin).</li>
                    <li><strong>Secondary Centre:</strong> a region of high diversity that developed after the crop spread from its original centre.</li>
                  </ul>
                </InfoBox>
                <p>
                  Identifying the centre of origin has practical value: wild relatives in that region often carry disease resistance, drought tolerance, and other traits that have been lost during domestication.
                </p>
              </div>
            </section>

            {/* 3. Vavilov's Theory */}
            <section
              id="vavilovs-theory"
              ref={setRef("vavilovs-theory")}
              className="scroll-mt-24"
            >
              <SectionHeading icon={FlaskConical} color="text-violet-600" bg="bg-violet-50" border="border-violet-200">
                Vavilov's Theory
              </SectionHeading>
              <div className="prose prose-sm max-w-none mt-4 text-foreground/90">
                <p>
                  Russian botanist <strong>Nikolai Ivanovich Vavilov</strong> (1887–1943) conducted extensive plant-collecting expeditions across five continents and proposed a systematic theory of crop plant origins, published in his landmark work <em>Origin and Geography of Cultivated Plants</em> (1926).
                </p>
                <p>
                  Vavilov's central hypothesis was that <strong>centres of greatest genetic diversity in a crop species tend to coincide with, or be close to, its centre of origin</strong>. He proposed that the oldest region of cultivation would show the greatest morphological and genetic variation because more generations of mutation, recombination, and selection had occurred there.
                </p>
                <InfoBox title="Vavilov's Key Principles">
                  <ol>
                    <li>The centre of origin is a geographically limited area, usually a mountainous region.</li>
                    <li>The highest degree of genetic diversity (dominant alleles, recessive alleles, and unique traits) is found in the centre of origin.</li>
                    <li>Dominant traits tend to be concentrated at the centre; recessive traits appear more at the periphery.</li>
                    <li>Each crop species has one (or sometimes two) primary centres.</li>
                  </ol>
                </InfoBox>
                <p>
                  Vavilov also formulated the <strong>Law of Homologous Series in Hereditary Variation</strong>, which states that related species tend to display parallel variation, allowing breeders to predict what variation might be found in unexplored relatives.
                </p>
              </div>
            </section>

            {/* 4. Centres of Diversity */}
            <section
              id="centres-of-diversity"
              ref={setRef("centres-of-diversity")}
              className="scroll-mt-24"
            >
              <SectionHeading icon={MapPin} color="text-amber-600" bg="bg-amber-50" border="border-amber-200">
                Centres of Diversity
              </SectionHeading>
              <div className="prose prose-sm max-w-none mt-4 text-foreground/90">
                <p>
                  Vavilov originally identified <strong>eight primary centres of crop plant diversity</strong>, later revised by subsequent researchers to include secondary centres and micro-centres. These regions are concentrated in tropical and subtropical mountain areas where diverse microclimates and isolation promoted genetic differentiation.
                </p>
                <div className="not-prose overflow-x-auto rounded-xl border border-border mt-3">
                  <table className="w-full text-sm">
                    <thead className="bg-muted/60">
                      <tr>
                        <th className="px-4 py-2.5 text-left font-semibold">Centre</th>
                        <th className="px-4 py-2.5 text-left font-semibold">Region</th>
                        <th className="px-4 py-2.5 text-left font-semibold">Key Crops</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {[
                        ["Chinese Centre", "Central & Eastern China", "Soybean, millet, rice, sorghum"],
                        ["Indian Centre", "India, Myanmar, Assam", "Rice, sugarcane, mango, eggplant"],
                        ["Central Asian Centre", "Afghanistan, Tajikistan, NW India", "Wheat, pea, lentil, cotton"],
                        ["Near Eastern Centre", "Turkey, Iran, Caucasus", "Wheat, barley, rye, oat"],
                        ["Mediterranean Centre", "Mediterranean basin", "Oat, beet, cabbage, olive"],
                        ["Abyssinian Centre", "Ethiopia, Somalia, Eritrea", "Coffee, teff, sorghum, barley"],
                        ["South Mexican & Central American Centre", "Mexico, Guatemala", "Maize, bean, squash, cacao"],
                        ["South American Centre", "Peru, Ecuador, Bolivia", "Potato, tomato, groundnut, cassava"],
                      ].map(([centre, region, crops]) => (
                        <tr key={centre} className="hover:bg-muted/30">
                          <td className="px-4 py-2.5 font-medium">{centre}</td>
                          <td className="px-4 py-2.5 text-muted-foreground">{region}</td>
                          <td className="px-4 py-2.5">{crops}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="mt-4">
                  Notably, sub-Saharan Africa — particularly the <strong>West African Centre</strong> — was later added as a primary centre for crops such as African rice (<em>Oryza glaberrima</em>), yam (<em>Dioscorea</em> spp.), cowpea (<em>Vigna unguiculata</em>), and sorghum.
                </p>
              </div>
            </section>

            {/* 5. Domestication */}
            <section
              id="domestication"
              ref={setRef("domestication")}
              className="scroll-mt-24"
            >
              <SectionHeading icon={Sprout} color="text-lime-600" bg="bg-lime-50" border="border-lime-200">
                Domestication
              </SectionHeading>
              <div className="prose prose-sm max-w-none mt-4 text-foreground/90">
                <p>
                  <strong>Domestication</strong> is the evolutionary process by which wild plants become increasingly adapted to human cultivation and use, driven by deliberate or unconscious human selection over generations.
                </p>
                <p>
                  The domestication of crop plants began approximately <strong>10,000–12,000 years ago</strong> in the Near East (the "Fertile Crescent"), with independent domestication events occurring subsequently in East Asia, South Asia, sub-Saharan Africa, Mesoamerica, and South America.
                </p>
                <h3 className="font-semibold mt-4 mb-2">Stages of Domestication</h3>
                <ol>
                  <li><strong>Wild harvesting:</strong> humans gather wild plants with no cultivation.</li>
                  <li><strong>Incipient cultivation:</strong> selected plants are grown near habitations; unconscious selection begins.</li>
                  <li><strong>Primary domestication:</strong> key traits (seed non-shattering, larger seeds, uniform germination) are fixed.</li>
                  <li><strong>Secondary domestication:</strong> further improvement and adaptation to local conditions.</li>
                  <li><strong>Modern breeding:</strong> systematic selection and hybridisation for defined target traits.</li>
                </ol>
                <p>
                  Domestication typically resulted in a <strong>genetic bottleneck</strong>, reducing diversity relative to wild progenitors, as only a subset of wild alleles was maintained in cultivated populations.
                </p>
              </div>
            </section>

            {/* 6. Domestication Syndrome */}
            <section
              id="domestication-syndrome"
              ref={setRef("domestication-syndrome")}
              className="scroll-mt-24"
            >
              <SectionHeading icon={Dna} color="text-rose-600" bg="bg-rose-50" border="border-rose-200">
                Domestication Syndrome
              </SectionHeading>
              <div className="prose prose-sm max-w-none mt-4 text-foreground/90">
                <p>
                  The <strong>domestication syndrome</strong> refers to the suite of morphological and physiological traits that distinguish domesticated crops from their wild progenitors. These traits are not unique to a single crop; rather, strikingly similar changes have evolved independently in many different crop species — suggesting convergent selection pressures.
                </p>
                <InfoBox title="Common Domestication Syndrome Traits">
                  <div className="not-prose grid grid-cols-1 sm:grid-cols-2 gap-2 mt-2 text-sm">
                    {[
                      ["Non-shattering rachis/pod", "Seeds remain on plant for harvest"],
                      ["Increased seed/fruit size", "Greater caloric yield per plant"],
                      ["Reduced seed dormancy", "Uniform, rapid germination"],
                      ["Reduced branching / erect growth", "Higher harvestable yield"],
                      ["Loss of photoperiod sensitivity", "Wider geographical adaptation"],
                      ["Reduced anti-nutritional compounds", "Better palatability, nutrition"],
                      ["Increased harvest index", "More energy allocated to edible parts"],
                      ["Reduced lateral branching in cereals", "Single-stem forms for density planting"],
                    ].map(([trait, note]) => (
                      <div key={trait} className="rounded-lg border border-rose-100 bg-rose-50/60 px-3 py-2">
                        <p className="font-medium text-rose-800">{trait}</p>
                        <p className="text-rose-600 text-xs mt-0.5">{note}</p>
                      </div>
                    ))}
                  </div>
                </InfoBox>
                <p className="mt-4">
                  The genetic basis of many domestication syndrome traits has now been identified. For example, the <em>sh4</em> gene controlling non-shattering in rice and the <em>tb1</em> gene controlling reduced branching in maize are well-characterised domestication loci.
                </p>
              </div>
            </section>

            {/* 7. Plant Introduction */}
            <section
              id="plant-introduction"
              ref={setRef("plant-introduction")}
              className="scroll-mt-24"
            >
              <SectionHeading icon={Plane} color="text-sky-600" bg="bg-sky-50" border="border-sky-200">
                Plant Introduction
              </SectionHeading>
              <div className="prose prose-sm max-w-none mt-4 text-foreground/90">
                <p>
                  <strong>Plant introduction</strong> is the intentional or accidental transfer of plant germplasm (seeds, cuttings, propagules) from one geographical location to another for the purpose of cultivation, research, or breeding. It is one of the oldest and most important activities in crop improvement.
                </p>
                <h3 className="font-semibold mt-4 mb-2">Types of Plant Introduction</h3>
                <ul>
                  <li>
                    <strong>Primary introduction:</strong> the introduced plant is used directly in cultivation without modification — it is immediately valuable as a new variety.
                  </li>
                  <li>
                    <strong>Secondary introduction:</strong> the introduced plant is used as a parent in breeding programmes, not directly as a new variety.
                  </li>
                </ul>
                <h3 className="font-semibold mt-4 mb-2">Steps in Plant Introduction</h3>
                <ol>
                  <li>Collection and documentation of germplasm from the source region.</li>
                  <li>Phytosanitary inspection and quarantine to prevent introduction of pests/diseases.</li>
                  <li>Evaluation in the new environment (initial trials).</li>
                  <li>Selection and multiplication of promising accessions.</li>
                  <li>Release and distribution (if directly usable) or use in crossing programmes.</li>
                </ol>
                <InfoBox title="Key Institutions">
                  International germplasm banks such as the <strong>CGIAR centres</strong> (CIMMYT, IRRI, IITA, ICRISAT) maintain extensive collections of introduced germplasm and distribute it to breeders worldwide.
                </InfoBox>
              </div>
            </section>

            {/* 8. Nigerian Case Studies */}
            <section
              id="nigerian-case-studies"
              ref={setRef("nigerian-case-studies")}
              className="scroll-mt-24"
            >
              <SectionHeading icon={MapPin} color="text-orange-600" bg="bg-orange-50" border="border-orange-200">
                Nigerian Case Studies
              </SectionHeading>
              <div className="prose prose-sm max-w-none mt-4 text-foreground/90">
                <p>
                  Nigeria is located within West Africa, which is recognised as a primary centre of domestication for several economically important crop species. Understanding these local origins has direct relevance for Nigerian plant breeders and agricultural policy.
                </p>
                <h3 className="font-semibold mt-4 mb-2">White Guinea Yam (<em>Dioscorea rotundata</em>)</h3>
                <p>
                  West Africa — particularly the "Yam Belt" stretching from Côte d'Ivoire through Ghana, Togo, Benin, and Nigeria — is the primary centre of diversity for yam. Nigeria is the world's largest producer, accounting for over 65% of global output. Wild <em>Dioscorea</em> species are still found across the forest–savanna transition zone, and significant diversity exists in traditional varieties (landraces) maintained by smallholder farmers.
                </p>
                <h3 className="font-semibold mt-4 mb-2">Cowpea (<em>Vigna unguiculata</em>)</h3>
                <p>
                  West and Central Africa is the primary centre of diversity for cowpea. Nigeria is the world's largest producer and consumer of cowpea. The extensive diversity in cowpea — in seed colour, photoperiod response, drought tolerance, and pest resistance — reflects its long history of cultivation in the region. IITA (based in Ibadan, Nigeria) maintains the world's largest cowpea germplasm collection (~15,000 accessions).
                </p>
                <h3 className="font-semibold mt-4 mb-2">Sorghum (<em>Sorghum bicolor</em>)</h3>
                <p>
                  Sorghum was domesticated in the Sahelian zone of Africa approximately 5,000–8,000 years ago, with evidence pointing to a corridor from Ethiopia to Nigeria. Nigeria is a major sorghum-producing country, and the crop plays a critical role in food security across the dry-savanna and Sudan-Sahel zones of the country.
                </p>
                <h3 className="font-semibold mt-4 mb-2">Notable Introduced Crops in Nigeria</h3>
                <div className="not-prose overflow-x-auto rounded-xl border border-border mt-3">
                  <table className="w-full text-sm">
                    <thead className="bg-muted/60">
                      <tr>
                        <th className="px-4 py-2.5 text-left font-semibold">Crop</th>
                        <th className="px-4 py-2.5 text-left font-semibold">Origin</th>
                        <th className="px-4 py-2.5 text-left font-semibold">Introduced via</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {[
                        ["Maize", "Mesoamerica", "Portuguese traders, ~16th century"],
                        ["Cassava", "South America (Brazil)", "Portuguese, ~17th century"],
                        ["Groundnut", "South America (Bolivia/Argentina)", "Portuguese/Spanish traders"],
                        ["Tomato", "South America (Andes)", "European colonists"],
                        ["Cocoa", "Mesoamerica", "British colonists, 19th century"],
                      ].map(([crop, origin, intro]) => (
                        <tr key={crop} className="hover:bg-muted/30">
                          <td className="px-4 py-2.5 font-medium">{crop}</td>
                          <td className="px-4 py-2.5 text-muted-foreground">{origin}</td>
                          <td className="px-4 py-2.5">{intro}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>

            {/* 9. Revision Quiz */}
            <section
              id="revision-quiz"
              ref={setRef("revision-quiz")}
              className="scroll-mt-24"
            >
              <SectionHeading icon={HelpCircle} color="text-indigo-600" bg="bg-indigo-50" border="border-indigo-200">
                Revision Quiz
              </SectionHeading>
              <p className="text-sm text-muted-foreground mt-2 mb-5">
                Test your understanding of this module. Select one answer per question, then submit.
              </p>
              <RevisionQuiz />
            </section>

          </main>
        </div>
      </div>
    </Layout>
  );
}

// ── Shared sub-components ─────────────────────────────────────────────────────

function SectionHeading({
  icon: Icon,
  color,
  bg,
  border,
  children,
}: {
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  bg: string;
  border: string;
  children: React.ReactNode;
}) {
  return (
    <div className={cn("flex items-center gap-3 pb-3 border-b", border)}>
      <div className={cn("w-9 h-9 rounded-lg flex items-center justify-center shrink-0", bg)}>
        <Icon className={cn("w-5 h-5", color)} />
      </div>
      <h2 className={cn("font-serif text-xl font-bold", color)}>{children}</h2>
    </div>
  );
}

function InfoBox({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="not-prose my-4 rounded-xl border border-blue-200 bg-blue-50/60 px-5 py-4">
      <p className="text-sm font-semibold text-blue-800 mb-2">{title}</p>
      <div className="text-sm text-blue-900/80">{children}</div>
    </div>
  );
}
