import { Layout } from "@/components/layout/Layout";
import { Link } from "react-router-dom";
import { ArrowLeft, Linkedin, MessageCircle, Share2 } from "lucide-react";
import { Button } from "@/components/ui/button";

const ARTICLE_URL = "https://field-to-insight-forge.lovable.app/technical-insights/scaling-agricultural-expertise";
const ARTICLE_TITLE = "Scaling Agricultural Expertise: How I Guided 200 Students through Plant Genetics using a Custom AI Tutor";

function ShareButtons() {
  const encoded = encodeURIComponent(ARTICLE_TITLE);
  const encodedUrl = encodeURIComponent(ARTICLE_URL);

  return (
    <div className="flex items-center gap-3 flex-wrap">
      <span className="text-sm font-semibold text-muted-foreground">Share:</span>
      <Button variant="outline" size="sm" asChild>
        <a
          href={`https://www.linkedin.com/sharing/share-offsite/?url=${encodedUrl}`}
          target="_blank"
          rel="noopener noreferrer"
        >
          <Linkedin className="w-4 h-4 mr-1" /> LinkedIn
        </a>
      </Button>
      <Button variant="outline" size="sm" asChild>
        <a
          href={`https://twitter.com/intent/tweet?text=${encoded}&url=${encodedUrl}`}
          target="_blank"
          rel="noopener noreferrer"
        >
          <Share2 className="w-4 h-4 mr-1" /> Twitter / X
        </a>
      </Button>
      <Button variant="outline" size="sm" asChild>
        <a
          href={`https://wa.me/?text=${encoded}%20${encodedUrl}`}
          target="_blank"
          rel="noopener noreferrer"
        >
          <MessageCircle className="w-4 h-4 mr-1" /> WhatsApp
        </a>
      </Button>
    </div>
  );
}

export default function ScalingAgriculturalExpertise() {
  return (
    <Layout>
      {/* Hero */}
      <section className="bg-primary text-primary-foreground py-16 md:py-24">
        <div className="container-wide max-w-4xl">
          <Link
            to="/technical-insights"
            className="inline-flex items-center gap-1.5 text-sm text-primary-foreground/70 hover:text-primary-foreground mb-6 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" /> Back to Technical Insights
          </Link>
          <div className="flex flex-wrap gap-2 mb-4">
            <span className="text-xs font-semibold bg-accent text-accent-foreground px-3 py-1 rounded-full">
              AI in Agriculture
            </span>
            <span className="text-xs font-semibold bg-primary-foreground/15 text-primary-foreground px-3 py-1 rounded-full">
              Ed-Tech
            </span>
          </div>
          <h1 className="font-serif text-3xl md:text-4xl lg:text-[2.6rem] font-bold leading-tight mb-5">
            {ARTICLE_TITLE}
          </h1>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-primary-foreground/75">
            <span>Dr. Lawrence Stephen Fayeun</span>
            <span className="hidden sm:inline">•</span>
            <span>April 2026</span>
          </div>
        </div>
      </section>

      {/* Body */}
      <article className="section-padding">
        <div className="container-wide max-w-3xl prose prose-lg dark:prose-invert prose-headings:font-serif prose-headings:text-foreground prose-p:text-foreground/85 prose-strong:text-foreground">

          {/* ---- Section 1 ---- */}
          <h2>The Challenge: Personalized Education at Scale</h2>
          <p>
            In agricultural sciences, subjects like <strong>Quantitative Genetics</strong> and{" "}
            <strong>Biometrical Genetics</strong> are notoriously difficult to master. They require a
            deep understanding of biological mechanisms combined with statistical rigor. For an
            instructor managing a cohort of <strong>200+ students</strong>, providing individualized,
            high-quality guidance is a significant scalability challenge.
          </p>
          <p>
            To solve this, I moved beyond traditional teaching assistants and engineered a{" "}
            <strong>Custom AI Plant Genetics Tutor</strong>. The goal was not just to answer
            questions, but to <strong>scale expert-level mentoring</strong> without sacrificing
            scientific accuracy.
          </p>

          {/* ---- Section 2 ---- */}
          <h2>System Architecture: The 5-Layer Prompt Framework</h2>
          <p>
            Generic AI models often struggle with technical "hallucinations" in niche scientific
            fields. To prevent this, I developed a <strong>Layered Prompt Architecture</strong> that
            acts as a digital guardrail for the learning process.
          </p>

          <h3>1. Role Definition</h3>
          <p>
            The system is anchored in the persona of a <strong>Senior Plant Breeder</strong>. This
            ensures the tone remains academic and the perspective remains grounded in crop
            improvement goals.
          </p>

          <h3>2. Knowledge Constraints</h3>
          <p>
            The AI is restricted to established genetic principles (Mendelian inheritance, population
            genetics, and molecular biology). If a query falls outside these boundaries, the system
            is programmed to guide the student back to the relevant curriculum.
          </p>

          <h3>3. Pedagogical Logic</h3>
          <p>
            The tutor operates in <strong>Socratic Mode</strong>. Instead of providing immediate
            answers, it asks guiding questions, encouraging students to reason through the "why"
            behind genetic variances.
          </p>

          <h3>4. Scientific Guardrails</h3>
          <p>
            Strict rules are hard-coded to prevent the invention of genetic mechanisms. This ensures
            that every explanation provided is <strong>scientifically defensible</strong>.
          </p>

          <h3>5. Output Orchestration</h3>
          <p>Responses are delivered in a structured format:</p>
          <p className="text-center font-semibold">
            Definition → Concept Explanation → Agricultural Example → Summary
          </p>

          {/* ---- Section 3 ---- */}
          <h2>Impact: 200 Students, Zero Hallucinations</h2>
          <p>
            Deployed at the <strong>Federal University of Technology, Akure (FUTA)</strong>, the
            results have been transformative:
          </p>
          <ul>
            <li>
              <strong>24/7 Accessibility:</strong> Students received instant feedback on complex
              topics like Genotype × Environment (GxE) interaction outside of lecture hours.
            </li>
            <li>
              <strong>Reduced Cognitive Load:</strong> By handling repetitive foundational questions,
              the AI allowed me to focus classroom time on advanced research methodology.
            </li>
            <li>
              <strong>High Engagement:</strong> With over 200 active users, the platform proved that
              students are eager to embrace AI when it is framed as a rigorous academic tool.
            </li>
          </ul>

          {/* ---- Section 4 ---- */}
          <h2>Leadership Insight</h2>
          <p>
            Technical documentation is a form of leadership. By architecting a system that
            prioritizes <strong>Scientific Integrity</strong> over simple automation, we create a
            model that can be replicated across all agricultural research and education domains.
          </p>

          {/* ---- Section 5 ---- */}
          <h2>The Road Ahead: VivaSense Integration</h2>
          <p>
            The success of this AI Tutor is only the first step. I am currently integrating these
            pedagogical insights into <strong>VivaSense</strong>, my flagship platform for
            agricultural data analysis. The future of agricultural development lies in the synergy
            between <strong>human expertise</strong> and <strong>engineered digital intelligence</strong>.
          </p>

          {/* ---- Divider ---- */}
          <hr className="my-10 border-border" />

          {/* ---- Author Bio ---- */}
          <div className="bg-muted rounded-xl p-6 md:p-8 not-prose border border-border">
            <h3 className="font-serif text-lg font-semibold text-foreground mb-2">
              About the Author
            </h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              <strong className="text-foreground">Dr. Lawrence Stephen Fayeun</strong> is a Plant
              Breeder, Quantitative Geneticist, Data Scientist, and the Founder of Field-to-Insight
              Academy. He specializes in bridging the gap between field-based crop research and
              AI-driven agricultural informatics.
            </p>
          </div>

          {/* ---- Share ---- */}
          <div className="mt-8 not-prose">
            <ShareButtons />
          </div>
        </div>
      </article>

      {/* Related Articles */}
      <section className="section-padding bg-muted/50">
        <div className="container-wide max-w-3xl">
          <h2 className="font-serif text-2xl font-bold text-foreground mb-6">Related Articles</h2>
          <div className="grid gap-4">
            {[
              {
                title: "Designing Scientific Guardrails for AI in Crop Improvement Research",
                category: "AI in Agriculture",
              },
              {
                title: "From Field Data to Publishable Tables: Automating Research Outputs with VivaSense",
                category: "Data Science in Agriculture",
              },
              {
                title: "Socratic AI: Teaching Statistical Thinking to Postgraduate Plant Breeders",
                category: "Ed-Tech",
              },
            ].map((item) => (
              <div
                key={item.title}
                className="bg-background border border-border rounded-xl p-5 flex flex-col gap-1"
              >
                <span className="text-xs font-semibold text-primary">{item.category}</span>
                <h3 className="font-serif text-base font-semibold text-foreground">{item.title}</h3>
                <p className="text-xs text-muted-foreground">Coming Soon</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </Layout>
  );
}
