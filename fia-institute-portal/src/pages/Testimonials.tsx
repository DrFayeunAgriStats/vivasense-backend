import { Link } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import { ArrowRight, Quote, MapPin } from "lucide-react";

const testimonials = [
  {
    quote: "Before FIA, I was just clicking buttons in SPSS without understanding why. Now I can confidently choose the right analysis for my data and explain my methodology during supervision meetings. My thesis chapter on methodology went from rejection to approval.",
    author: "Adaeze Okonkwo",
    role: "PhD Candidate, Plant Breeding",
    institution: "University of Nigeria, Nsukka",
    location: "Nigeria",
    highlight: "Thesis approval after revision",
  },
  {
    quote: "My supervisor was genuinely impressed when I could explain the assumptions behind ANOVA and why RCBD was the appropriate design for my field trial. The confidence I gained from understanding 'why' rather than just 'how' changed everything.",
    author: "Emmanuel Kyeremeh",
    role: "MSc Student, Agronomy",
    institution: "Université Abdou Moumouni",
    location: "Niger",
    highlight: "Improved supervisor confidence",
  },
  {
    quote: "The G×E analysis sessions transformed how I approach my multi-environment trial data. I finally understand what AMMI and GGE biplot results are telling me about genotype performance. My manuscript is now publication-ready.",
    author: "Dr. Aminata Kamara",
    role: "Research Scientist",
    institution: "Sierra Leone Agricultural Research Institute",
    location: "Sierra Leone",
    highlight: "Publication-ready manuscript",
  },
  {
    quote: "As an international researcher, I was skeptical about an online program from Nigeria. But the quality of instruction and depth of agricultural-specific content exceeded my expectations. Dr. Fayeun's teaching approach is exceptional.",
    author: "Dr. James Mitchell",
    role: "Postdoctoral Researcher",
    institution: "University of Reading",
    location: "United Kingdom",
    highlight: "International quality standard",
  },
];

const outcomes = [
  { number: "5", label: "Countries represented" },
  { number: "100%", label: "Would recommend FIA" },
];

export default function Testimonials() {
  return (
    <Layout>
      {/* Hero */}
      <section className="bg-primary text-primary-foreground py-20 md:py-28">
        <div className="container-wide">
          <div className="max-w-3xl">
            <h1 className="font-serif text-4xl md:text-5xl font-bold mb-6">
              What Participants Say
            </h1>
            <p className="text-xl text-primary-foreground/85 leading-relaxed">
              Hear from researchers who have transformed their approach to 
              agricultural data analysis through FIA–ADAP™ Foundations.
            </p>
          </div>
        </div>
      </section>

      {/* Stats */}
      <section className="py-12 bg-accent/10">
        <div className="container-wide">
          <div className="flex justify-center gap-16">
            {outcomes.map((stat) => (
              <div key={stat.label} className="text-center">
                <div className="text-3xl md:text-4xl font-bold text-primary mb-1">
                  {stat.number}
                </div>
                <div className="text-muted-foreground text-sm">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials Grid */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="grid md:grid-cols-2 gap-6 max-w-5xl mx-auto">
            {testimonials.map((testimonial, index) => (
              <div
                key={index}
                className="card-elevated p-8 flex flex-col"
              >
                <div className="flex items-start gap-4 mb-4">
                  <Quote className="w-8 h-8 text-accent flex-shrink-0" />
                  <span className="px-3 py-1 bg-accent/10 rounded-full text-accent text-xs font-semibold">
                    {testimonial.highlight}
                  </span>
                </div>
                
                <blockquote className="text-foreground leading-relaxed mb-6 flex-grow">
                  "{testimonial.quote}"
                </blockquote>
                
                <div className="border-t border-border pt-4">
                  <div className="font-semibold text-foreground">
                    {testimonial.author}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {testimonial.role}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {testimonial.institution}
                  </div>
                  <div className="flex items-center gap-1 text-sm text-primary font-medium mt-2">
                    <MapPin className="w-3 h-3" />
                    {testimonial.location}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* International Note */}
      <section className="section-padding bg-muted">
        <div className="container-wide">
          <div className="max-w-2xl mx-auto text-center">
            <p className="text-lg text-muted-foreground">
              Participants in FIA pilot cohorts have included researchers from 
              <strong className="text-foreground"> Nigeria, Niger, Sierra Leone, the United Kingdom, and the United States</strong>.
            </p>
          </div>
        </div>
      </section>

      {/* Share Your Story */}
      <section className="section-padding bg-secondary">
        <div className="container-wide">
          <div className="max-w-2xl mx-auto text-center">
            <h2 className="font-serif text-3xl font-bold text-foreground mb-4">
              Your Story Could Be Here
            </h2>
            <p className="text-lg text-muted-foreground mb-8">
              Join the growing community of agricultural researchers who have 
              transformed their data analysis skills with FIA–ADAP™ Foundations.
            </p>
            <Button variant="gold" size="lg" asChild>
              <Link to="/apply">
                Apply Now
                <ArrowRight className="w-4 h-4" />
              </Link>
            </Button>
          </div>
        </div>
      </section>
    </Layout>
  );
}
