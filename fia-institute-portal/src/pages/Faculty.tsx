import { Link } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import {
  ArrowRight,
  GraduationCap,
  BookOpen,
  Users,
  Award,
  Microscope,
  Globe,
  Code,
  Sprout,
} from "lucide-react";

const faculty = [
  {
    name: "Dr. Lawrence Stephen Fayeun",
    title: "Founder & Lead Faculty",
    affiliation: "Department of Crop, Soil and Pest Management, Federal University of Technology, Akure (FUTA), Nigeria",
    specialization: "Plant Breeding & Genetics",
    icon: GraduationCap,
    bio: "Dr. Lawrence Stephen Fayeun is a university lecturer and researcher specializing in Plant Breeding and Genetics. With years of experience supervising postgraduate students and conducting agricultural research, he has witnessed firsthand the struggles students face with data analysis. This inspired the creation of Field-to-Insight Academy—a training platform specifically designed to bridge the gap between theoretical statistics and practical agricultural research methodology.",
    expertise: [
      "Experimental Design & Statistical Analysis",
      "Genotype × Environment Interaction Analysis",
      "GGE Biplot & AMMI Analysis",
      "Plant Breeding & Genetics",
      "Multivariate Data Analysis",
      "Research Methodology Training",
    ],
    fiaRole: "Architect of the FIA Field-to-Insight framework and experienced postgraduate supervisor. Leads the curriculum design and delivers core sessions on experimental design, ANOVA, G×E interaction, and research defense.",
    collaborations: [
      "International Institute of Tropical Agriculture (IITA)",
      "Cocoa Research Institute of Nigeria (CRIN)",
      "Multiple Nigerian universities and research institutions",
    ],
  },
  {
    name: "Dr. Paul Adunola, Ph.D.",
    title: "Faculty Member",
    affiliation: "Plant Breeding, Quantitative Genetics & Data-Driven Crop Improvement",
    specialization: "Quantitative Genetics & Genomic Selection",
    icon: Microscope,
    bio: "Dr. Paul Adunola is a plant breeding scientist and quantitative geneticist specializing in genomic and phenomic selection. His research focuses on applying advanced statistical methods and machine learning to accelerate crop improvement programs.",
    expertise: [
      "Ph.D. in Horticultural Science – University of Florida",
      "MSc in Plant Breeding – Erasmus emPLANT Program (France & Finland)",
      "FFAR Fellow (2023–2026), sponsored by Syngenta",
      "Genomic and Phenomic Selection",
      "Quantitative Genetics",
      "Machine Learning in Crop Improvement",
    ],
    fiaRole: "At FIA, Dr. Adunola teaches how advanced statistics, machine learning, and breeding technologies translate field data into sound breeding decisions. He brings international perspective and cutting-edge quantitative methods to the program.",
    collaborations: [],
  },
  {
    name: "Mr. Haruna Isola Aremu",
    title: "Faculty Member",
    affiliation: "Data Analysis, Computational Tools & Seed Science Applications",
    specialization: "Computational Data Analysis",
    icon: Code,
    bio: "Mr. Haruna Isola Aremu is a Ph.D. candidate in Seed Science at Obafemi Awolowo University. He is an expert in Python, R, SQL, and Excel, with extensive experience applying machine learning and statistical methods to seed science and crop improvement research.",
    expertise: [
      "Ph.D. Candidate, Seed Science – Obafemi Awolowo University",
      "Python Programming & Data Science",
      "R Statistical Computing",
      "SQL Database Management",
      "Machine Learning Applications",
      "Excel Data Analysis",
    ],
    fiaRole: "At FIA, Mr. Aremu focuses on data preparation, exploratory analysis, computational workflows, and disciplined data handling. He ensures participants develop strong foundations in data management and computational thinking before advanced analysis.",
    collaborations: [],
  },
];

const teachingPhilosophy = [
  {
    title: "Understanding 'Why', Not Just 'How'",
    description: "Statistical techniques should be understood conceptually, not just procedurally. Knowing why a method works enables you to apply it correctly in new situations.",
  },
  {
    title: "Agriculture-Specific Context",
    description: "Generic statistics courses miss the nuances of agricultural research. Every example and dataset comes from real agricultural experiments.",
  },
  {
    title: "Competence Through Practice",
    description: "True learning happens when you apply concepts to real data. The program emphasizes hands-on work with immediate feedback.",
  },
  {
    title: "Defense-Ready Researchers",
    description: "The goal isn't just completing analyses—it's being able to confidently defend every methodological choice you make.",
  },
];

export default function Faculty() {
  return (
    <Layout>
      {/* Hero */}
      <section className="bg-primary text-primary-foreground py-20 md:py-28">
        <div className="container-wide">
          <div className="max-w-3xl">
            <h1 className="font-serif text-4xl md:text-5xl font-bold mb-6">
              Faculty
            </h1>
            <p className="text-xl text-primary-foreground/85 leading-relaxed">
              Learn from experienced researchers and educators with deep expertise 
              in agricultural data analysis and plant science.
            </p>
          </div>
        </div>
      </section>

      {/* Faculty Members */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="space-y-16">
            {faculty.map((member, index) => (
              <div 
                key={member.name} 
                className={`grid lg:grid-cols-3 gap-12 ${index > 0 ? 'pt-16 border-t border-border' : ''}`}
              >
                {/* Profile Card */}
                <div className="lg:col-span-1">
                  <div className="card-elevated p-8 text-center sticky top-24">
                    <div className="w-24 h-24 rounded-full bg-primary/10 flex items-center justify-center mx-auto mb-6">
                      <member.icon className="w-12 h-12 text-primary" />
                    </div>
                    <h2 className="font-serif text-xl font-bold text-foreground mb-2">
                      {member.name}
                    </h2>
                    <p className="text-primary font-medium mb-2">
                      {member.title}
                    </p>
                    <p className="text-muted-foreground text-sm mb-4">
                      {member.specialization}
                    </p>
                    <div className="text-left space-y-2 pt-4 border-t border-border">
                      <div className="flex items-center gap-3 text-sm">
                        <BookOpen className="w-4 h-4 text-primary" />
                        <span className="text-muted-foreground">{member.affiliation}</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Bio Content */}
                <div className="lg:col-span-2 space-y-8">
                  {/* About */}
                  <div>
                    <h3 className="font-serif text-2xl font-bold text-foreground mb-4">
                      About
                    </h3>
                    <p className="text-muted-foreground leading-relaxed">
                      {member.bio}
                    </p>
                  </div>

                  {/* FIA Role */}
                  <div className="bg-primary/5 rounded-xl p-6 border border-primary/20">
                    <h4 className="font-semibold text-foreground mb-3 flex items-center gap-2">
                      <Sprout className="w-5 h-5 text-primary" />
                      Role at FIA
                    </h4>
                    <p className="text-muted-foreground">
                      {member.fiaRole}
                    </p>
                  </div>

                  {/* Expertise */}
                  <div>
                    <h3 className="font-serif text-xl font-bold text-foreground mb-4">
                      Expertise & Credentials
                    </h3>
                    <div className="grid md:grid-cols-2 gap-3">
                      {member.expertise.map((item) => (
                        <div
                          key={item}
                          className="flex items-center gap-3 p-3 bg-muted rounded-lg"
                        >
                          <Award className="w-4 h-4 text-primary flex-shrink-0" />
                          <span className="text-foreground text-sm">{item}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Collaborations (if any) */}
                  {member.collaborations.length > 0 && (
                    <div>
                      <h3 className="font-serif text-xl font-bold text-foreground mb-4">
                        Research Collaborations
                      </h3>
                      <div className="space-y-3">
                        {member.collaborations.map((item) => (
                          <div
                            key={item}
                            className="flex items-center gap-3 p-3 bg-secondary rounded-lg"
                          >
                            <Globe className="w-4 h-4 text-primary flex-shrink-0" />
                            <span className="text-foreground text-sm">{item}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Teaching Philosophy */}
      <section className="section-padding bg-secondary">
        <div className="container-wide">
          <div className="text-center mb-12">
            <h2 className="font-serif text-3xl font-bold text-foreground mb-4">
              Teaching Philosophy
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              The guiding principles behind FIA's approach to agricultural 
              data analysis training.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 gap-6 max-w-4xl mx-auto">
            {teachingPhilosophy.map((item) => (
              <div key={item.title} className="bg-card rounded-xl p-6 border border-border">
                <h3 className="font-serif text-xl font-semibold text-foreground mb-3">
                  {item.title}
                </h3>
                <p className="text-muted-foreground">
                  {item.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Quote */}
      <section className="py-16 bg-primary text-primary-foreground">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto text-center">
            <blockquote className="font-serif text-2xl md:text-3xl italic leading-relaxed mb-6">
              "Our goal is not just to teach you how to run an ANOVA or create a 
              GGE biplot. It's to ensure you understand why you're doing it, when 
              it's appropriate, and how to defend your choice to anyone who asks."
            </blockquote>
            <p className="text-primary-foreground/80 font-medium">
              — Dr. Lawrence Stephen Fayeun, Founder
            </p>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="section-padding">
        <div className="container-wide text-center">
          <h2 className="font-serif text-3xl font-bold text-foreground mb-4">
            Learn from Our Faculty
          </h2>
          <p className="text-lg text-muted-foreground mb-8 max-w-2xl mx-auto">
            Join FIA–ADAP™ Foundations and benefit from expert instruction 
            tailored to agricultural researchers.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button variant="gold" size="lg" asChild>
              <Link to="/apply">
                Apply Now
                <ArrowRight className="w-4 h-4" />
              </Link>
            </Button>
            <Button variant="outline" size="lg" asChild>
              <Link to="/program">View Program</Link>
            </Button>
          </div>
        </div>
      </section>
    </Layout>
  );
}
