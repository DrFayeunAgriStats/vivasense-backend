import { Layout } from "@/components/layout/Layout";
import { Card, CardContent } from "@/components/ui/card";
import { Shield, CheckCircle, AlertTriangle } from "lucide-react";

export default function RWSAcademicIntegrity() {
  return (
    <Layout>
      <section className="bg-primary text-primary-foreground py-12 md:py-16">
        <div className="container max-w-4xl">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-12 h-12 rounded-full bg-primary-foreground/10 flex items-center justify-center">
              <Shield className="w-6 h-6" />
            </div>
            <div>
              <h1 className="font-serif text-3xl md:text-4xl font-bold">Academic Integrity Statement</h1>
              <p className="text-primary-foreground/70 text-sm mt-1">FIA Research Writing System</p>
            </div>
          </div>
        </div>
      </section>

      <section className="container max-w-3xl py-12 space-y-8">
        <Card>
          <CardContent className="pt-6 space-y-6">
            <div>
              <h2 className="font-serif text-xl font-bold text-foreground mb-3">Our Commitment</h2>
              <p className="text-muted-foreground leading-relaxed">
                The Field-to-Insight Academy (FIA) Research Writing System is an educational platform 
                dedicated to teaching students the skills of independent research thinking, data analysis, 
                and academic writing. Our platform upholds the highest standards of academic integrity.
              </p>
            </div>

            <div className="p-4 rounded-lg bg-destructive/5 border border-destructive/20">
              <div className="flex gap-3">
                <AlertTriangle className="w-5 h-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground text-sm mb-2">What FIA Does NOT Do</p>
                  <ul className="space-y-1.5 text-sm text-muted-foreground">
                    <li>• FIA does NOT write theses, dissertations, or project reports for students</li>
                    <li>• FIA does NOT generate ready-to-submit academic text</li>
                    <li>• FIA does NOT provide shortcuts to bypass the learning process</li>
                    <li>• FIA does NOT replace the role of the academic supervisor</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="p-4 rounded-lg bg-primary/5 border border-primary/20">
              <div className="flex gap-3">
                <CheckCircle className="w-5 h-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground text-sm mb-2">What FIA Does</p>
                  <ul className="space-y-1.5 text-sm text-muted-foreground">
                    <li>• Teaches students to analyse their own data and interpret results</li>
                    <li>• Provides structured writing prompts and conceptual guidance</li>
                    <li>• Uses AI tools for guidance and feedback only</li>
                    <li>• Generates supervisor discussion notes — not thesis content</li>
                    <li>• Reinforces critical thinking and evidence-based reasoning</li>
                    <li>• Labels all AI-generated guidance clearly as "NOT THESIS TEXT"</li>
                  </ul>
                </div>
              </div>
            </div>

            <div>
              <h2 className="font-serif text-xl font-bold text-foreground mb-3">Student Responsibility</h2>
              <p className="text-muted-foreground leading-relaxed">
                All students using the FIA Research Writing System confirm that they will write their 
                thesis, dissertation, or project report in their own words. AI-generated content from 
                this platform is provided as learning guidance and must not be submitted as original work. 
                Students are expected to inform their supervisors about using AI-assisted learning tools 
                and to comply with their institution's academic integrity policies.
              </p>
            </div>

            <div>
              <h2 className="font-serif text-xl font-bold text-foreground mb-3">Transparency</h2>
              <p className="text-muted-foreground leading-relaxed">
                Every PDF exported from the platform includes watermarks clearly marking content as 
                "AI GUIDANCE NOTES — NOT THESIS TEXT" or "NOT PROPOSAL TEXT". Session metadata, timestamps, 
                and student identifiers are embedded in all exports for accountability.
              </p>
            </div>
          </CardContent>
        </Card>
      </section>
    </Layout>
  );
}
