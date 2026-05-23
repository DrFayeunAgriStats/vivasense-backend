import { Building, Users, BookOpen, Briefcase, Mail } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function VivaSenseInstitutional() {
  return (
    <>
      {/* For Supervisors & Institutions */}
      <section className="py-20 bg-primary/5">
        <div className="container-wide">
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-12">
              <h2 className="font-serif text-3xl lg:text-4xl font-bold text-foreground mb-4">
                For Supervisors & Institutions
              </h2>
            </div>
            
            <Card className="border-primary/20">
              <CardContent className="p-8 lg:p-10">
                <p className="text-lg text-muted-foreground leading-relaxed mb-6">
                  VivaSense is an academic support platform developed by Field-to-Insight Academy (FIA) 
                  to enhance research quality and statistical understanding among final-year undergraduates 
                  and postgraduate students.
                </p>
                <p className="text-lg text-muted-foreground leading-relaxed mb-6">
                  VivaSense <strong>does not replace</strong> supervision, authorship, or institutional assessment. 
                  It supports learning, promotes methodological rigor, generates transparent analyses, and helps 
                  students avoid common statistical and interpretational errors.
                </p>
                <p className="text-lg text-muted-foreground leading-relaxed">
                  Outputs are level-aware and aligned with accepted academic standards.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Institutional Licensing */}
      <section className="py-20">
        <div className="container-wide">
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-12">
              <h2 className="font-serif text-3xl lg:text-4xl font-bold text-foreground mb-4">
                Institutional Licensing
              </h2>
              <p className="text-lg text-muted-foreground">
                Scale VivaSense across your organisation
              </p>
              <p className="text-base text-muted-foreground mt-2">
                Flexible pricing for Nigerian universities, research institutes and NGOs — contact us to discuss.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
              <Card className="hover:border-primary/30 transition-colors">
                <CardHeader>
                  <CardTitle className="flex items-center gap-3">
                    <Building className="w-6 h-6 text-primary" />
                    Department-wide
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">
                    License VivaSense for all students and researchers within a single academic department.
                  </p>
                </CardContent>
              </Card>

              <Card className="hover:border-primary/30 transition-colors">
                <CardHeader>
                  <CardTitle className="flex items-center gap-3">
                    <Users className="w-6 h-6 text-primary" />
                    Faculty-wide
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">
                    Extend access across an entire faculty or college for comprehensive coverage.
                  </p>
                </CardContent>
              </Card>

              <Card className="hover:border-primary/30 transition-colors">
                <CardHeader>
                  <CardTitle className="flex items-center gap-3">
                    <BookOpen className="w-6 h-6 text-primary" />
                    Institution-wide
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">
                    University-wide licensing for maximum reach and consistent academic support.
                  </p>
                </CardContent>
              </Card>

              <Card className="hover:border-primary/30 transition-colors">
                <CardHeader>
                  <CardTitle className="flex items-center gap-3">
                    <Briefcase className="w-6 h-6 text-primary" />
                    NGO / Research Program
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">
                    Custom licensing for research organisations, NGOs, and development programmes.
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Contact */}
            <Card className="bg-primary text-primary-foreground">
              <CardContent className="p-8 text-center">
                <h3 className="font-serif text-2xl font-bold mb-4">Interested in Institutional Licensing?</h3>
                <p className="text-primary-foreground/80 mb-6">
                  Contact us to discuss licensing options for your institution or organisation.
                </p>
                <a 
                  href="mailto:info@fieldtoinsightacademy.com.ng"
                  className="inline-flex items-center gap-2 text-lg font-medium hover:underline"
                >
                  <Mail className="w-5 h-5" />
                  info@fieldtoinsightacademy.com.ng
                </a>
                <p className="mt-3 text-base font-medium text-primary-foreground">
                  📱 WhatsApp: +234 902 215 8026
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Footer Disclaimer */}
      <section className="py-8 bg-muted/50 border-t border-border">
        <div className="container-wide">
          <div className="max-w-4xl mx-auto text-center">
            <p className="text-sm text-muted-foreground leading-relaxed">
              <strong>Disclaimer:</strong> VivaSense is an academic support platform developed by Field-to-Insight Academy (FIA). 
              It supports learning and research quality and does not replace supervision, authorship, or institutional regulations.
            </p>
          </div>
        </div>
      </section>
    </>
  );
}
