import { Link } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { ArrowRight, HelpCircle } from "lucide-react";

const faqs = [
  {
    category: "Prerequisites",
    questions: [
      {
        question: "Do I need a statistics background to join?",
        answer: "No prior statistics background is required. FIA–ADAP™ Foundations is designed to take you from fundamentals to competence. However, you should have basic computer literacy and familiarity with Excel. If you've ever run a statistical test without fully understanding it, this program is for you.",
      },
      {
        question: "What software do I need?",
        answer: "You'll need access to R Studio (free), and ideally SAS Studio, Minitab, and Excel. We provide guidance on obtaining student/academic licenses where applicable. The emphasis is on understanding methodology—once you grasp concepts, you can apply them in any software.",
      },
      {
        question: "Is this only for plant science researchers?",
        answer: "While many examples come from crop science and plant breeding, the statistical methods apply across agricultural disciplines. Past participants have included soil scientists, animal scientists, agricultural economists, and extension professionals.",
      },
    ],
  },
  {
    category: "Program Logistics",
    questions: [
      {
        question: "What if I miss a live session?",
        answer: "All sessions are recorded and made available within 24 hours. While we encourage attending live for the interactive Q&A, you can catch up with recordings. However, consistent live attendance leads to better learning outcomes.",
      },
      {
        question: "What are the session times?",
        answer: "Sessions are scheduled to be convenient for West African Time (WAT) zones. Specific times are announced before each cohort begins. We aim for evening or weekend sessions to accommodate working professionals and students.",
      },
      {
        question: "How much time should I dedicate weekly?",
        answer: "Plan for approximately 6-8 hours per week: 4 hours for live sessions (2 x 2-hour sessions) and 2-4 hours for practical exercises and revision. The investment of time pays off in competence.",
      },
    ],
  },
  {
    category: "Certification",
    questions: [
      {
        question: "Is the certificate recognized?",
        answer: "The FIA Certificate of Competence demonstrates verified ability in agricultural data analysis. While it's not a government-issued credential, it's increasingly recognized by supervisors and institutions as evidence of genuine statistical competence. The assessment-based nature distinguishes it from typical participation certificates.",
      },
      {
        question: "What if I don't pass the assessment?",
        answer: "We provide feedback on assessments to help you improve. If you don't pass initially, you have opportunities to resubmit. Our goal is your competence, not creating barriers. The vast majority of engaged participants successfully earn their certificates.",
      },
    ],
  },
  {
    category: "Practical Application",
    questions: [
      {
        question: "Can I use my own dataset during the program?",
        answer: "Absolutely! While we provide agricultural datasets for exercises, you're encouraged to apply techniques to your own research data. Many participants find the program most valuable when working on their actual thesis or project data.",
      },
      {
        question: "Will this help with my thesis defense?",
        answer: "Yes—this is a core goal of the program. Week 6 specifically covers writing results sections and defending methodology. You'll learn to articulate why you chose specific designs and analyses, and how to respond to examiner questions confidently.",
      },
      {
        question: "What if my research uses a design not covered?",
        answer: "The program covers the most common designs in agricultural research. If you have a specialized design (e.g., augmented designs, alpha-lattice), the foundational understanding you gain will help you approach it correctly. We also address questions about specific designs during Q&A sessions.",
      },
    ],
  },
  {
    category: "Payment & Registration",
    questions: [
      {
        question: "What payment methods are accepted?",
        answer: "We use Paystack for secure payments, accepting Nigerian bank cards, bank transfers, and USSD. International participants can pay via card. Receipts are issued automatically upon successful payment.",
      },
      {
        question: "Is there a refund policy?",
        answer: "Full refunds are available if you withdraw before the program starts. Once sessions begin, we offer a partial refund (minus administrative costs) within the first week. After the first week, no refunds are available as significant program value has been delivered.",
      },
      {
        question: "Are there group discounts?",
        answer: "For groups of 5 or more from the same institution, contact us directly at info@fieldtoinsightacademy.com.ng to discuss group arrangements. We're committed to making quality training accessible.",
      },
    ],
  },
];

export default function FAQ() {
  return (
    <Layout>
      {/* Hero */}
      <section className="bg-primary text-primary-foreground py-20 md:py-28">
        <div className="container-wide">
          <div className="max-w-3xl">
            <h1 className="font-serif text-4xl md:text-5xl font-bold mb-6">
              Frequently Asked Questions
            </h1>
            <p className="text-xl text-primary-foreground/85 leading-relaxed">
              Everything you need to know about FIA–ADAP™ Foundations. 
              Can't find what you're looking for? Contact us directly.
            </p>
          </div>
        </div>
      </section>

      {/* FAQ Content */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto">
            {faqs.map((category) => (
              <div key={category.category} className="mb-12">
                <h2 className="font-serif text-2xl font-bold text-foreground mb-6 flex items-center gap-3">
                  <HelpCircle className="w-6 h-6 text-primary" />
                  {category.category}
                </h2>
                <Accordion type="single" collapsible className="space-y-3">
                  {category.questions.map((faq, index) => (
                    <AccordionItem
                      key={index}
                      value={`${category.category}-${index}`}
                      className="bg-card border border-border rounded-xl overflow-hidden px-6"
                    >
                      <AccordionTrigger className="py-4 hover:no-underline text-left">
                        <span className="font-semibold text-foreground pr-4">
                          {faq.question}
                        </span>
                      </AccordionTrigger>
                      <AccordionContent className="pb-4 text-muted-foreground">
                        {faq.answer}
                      </AccordionContent>
                    </AccordionItem>
                  ))}
                </Accordion>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Still Have Questions */}
      <section className="section-padding bg-secondary">
        <div className="container-wide">
          <div className="max-w-2xl mx-auto text-center">
            <h2 className="font-serif text-3xl font-bold text-foreground mb-4">
              Still Have Questions?
            </h2>
            <p className="text-lg text-muted-foreground mb-8">
              We're here to help. Reach out directly and we'll respond promptly.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button variant="default" size="lg" asChild>
                <a href="mailto:info@fieldtoinsightacademy.com.ng">
                  Email Us
                </a>
              </Button>
              <Button variant="outline" size="lg" asChild>
                <Link to="/apply">Apply Now</Link>
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="section-padding">
        <div className="container-wide text-center">
          <h2 className="font-serif text-3xl font-bold text-foreground mb-4">
            Ready to Transform Your Research?
          </h2>
          <p className="text-lg text-muted-foreground mb-8 max-w-2xl mx-auto">
            Join the next cohort of FIA–ADAP™ Foundations and gain the confidence 
            to design, analyse, and defend your agricultural research data.
          </p>
          <Button variant="gold" size="lg" asChild>
            <Link to="/apply">
              Apply Now
              <ArrowRight className="w-4 h-4" />
            </Link>
          </Button>
        </div>
      </section>
    </Layout>
  );
}
