// Backend base URL for the Manuscript Reviewer API
export const MANUSCRIPT_REVIEWER_BASE =
  import.meta.env.VITE_API_BASE ||
  "https://fia-manuscript-reviewer.onrender.com";

export interface LanguageIssue {
  paragraph_line: string;
  excerpt: string;
  issue_type: string;
  problem: string;
  suggested_correction: string;
  severity: "Minor" | "Moderate" | "Major" | "Critical";
}

export interface SimilarityMatch {
  source: string;
  matched_text: string;
  similarity_percent: number;
}

export interface SimilarityReview {
  overall_similarity_percent: number;
  flagged_sections: SimilarityMatch[];
  verdict: string;
}

export interface AiWritingRiskReview {
  overall_ai_probability: number;
  flagged_sections: { section: string; ai_probability: number; explanation: string }[];
  verdict: string;
}

export interface ExportsAvailable {
  full_review_report?: string;
  comments_to_author?: string;
  confidential_editor_note?: string;
  citation_review?: string;
  similarity_review?: string;
  ai_risk_review?: string;
  annotated_manuscript?: string;
}

export interface ReviewResult {
  manuscript_title: string;
  language_review: {
    manuscript_title: string;
    issues: LanguageIssue[];
  };
  technical_review: {
    major_scientific_concerns: string[];
    methodological_concerns: string[];
    statistical_concerns: string[];
    interpretation_problems: string[];
  };
  final_summary: {
    overall_assessment: string;
    top_10_priority_revisions: string[];
    final_recommendation: string;
  };
  citation_audit: {
    missing_from_references: string[];
    uncited_references: string[];
    inconsistencies: string[];
    formatting_issues: string[];
  };
  similarity_review?: SimilarityReview;
  ai_writing_risk_review?: AiWritingRiskReview;
  exports_available?: ExportsAvailable;
}

export const MOCK_REVIEW_RESULT: ReviewResult = {
  manuscript_title: "Effects of Organic Fertilizer on Maize Yield Under Rain-fed Conditions",
  language_review: {
    manuscript_title: "Effects of Organic Fertilizer on Maize Yield Under Rain-fed Conditions",
    issues: [
      {
        paragraph_line: "Paragraph 3, Line 2",
        excerpt: "The result shows significant differences between treatments...",
        issue_type: "Grammar",
        problem: "Subject-verb agreement: 'result' is singular but should be plural to match the context.",
        suggested_correction: "The results show significant differences between treatments...",
        severity: "Minor",
      },
      {
        paragraph_line: "Paragraph 7, Line 4",
        excerpt: "This is in agreement with the findings of Adeyemi et al (2019) who reported that...",
        issue_type: "Citation Format",
        problem: "Missing period after 'al' in 'et al.'",
        suggested_correction: "This is in agreement with the findings of Adeyemi et al. (2019), who reported that...",
        severity: "Minor",
      },
      {
        paragraph_line: "Paragraph 12, Line 1",
        excerpt: "The data was analyzed using analysis of variance.",
        issue_type: "Grammar",
        problem: "'Data' is typically treated as plural in academic writing.",
        suggested_correction: "The data were analyzed using analysis of variance.",
        severity: "Minor",
      },
      {
        paragraph_line: "Paragraph 15, Line 3",
        excerpt: "However the treatment with highest yield was T3.",
        issue_type: "Punctuation",
        problem: "Missing comma after introductory adverb 'However'.",
        suggested_correction: "However, the treatment with the highest yield was T3.",
        severity: "Minor",
      },
      {
        paragraph_line: "Paragraph 20, Line 2",
        excerpt: "The implication of this study is far reaching and could impact future research directions.",
        issue_type: "Style",
        problem: "'Far reaching' should be hyphenated as a compound adjective. Sentence is also vague.",
        suggested_correction: "The implications of this study are far-reaching and may inform future research on organic amendments in tropical cropping systems.",
        severity: "Moderate",
      },
    ],
  },
  technical_review: {
    major_scientific_concerns: [
      "The hypothesis is not clearly stated in the introduction.",
      "No justification provided for the choice of organic fertilizer types used.",
    ],
    methodological_concerns: [
      "Plot size is not specified for the field experiment.",
      "Randomization procedure is not described — unclear whether CRD or RCBD was used.",
      "Soil baseline characterization data is missing.",
    ],
    statistical_concerns: [
      "ANOVA assumptions (normality, homogeneity of variance) were not tested or reported.",
      "Mean separation test used (LSD) should be justified; Tukey HSD may be more appropriate for multiple comparisons.",
    ],
    interpretation_problems: [
      "Results discussion does not adequately compare findings with published literature.",
      "Conclusions extend beyond the data presented — causal language used for correlational findings.",
    ],
  },
  final_summary: {
    overall_assessment:
      "The manuscript addresses an important agricultural topic but requires substantial revision in methodology reporting, statistical analysis, and interpretation. Language issues are mostly minor but should be corrected for publication readiness.",
    top_10_priority_revisions: [
      "State the research hypothesis clearly in the introduction.",
      "Specify the experimental design (CRD, RCBD, etc.) and randomization procedure.",
      "Report plot dimensions and spacing.",
      "Include soil baseline characterization data.",
      "Test and report ANOVA assumptions.",
      "Justify the choice of mean separation test.",
      "Correct all subject-verb agreement issues.",
      "Strengthen the discussion with relevant literature comparisons.",
      "Avoid causal language where only correlational data is presented.",
      "Fix citation formatting inconsistencies throughout.",
    ],
    final_recommendation: "Major Revision",
  },
  citation_audit: {
    missing_from_references: [
      "Adeyemi et al. (2019) — cited in text but not listed in references.",
      "FAO (2021) — cited in introduction but absent from reference list.",
    ],
    uncited_references: [
      "Ogunlade, B.T. (2018). Soil fertility management in the tropics. — listed in references but never cited.",
    ],
    inconsistencies: [
      "Akinola (2020) is cited as Akinola (2021) in two places.",
      "Ibrahim and Hassan (2017) appears as Ibrahim & Hassan (2017) inconsistently.",
    ],
    formatting_issues: [
      "Reference list uses mixed APA and Harvard styles.",
      "Several references missing DOI or URL where available.",
      "Journal names are inconsistently italicized.",
    ],
  },
};
