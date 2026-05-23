import type { ReviewResult, LanguageIssue } from "@/config/manuscriptReviewer";

/**
 * Transforms the live backend response into the ReviewResult shape
 * expected by the frontend dashboard.
 *
 * Backend field names differ from the frontend interface — this function
 * bridges the gap so the UI code doesn't need to change.
 */
export function transformReviewResponse(raw: any): ReviewResult {
  return {
    manuscript_title: raw.manuscript_title ?? "Untitled Manuscript",
    language_review: transformLanguageReview(raw.language_review),
    technical_review: transformTechnicalReview(raw.technical_review),
    final_summary: transformFinalSummary(raw.final_summary),
    citation_audit: transformCitationAudit(raw.citation_audit),
    similarity_review: raw.similarity_review ?? undefined,
    ai_writing_risk_review: raw.ai_writing_risk_review ?? undefined,
    exports_available: raw.exports_available ?? undefined,
  };
}

function transformLanguageReview(lr: any): ReviewResult["language_review"] {
  // Backend returns { results: [{ paragraph_index, paragraph_text, issues, suggestions }] }
  // Frontend expects { manuscript_title, issues: LanguageIssue[] }
  if (lr?.issues && Array.isArray(lr.issues) && lr.issues[0]?.paragraph_line) {
    // Already in expected format
    return lr;
  }

  const issues: LanguageIssue[] = [];

  if (Array.isArray(lr?.results)) {
    for (const para of lr.results) {
      const paraIssues: string[] = Array.isArray(para.issues) ? para.issues : [];
      const paraSuggestions: string[] = Array.isArray(para.suggestions) ? para.suggestions : [];

      for (let j = 0; j < paraIssues.length; j++) {
        issues.push({
          paragraph_line: `Paragraph ${(para.paragraph_index ?? 0) + 1}`,
          excerpt: para.paragraph_text?.substring(0, 120) ?? "",
          issue_type: "Language",
          problem: paraIssues[j],
          suggested_correction: paraSuggestions[j] ?? "No suggestion provided.",
          severity: "Minor",
        });
      }
    }
  }

  return {
    manuscript_title: lr?.manuscript_title ?? "",
    issues,
  };
}

function transformTechnicalReview(tr: any): ReviewResult["technical_review"] {
  // Backend returns: methodology_notes, statistical_notes, clarity_notes, recommendations
  // Frontend expects: major_scientific_concerns, methodological_concerns, statistical_concerns, interpretation_problems
  if (tr?.major_scientific_concerns) {
    return tr; // already in expected format
  }

  return {
    major_scientific_concerns: safeArray(tr?.clarity_notes),
    methodological_concerns: safeArray(tr?.methodology_notes),
    statistical_concerns: safeArray(tr?.statistical_notes),
    interpretation_problems: safeArray(tr?.recommendations),
  };
}

function transformFinalSummary(fs: any): ReviewResult["final_summary"] {
  // Backend returns: strengths, weaknesses, overall_assessment, recommendation, key_improvements
  // Frontend expects: overall_assessment, top_10_priority_revisions, final_recommendation
  if (fs?.final_recommendation && fs?.top_10_priority_revisions) {
    return fs; // already in expected format
  }

  return {
    overall_assessment: fs?.overall_assessment ?? "",
    top_10_priority_revisions: safeArray(fs?.key_improvements).slice(0, 10),
    final_recommendation: capitalizeRecommendation(fs?.recommendation ?? "Major Revision"),
  };
}

function transformCitationAudit(ca: any): ReviewResult["citation_audit"] {
  // Backend returns: { issues: [...], total_citations, health_score, ... }
  // Frontend expects: { missing_from_references, uncited_references, inconsistencies, formatting_issues }
  if (ca?.missing_from_references) {
    return ca; // already in expected format
  }

  // Categorize issues from the flat array
  const missing: string[] = [];
  const uncited: string[] = [];
  const inconsistencies: string[] = [];
  const formatting: string[] = [];

  if (Array.isArray(ca?.issues)) {
    for (const issue of ca.issues) {
      const text = typeof issue === "string" ? issue : issue?.description ?? String(issue);
      const lower = text.toLowerCase();
      if (lower.includes("missing") || lower.includes("not found in ref")) {
        missing.push(text);
      } else if (lower.includes("uncited") || lower.includes("not cited")) {
        uncited.push(text);
      } else if (lower.includes("inconsisten") || lower.includes("mismatch")) {
        inconsistencies.push(text);
      } else {
        formatting.push(text);
      }
    }
  }

  return {
    missing_from_references: missing,
    uncited_references: uncited,
    inconsistencies,
    formatting_issues: formatting,
  };
}

function safeArray(val: any): string[] {
  return Array.isArray(val) ? val : [];
}

function capitalizeRecommendation(rec: string): string {
  const map: Record<string, string> = {
    reject: "Reject",
    "major revision": "Major Revision",
    "minor revision": "Minor Revision",
    accept: "Accept",
  };
  return map[rec.toLowerCase()] ?? rec;
}
