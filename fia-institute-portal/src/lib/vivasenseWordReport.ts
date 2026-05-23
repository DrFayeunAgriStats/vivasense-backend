/**
 * VivaSense — Unified branded Word (.docx) report builder.
 *
 * Every VivaSense module (Regression, ANOVA, future) that exports a Word
 * report on the client must use this builder so that headings, footers,
 * spacing, colors and section names stay consistent.
 *
 * Brand rules enforced here:
 *  - Title block: "VivaSense Statistical Analysis Report" + module + context
 *  - Standard section ordering and identical wording across modules
 *  - Footer signature on every report
 *  - Neutral, domain-agnostic tone (no breeding bias)
 *  - Color logic: red = unreliable, green = valid/significant, grey = neutral
 */

import {
  Document,
  Packer,
  Paragraph,
  TextRun,
  HeadingLevel,
  AlignmentType,
  Table,
  TableRow,
  TableCell,
  WidthType,
  BorderStyle,
  ShadingType,
  Footer,
  PageNumber,
} from "docx";

// ────────────────────────────────────────────────────────────────────────────
// Brand colors and section names are sourced from the shared design system
// so the live UI and the Word report stay visually + structurally aligned.
// ────────────────────────────────────────────────────────────────────────────
import {
  VS_COLORS as VS_COLORS_SHARED,
  VS_REPORT_SECTIONS,
} from "@/lib/vivasenseDesignSystem";

export const VS_COLORS = VS_COLORS_SHARED;
export const VS_SECTIONS = VS_REPORT_SECTIONS;

const FIXED_INTERPRETATION_NOTE =
  "Statistical results describe association and variability under the model's assumptions, not causation. Domain-specific conclusions should be made in the context of study design, measurement quality, and subject-matter knowledge.";

// ────────────────────────────────────────────────────────────────────────────
// Number formatting
// ────────────────────────────────────────────────────────────────────────────
export function vsFmtNumber(n: number, digits = 2): string {
  if (!Number.isFinite(n)) return "—";
  return n.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function vsFmtP(p: number): string {
  if (!Number.isFinite(p)) return "—";
  if (p < 0.001) return "< 0.001 ***";
  if (p < 0.01) return `${p.toFixed(3)} **`;
  if (p < 0.05) return `${p.toFixed(3)} *`;
  return p.toFixed(3);
}

// ────────────────────────────────────────────────────────────────────────────
// Paragraph & table primitives
// ────────────────────────────────────────────────────────────────────────────
export interface VsParaOpts {
  bold?: boolean;
  italic?: boolean;
  color?: string;
  size?: number; // half-points
  heading?: (typeof HeadingLevel)[keyof typeof HeadingLevel];
  align?: (typeof AlignmentType)[keyof typeof AlignmentType];
  spaceAfter?: number;
  spaceBefore?: number;
}

export function vsP(text: string, opts: VsParaOpts = {}): Paragraph {
  return new Paragraph({
    heading: opts.heading,
    alignment: opts.align,
    spacing: { after: opts.spaceAfter ?? 120, before: opts.spaceBefore },
    children: [
      new TextRun({
        text,
        bold: opts.bold,
        italics: opts.italic,
        color: opts.color,
        size: opts.size,
      }),
    ],
  });
}

export function vsBullet(text: string, color?: string): Paragraph {
  return new Paragraph({
    spacing: { after: 80 },
    indent: { left: 360, hanging: 180 },
    children: [
      new TextRun({ text: "• ", bold: true, color }),
      new TextRun({ text, color }),
    ],
  });
}

export function vsSectionHeading(text: string): Paragraph {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 140 },
    children: [
      new TextRun({ text, bold: true, color: VS_COLORS.ink, size: 28 }),
    ],
  });
}

// Two-column key/value table row with subtle grid
export function vsKvRow(key: string, value: string): TableRow {
  const border = { style: BorderStyle.SINGLE, size: 4, color: VS_COLORS.cellBorder };
  const borders = { top: border, bottom: border, left: border, right: border };
  return new TableRow({
    children: [
      new TableCell({
        borders,
        width: { size: 3000, type: WidthType.DXA },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        shading: { fill: VS_COLORS.headerFill, type: ShadingType.CLEAR, color: "auto" },
        children: [new Paragraph({ children: [new TextRun({ text: key, bold: true })] })],
      }),
      new TableCell({
        borders,
        width: { size: 6360, type: WidthType.DXA },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [new Paragraph({ children: [new TextRun(value)] })],
      }),
    ],
  });
}

export function vsKvTable(rows: Array<[string, string]>): Table {
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [3000, 6360],
    rows: rows.map(([k, v]) => vsKvRow(k, v)),
  });
}

// Generic data table with header row
export function vsDataTable(headers: string[], rows: string[][]): Table {
  const border = { style: BorderStyle.SINGLE, size: 4, color: VS_COLORS.cellBorder };
  const borders = { top: border, bottom: border, left: border, right: border };
  const totalWidth = 9360;
  const colWidth = Math.floor(totalWidth / headers.length);
  const columnWidths = headers.map(() => colWidth);

  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h) =>
      new TableCell({
        borders,
        width: { size: colWidth, type: WidthType.DXA },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        shading: { fill: VS_COLORS.headerFill, type: ShadingType.CLEAR, color: "auto" },
        children: [new Paragraph({ children: [new TextRun({ text: h, bold: true })] })],
      })
    ),
  });

  const bodyRows = rows.map((row) =>
    new TableRow({
      children: row.map((cell) =>
        new TableCell({
          borders,
          width: { size: colWidth, type: WidthType.DXA },
          margins: { top: 60, bottom: 60, left: 120, right: 120 },
          children: [new Paragraph({ children: [new TextRun(cell)] })],
        })
      ),
    })
  );

  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths,
    rows: [headerRow, ...bodyRows],
  });
}

// ────────────────────────────────────────────────────────────────────────────
// Branded title block (top of every report)
// ────────────────────────────────────────────────────────────────────────────
export interface VsHeaderInput {
  /** Module label, e.g. "Regression", "ANOVA" */
  moduleType: string;
  /** Short context line, e.g. "Linear model · n = 25 · 1 predictor" */
  contextLine?: string;
  /** Override generated date (defaults to now) */
  date?: Date;
}

export function vsHeader(input: VsHeaderInput): Paragraph[] {
  const date = input.date ?? new Date();
  const dateStr = date.toLocaleString();
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 80 },
      children: [
        new TextRun({
          text: "VivaSense Statistical Analysis Report",
          bold: true,
          size: 40,
          color: VS_COLORS.ink,
        }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 60 },
      children: [
        new TextRun({
          text: `Module: ${input.moduleType}`,
          bold: true,
          size: 24,
          color: VS_COLORS.brand,
        }),
      ],
    }),
    ...(input.contextLine
      ? [
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 60 },
            children: [
              new TextRun({
                text: input.contextLine,
                size: 20,
                color: VS_COLORS.muted,
              }),
            ],
          }),
        ]
      : []),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 280 },
      children: [
        new TextRun({
          text: `Report generated: ${dateStr}`,
          italics: true,
          size: 18,
          color: VS_COLORS.muted,
        }),
      ],
    }),
  ];
}

// ────────────────────────────────────────────────────────────────────────────
// Caution block (red) and "all clear" block (green)
// ────────────────────────────────────────────────────────────────────────────
export function vsCautionParagraphs(messages: string[]): Paragraph[] {
  if (messages.length === 0) {
    return [
      vsP("✓ No major reliability warnings were detected for this model.", {
        color: VS_COLORS.good,
        bold: true,
      }),
    ];
  }
  return messages.map((m) => vsBullet(m, VS_COLORS.bad));
}

// ────────────────────────────────────────────────────────────────────────────
// Footer signature
// ────────────────────────────────────────────────────────────────────────────
function buildFooter(): Footer {
  return new Footer({
    children: [
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 80, after: 0 },
        children: [
          new TextRun({
            text: "VivaSense Engine v1.0 · Statistical analysis powered by Dockerized R backend",
            size: 16,
            color: VS_COLORS.muted,
          }),
        ],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 0 },
        children: [
          new TextRun({
            text: "“From Statistical Output to Scientific Insight”",
            italics: true,
            size: 16,
            color: VS_COLORS.brand,
          }),
        ],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [
          new TextRun({ text: "Page ", size: 14, color: VS_COLORS.muted }),
          new TextRun({ children: [PageNumber.CURRENT], size: 14, color: VS_COLORS.muted }),
          new TextRun({ text: " of ", size: 14, color: VS_COLORS.muted }),
          new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 14, color: VS_COLORS.muted }),
        ],
      }),
    ],
  });
}

// ────────────────────────────────────────────────────────────────────────────
// Standard interpretation note (footer of every report's body)
// ────────────────────────────────────────────────────────────────────────────
export function vsInterpretationNote(): Paragraph {
  return vsP(FIXED_INTERPRETATION_NOTE, { italic: true, color: VS_COLORS.muted });
}

// ────────────────────────────────────────────────────────────────────────────
// Chat history rendering helper
// ────────────────────────────────────────────────────────────────────────────
export interface VsChatMsg {
  role: "user" | "assistant" | "system";
  content: string;
}

export function vsChatHistory(history: VsChatMsg[]): Paragraph[] {
  if (history.length === 0) {
    return [
      vsP("No follow-up conversation was recorded for this analysis.", {
        italic: true,
        color: VS_COLORS.muted,
      }),
    ];
  }
  const out: Paragraph[] = [];
  history.forEach((m) => {
    const label =
      m.role === "user" ? "Researcher" : m.role === "assistant" ? "VivaSense" : "Note";
    out.push(vsP(`${label}:`, { bold: true, color: VS_COLORS.brand }));
    out.push(vsP(m.content));
  });
  return out;
}

// ────────────────────────────────────────────────────────────────────────────
// Document assembly
// ────────────────────────────────────────────────────────────────────────────
export interface VsDocumentInput {
  header: VsHeaderInput;
  /** Body content between header and footer (already includes section headings) */
  body: (Paragraph | Table)[];
}

export function buildVivaSenseDocument(input: VsDocumentInput): Document {
  return new Document({
    styles: {
      default: {
        document: { run: { font: "Calibri", size: 22, color: VS_COLORS.ink } },
      },
    },
    sections: [
      {
        properties: {
          page: {
            size: { width: 12240, height: 15840 },
            margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
          },
        },
        footers: { default: buildFooter() },
        children: [...vsHeader(input.header), ...input.body],
      },
    ],
  });
}

export async function downloadVivaSenseDocument(
  input: VsDocumentInput,
  filename: string,
): Promise<void> {
  const doc = buildVivaSenseDocument(input);
  const blob = await Packer.toBlob(doc);
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
