import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import PPTXGenJS from "pptxgenjs";

const DEFAULT_INPUT =
  "C:\\Users\\ADMIN\\Downloads\\CSP502_Lecture_Notes_Centre_of_Origin_Domestication_PlantIntroduction.docx";
const DEFAULT_OUTPUT = path.resolve(
  process.cwd(),
  "CSP502_Centre_of_Origin_Domestication_PlantIntroduction.pptx",
);

function decodeXmlEntities(value) {
  return value
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&#39;/g, "'")
    .replace(/&#x([0-9a-fA-F]+);/g, (_, hex) => String.fromCodePoint(Number.parseInt(hex, 16)))
    .replace(/&#(\d+);/g, (_, decimal) => String.fromCodePoint(Number.parseInt(decimal, 10)));
}

function extractDocxXml(docxPath) {
  const escapedPath = docxPath.replace(/'/g, "''");
  const powershellScript = `
$ErrorActionPreference = 'Stop'
$docxPath = '${escapedPath}'
$tempDir = Join-Path $env:TEMP ("lecture-" + [guid]::NewGuid().ToString("N"))
try {
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  New-Item -ItemType Directory -Path $tempDir | Out-Null
  [System.IO.Compression.ZipFile]::ExtractToDirectory($docxPath, $tempDir)
  $xmlPath = Join-Path $tempDir 'word/document.xml'
  Get-Content -LiteralPath $xmlPath -Raw
}
finally {
  if (Test-Path $tempDir) {
    Remove-Item -LiteralPath $tempDir -Recurse -Force
  }
}
`;

  return execFileSync(
    "powershell",
    ["-NoProfile", "-Command", powershellScript],
    {
      encoding: "utf8",
      maxBuffer: 20 * 1024 * 1024,
    },
  );
}

function extractParagraphs(xml) {
  const paragraphs = [];
  const paragraphMatches = xml.match(/<w:p[\s\S]*?<\/w:p>/g) ?? [];

  for (const paragraphXml of paragraphMatches) {
    const texts = [...paragraphXml.matchAll(/<w:t[^>]*>([\s\S]*?)<\/w:t>/g)].map((match) =>
      decodeXmlEntities(match[1]),
    );
    const paragraph = texts.join("").replace(/\s+/g, " ").trim();
    if (paragraph) paragraphs.push(paragraph);
  }

  return paragraphs;
}

function normalize(text) {
  return text.replace(/\s+/g, " ").trim();
}

function isSectionHeading(text) {
  return /^SECTION [A-D]:/i.test(text);
}

function isSubheading(text) {
  return (
    /^\d+(?:\.\d+)+\s+/.test(text) ||
    /—\s*\d+\s*Minutes$/i.test(text) ||
    /^Learning Objectives$/i.test(text) ||
    /^EXAM TIP$/i.test(text) ||
    /^Why Crop History Matters/i.test(text) ||
    /^Three features characterise/i.test(text) ||
    /^Primary and Secondary Centres/i.test(text) ||
    /^Stages of Domestication/i.test(text) ||
    /^Domestication Syndrome/i.test(text) ||
    /^Plant Introduction/i.test(text) ||
    /^Types of Plant Introduction/i.test(text) ||
    /^Steps in Plant Introduction/i.test(text)
  );
}

function sectionTitle(text) {
  return text.replace(/^SECTION ([A-D]):\s*/i, (_, letter) => `Section ${letter.toUpperCase()}: `);
}

function splitBullets(paragraphs) {
  const bullets = [];

  for (const paragraph of paragraphs) {
    const compact = normalize(paragraph)
      .replace(/^[-•]\s*/, "")
      .replace(/^\d+\.\s*/, "");

    if (!compact) continue;

    if (compact.length <= 180) {
      bullets.push(compact);
      continue;
    }

    const clauses = compact
      .split(/(?<=[.;:])\s+(?=[A-Z(])/)
      .map((clause) => clause.trim())
      .filter(Boolean);

    if (clauses.length > 1) {
      bullets.push(...clauses);
    } else {
      bullets.push(compact);
    }
  }

  return bullets;
}

function buildBlocks(paragraphs) {
  const blocks = [];
  let current = null;

  for (const paragraph of paragraphs) {
    const text = normalize(paragraph);
    if (!text) continue;

    if (/^(FEDERAL UNIVERSITY|Department of Crop, Soil and Pest Management|CSP 502 — Plant Improvement|LECTURE NOTES|Centre of Origin of Crops, Domestication & Plant Introduction|Prepared by:|Dr\. Fayeun Lawrence Stephen|Academic Session 2024\/2025|Course Code|Level|Duration|Sections)$/i.test(text)) {
      continue;
    }

    if (text === "+ 5 min Q&A") {
      continue;
    }

    if (text === "Learning Objectives") {
      current = { title: "Learning Objectives", items: [] };
      blocks.push(current);
      continue;
    }

    if (text === "EXAM TIP") {
      current = { title: "Exam Tip", items: [] };
      blocks.push(current);
      continue;
    }

    if (isSectionHeading(text)) {
      current = { title: sectionTitle(text), items: [] };
      blocks.push(current);
      continue;
    }

    if (isSubheading(text)) {
      current = { title: text, items: [] };
      blocks.push(current);
      continue;
    }

    if (!current) continue;
    current.items.push(text);
  }

  return blocks
    .map((block) => ({
      title: block.title,
      bullets: splitBullets(block.items),
    }))
    .filter((block) => block.title && block.bullets.length > 0);
}

function createSlide(pptx, { title, bullets, accent, subtitle }) {
  const slide = pptx.addSlide();
  slide.background = { color: "F8FAF4" };

  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: 13.333,
    h: 0.58,
    line: { color: accent, transparency: 100 },
    fill: { color: accent },
  });

  slide.addText(title, {
    x: 0.6,
    y: 0.18,
    w: 12.1,
    h: 0.34,
    fontFace: "Aptos Display",
    color: "FFFFFF",
    bold: true,
    size: 24,
  });

  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.7,
      y: 0.82,
      w: 12,
      h: 0.35,
      fontFace: "Aptos",
      size: 12,
      color: "56606B",
      italic: true,
    });
  }

  slide.addText(
    bullets.map((bullet) => ({ text: bullet, options: { bullet: { indent: 18 } } })),
    {
      x: 0.8,
      y: subtitle ? 1.25 : 1.0,
      w: 11.8,
      h: 5.65,
      fontFace: "Aptos",
      color: "203020",
      size: 17,
      valign: "top",
      breakLine: true,
      paraSpaceAfterPt: 9,
      fit: "shrink",
    },
  );

  slide.addText("CSP 502 · Centre of Origin, Domestication & Plant Introduction", {
    x: 0.7,
    y: 6.92,
    w: 8.2,
    h: 0.22,
    fontFace: "Aptos",
    size: 9,
    color: "6B7B69",
  });

  return slide;
}

function addTitleSlide(pptx, meta) {
  const slide = pptx.addSlide();
  slide.background = { color: "EAF2E5" };

  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: 13.333,
    h: 7.5,
    line: { color: "EAF2E5", transparency: 100 },
    fill: { color: "EAF2E5" },
  });

  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: 13.333,
    h: 0.5,
    line: { color: "235B2A", transparency: 100 },
    fill: { color: "235B2A" },
  });

  slide.addText(meta.course, {
    x: 0.8,
    y: 0.88,
    w: 5.6,
    h: 0.32,
    fontFace: "Aptos",
    color: "235B2A",
    size: 14,
    bold: true,
  });

  slide.addText(meta.title, {
    x: 0.78,
    y: 1.42,
    w: 11.6,
    h: 1.25,
    fontFace: "Aptos Display",
    color: "1C4D24",
    bold: true,
    size: 30,
    margin: 0,
  });

  slide.addText(meta.subtitle, {
    x: 0.82,
    y: 2.8,
    w: 11.2,
    h: 0.45,
    fontFace: "Aptos",
    color: "35583C",
    size: 15,
    italic: true,
  });

  slide.addText([
    { text: `Prepared by: ${meta.author}` },
    { text: `Academic session: ${meta.session}`, options: { breakLine: true } },
    { text: `Source file: ${meta.sourceFile}`, options: { breakLine: true } },
  ], {
    x: 0.84,
    y: 3.45,
    w: 6.2,
    h: 1.0,
    fontFace: "Aptos",
    color: "415A46",
    size: 14,
    breakLine: true,
    paraSpaceAfterPt: 8,
  });

  slide.addShape(pptx.ShapeType.roundRect, {
    x: 8.5,
    y: 1.2,
    w: 3.8,
    h: 3.9,
    rectRadius: 0.12,
    line: { color: "C9D8C4", pt: 1 },
    fill: { color: "FFFFFF", transparency: 3 },
  });

  slide.addText(
    [
      { text: "Lecture Focus", options: { bold: true, fontFace: "Aptos Display", color: "1C4D24" } },
      {
        text:
          "Centres of origin, Vavilov's theory, crop domestication, domestication syndrome, and plant introduction.",
        options: { breakLine: true },
      },
    ],
    {
      x: 8.82,
      y: 1.58,
      w: 3.2,
      h: 2.7,
      fontFace: "Aptos",
      size: 15,
      color: "24402B",
      valign: "mid",
      fit: "shrink",
    },
  );

  slide.addText("Derived from the lecture notes in your Downloads folder.", {
    x: 0.85,
    y: 6.62,
    w: 5.8,
    h: 0.24,
    fontFace: "Aptos",
    size: 10,
    color: "6B7B69",
  });
}

function addObjectivesSlide(pptx, objectives, accent) {
  createSlide(pptx, {
    title: "Learning Objectives",
    bullets: objectives,
    accent,
    subtitle: "The document's stated learning outcomes.",
  });
}

function buildPptx({ paragraphs, sourcePath, outputPath }) {
  const titleParagraph = paragraphs.find((paragraph) => /LECTURE NOTES/i.test(paragraph)) ?? "CSP 502 Lecture Notes";
  const topicParagraph = paragraphs.find((paragraph) => /Centre of Origin of Crops, Domestication/i.test(paragraph)) ?? "Centre of Origin of Crops, Domestication & Plant Introduction";
  const sessionParagraph = paragraphs.find((paragraph) => /Academic Session/i.test(paragraph)) ?? "Academic Session 2024/2025";
  const authorParagraph = paragraphs.find((paragraph) => /Prepared by:/i.test(paragraph)) ?? "Dr. Fayeun Lawrence Stephen";

  const objectivesStart = paragraphs.findIndex((paragraph) => /Learning Objectives/i.test(paragraph));
  const objectivesEnd = paragraphs.findIndex((paragraph, index) => index > objectivesStart && /EXAM TIP/i.test(paragraph));
  const objectives = objectivesStart >= 0 ? paragraphs.slice(objectivesStart + 1, objectivesEnd > objectivesStart ? objectivesEnd : undefined) : [];

  const blocks = buildBlocks(paragraphs);
  const pptx = new PPTXGenJS();
  pptx.layout = "LAYOUT_WIDE";
  pptx.author = authorParagraph.replace(/^Prepared by:\s*/i, "");
  pptx.company = "FIA Institute Portal";
  pptx.subject = topicParagraph;
  pptx.title = topicParagraph;
  pptx.lang = "en-US";
  pptx.theme = {
    headFontFace: "Aptos Display",
    bodyFontFace: "Aptos",
    lang: "en-US",
  };

  addTitleSlide(pptx, {
    course: "CSP 502 · Plant Improvement",
    title: titleParagraph,
    subtitle: topicParagraph,
    author: authorParagraph.replace(/^Prepared by:\s*/i, ""),
    session: sessionParagraph.replace(/^Academic Session:\s*/i, "") || sessionParagraph,
    sourceFile: path.basename(sourcePath),
  });

  if (objectives.length > 0) {
    addObjectivesSlide(pptx, objectives.slice(0, 7), "2E6B31");
  }

  const accentPalette = ["2E6B31", "2F6DA3", "8A5F14", "7B4E9E", "B04D2C", "296E7A"];
  let accentIndex = 0;

  for (const block of blocks) {
    const chunks = [];
    for (let index = 0; index < block.bullets.length; index += 5) {
      chunks.push(block.bullets.slice(index, index + 5));
    }

    for (let index = 0; index < chunks.length; index += 1) {
      createSlide(pptx, {
        title: index === 0 ? block.title : `${block.title} (continued)`,
        bullets: chunks[index],
        accent: accentPalette[accentIndex % accentPalette.length],
        subtitle: index === 0 ? "Extracted from the supplied lecture notes." : undefined,
      });
      accentIndex += 1;
    }
  }

  createSlide(pptx, {
    title: "Revision Takeaways",
    bullets: [
      "Know the definition and significance of centres of origin for plant breeding.",
      "Be able to explain Vavilov's theory and distinguish primary, secondary, and micro centres.",
      "Remember the domestication syndrome as a set of human-selected crop traits.",
      "Understand the steps, objectives, and quarantine importance of plant introduction.",
    ],
    accent: "235B2A",
    subtitle: "A final review slide for quick exam revision.",
  });

  return pptx.writeFile({ fileName: outputPath });
}

async function main() {
  const inputPath = path.resolve(process.argv[2] ?? DEFAULT_INPUT);
  const outputPath = path.resolve(process.argv[3] ?? DEFAULT_OUTPUT);

  if (!fs.existsSync(inputPath)) {
    throw new Error(`Input DOCX not found: ${inputPath}`);
  }

  const xml = extractDocxXml(inputPath);
  const paragraphs = extractParagraphs(xml);

  if (paragraphs.length === 0) {
    throw new Error("No readable paragraphs were found in the DOCX file.");
  }

  await buildPptx({ paragraphs, sourcePath: inputPath, outputPath });
  console.log(`Created ${outputPath}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});