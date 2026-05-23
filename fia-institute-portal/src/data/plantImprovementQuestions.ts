// ============================================================
// Plant Improvement — Centre of Origin / Domestication
// Expanded question bank for the Smart Quiz Engine
// ============================================================

export type QuestionType = "mcq" | "tf" | "fill" | "scenario";
export type Difficulty = "easy" | "medium" | "hard";
export type Topic =
  | "vavilov"
  | "origin"
  | "diversity"
  | "domestication"
  | "syndrome"
  | "introduction"
  | "nigeria"
  | "molecular"
  | "conservation";

export interface Question {
  id: string;
  type: QuestionType;
  topic: Topic;
  difficulty: Difficulty;
  q: string;
  options?: string[];        // for mcq, tf
  answer: number | string;   // index for mcq/tf, accepted string for fill
  acceptable?: string[];     // additional accepted answers (fill)
  explain: string;
}

export const TOPIC_LABELS: Record<Topic, string> = {
  vavilov: "Vavilov Centres",
  origin: "Centre of Origin",
  diversity: "Genetic Diversity",
  domestication: "Domestication",
  syndrome: "Domestication Syndrome",
  introduction: "Plant Introduction",
  nigeria: "Nigerian Crops",
  molecular: "Molecular Evidence",
  conservation: "Germplasm Conservation",
};

export const QUESTION_BANK: Question[] = [
  // ===== VAVILOV =====
  {
    id: "v1", type: "mcq", topic: "vavilov", difficulty: "easy",
    q: "Who proposed the theory of Centres of Origin of Cultivated Plants?",
    options: ["Charles Darwin", "N.I. Vavilov", "Gregor Mendel", "Norman Borlaug"],
    answer: 1,
    explain: "Russian botanist Nikolai Ivanovich Vavilov proposed 8 centres of origin in the 1920s–30s based on 115 expeditions and >250,000 accessions.",
  },
  {
    id: "v2", type: "mcq", topic: "vavilov", difficulty: "medium",
    q: "How many primary centres of origin did Vavilov originally propose in 1926?",
    options: ["5", "8", "11", "12"],
    answer: 1,
    explain: "Vavilov proposed 8 primary centres in 1926, later revised to 11 in 1935.",
  },
  {
    id: "v3", type: "tf", topic: "vavilov", difficulty: "easy",
    q: "True or False: Vavilov's Law of Homologous Series in Variation predicts parallel variation across related species.",
    options: ["True", "False"],
    answer: 0,
    explain: "True. The law states that genetically related species and genera show similar parallel series of heritable variation.",
  },
  {
    id: "v4", type: "fill", topic: "vavilov", difficulty: "medium",
    q: "The seed bank founded by Vavilov in Russia is known as the ____ Institute (acronym).",
    answer: "VIR",
    acceptable: ["vir", "n.i. vavilov research institute"],
    explain: "VIR — the N.I. Vavilov All-Russian Institute of Plant Genetic Resources — preserves >320,000 accessions.",
  },
  {
    id: "v5", type: "mcq", topic: "vavilov", difficulty: "hard",
    q: "Which is NOT one of Vavilov's primary centres?",
    options: ["Abyssinian (Ethiopian)", "South American", "Australian", "Mediterranean"],
    answer: 2,
    explain: "Australia is not a Vavilov centre — no major crop independently domesticated there before European contact.",
  },

  // ===== ORIGIN =====
  {
    id: "o1", type: "mcq", topic: "origin", difficulty: "easy",
    q: "Wheat was first domesticated in:",
    options: ["Mesoamerica", "Near East (Fertile Crescent)", "Andes", "Indo-China"],
    answer: 1,
    explain: "Wheat originated in the Fertile Crescent ~10,000 years before present (BP).",
  },
  {
    id: "o2", type: "mcq", topic: "origin", difficulty: "easy",
    q: "Maize (Zea mays) was domesticated in:",
    options: ["Africa", "Andes", "Mesoamerica (Mexico)", "China"],
    answer: 2,
    explain: "Maize was domesticated from teosinte in southern Mexico around 9,000 BP.",
  },
  {
    id: "o3", type: "fill", topic: "origin", difficulty: "medium",
    q: "The wild ancestor of maize is called ____.",
    answer: "teosinte",
    acceptable: ["zea mays parviglumis", "balsas teosinte"],
    explain: "Teosinte (Zea mays subsp. parviglumis) of the Balsas River valley is the direct wild progenitor of maize.",
  },
  {
    id: "o4", type: "mcq", topic: "origin", difficulty: "medium",
    q: "Potatoes were domesticated in which Vavilov centre?",
    options: ["Mediterranean", "South American (Andes)", "Central Asiatic", "Chinese"],
    answer: 1,
    explain: "Potatoes were domesticated in the Andes (Peru/Bolivia) ~8,000 BP.",
  },

  // ===== DIVERSITY =====
  {
    id: "d1", type: "mcq", topic: "diversity", difficulty: "medium",
    q: "A SECONDARY centre of diversity is best described as:",
    options: [
      "Where a crop was first domesticated",
      "A region where significant diversity arose AFTER the crop's introduction",
      "The smallest region with high diversity",
      "Where wild relatives went extinct",
    ],
    answer: 1,
    explain: "Secondary centres develop new diversity after introduction — e.g., maize in Africa, cassava in Africa.",
  },
  {
    id: "d2", type: "tf", topic: "diversity", difficulty: "easy",
    q: "True or False: Centres of origin typically host the wild relatives of crops.",
    options: ["True", "False"],
    answer: 0,
    explain: "True. Wild relatives co-occur with crops in their centres of origin and carry valuable alleles for resistance and adaptation.",
  },
  {
    id: "d3", type: "scenario", topic: "diversity", difficulty: "hard",
    q: "A breeder needs novel rust resistance for wheat. From which region should they ideally collect germplasm?",
    options: [
      "European wheat fields",
      "The Fertile Crescent (wheat's centre of origin)",
      "Australian wheat farms",
      "North American Plains",
    ],
    answer: 1,
    explain: "The centre of origin contains the deepest pool of wild relatives carrying novel resistance alleles, including for stem and leaf rust.",
  },

  // ===== DOMESTICATION =====
  {
    id: "dm1", type: "mcq", topic: "domestication", difficulty: "easy",
    q: "Domestication is best described as:",
    options: [
      "Random mutation",
      "Human-directed selection over generations",
      "Natural climate change",
      "Genetic engineering",
    ],
    answer: 1,
    explain: "Vavilov defined plant breeding as 'evolution directed by man' — domestication is cumulative human selection over millennia.",
  },
  {
    id: "dm2", type: "fill", topic: "domestication", difficulty: "medium",
    q: "Approximately how many years ago did agriculture and crop domestication begin? (write a single number in thousands, e.g. 10)",
    answer: "10",
    acceptable: ["10000", "10,000", "~10", "≈10"],
    explain: "Crop domestication began roughly 10,000 years BP at the start of the Neolithic.",
  },

  // ===== DOMESTICATION SYNDROME =====
  {
    id: "s1", type: "mcq", topic: "syndrome", difficulty: "medium",
    q: "Which trait is NOT part of the domestication syndrome?",
    options: ["Loss of shattering", "Increased seed dormancy", "Larger seeds", "Reduced bitterness"],
    answer: 1,
    explain: "Domestication REDUCES seed dormancy to ensure uniform germination after planting.",
  },
  {
    id: "s2", type: "tf", topic: "syndrome", difficulty: "easy",
    q: "True or False: A non-shattering rachis is a hallmark of cereal domestication.",
    options: ["True", "False"],
    answer: 0,
    explain: "True. Wild cereals shatter to disperse seeds; domesticated cereals retain seeds for harvest.",
  },
  {
    id: "s3", type: "mcq", topic: "syndrome", difficulty: "hard",
    q: "Bitterness loss in domesticated cassava reflects reduced levels of:",
    options: ["Tannins", "Cyanogenic glycosides", "Alkaloids", "Saponins"],
    answer: 1,
    explain: "Sweet cassava cultivars have reduced cyanogenic glycosides (linamarin) compared with bitter wild types.",
  },
  {
    id: "s4", type: "fill", topic: "syndrome", difficulty: "medium",
    q: "Domesticated crops are often ____-neutral, meaning they flower regardless of day length.",
    answer: "day",
    acceptable: ["photoperiod"],
    explain: "Domestication often selects for day-neutral or photoperiod-insensitive flowering for wider adaptation.",
  },

  // ===== INTRODUCTION =====
  {
    id: "i1", type: "mcq", topic: "introduction", difficulty: "easy",
    q: "Plant introduction primarily refers to:",
    options: [
      "Introducing students to plants",
      "Deliberate movement of germplasm across regions",
      "Plant breeding via hybridisation",
      "Genetic transformation",
    ],
    answer: 1,
    explain: "Plant introduction is the deliberate transfer of genotypes from one region to another for cultivation or breeding.",
  },
  {
    id: "i2", type: "mcq", topic: "introduction", difficulty: "medium",
    q: "The CORRECT order of plant introduction procedure is:",
    options: [
      "Collection → Quarantine → Evaluation → Multiplication → Release",
      "Quarantine → Collection → Release → Evaluation",
      "Evaluation → Quarantine → Collection → Release",
      "Release → Multiplication → Evaluation → Collection",
    ],
    answer: 0,
    explain: "Standard sequence: Collect → Quarantine (prevent pest entry) → Evaluate → Multiply → Release.",
  },
  {
    id: "i3", type: "fill", topic: "introduction", difficulty: "medium",
    q: "The international agreement (1992) protecting access to genetic resources is known as the ____.",
    answer: "CBD",
    acceptable: ["convention on biological diversity"],
    explain: "The Convention on Biological Diversity (CBD, 1992) and the Nagoya Protocol (2010) regulate access and benefit-sharing.",
  },

  // ===== NIGERIA =====
  {
    id: "n1", type: "mcq", topic: "nigeria", difficulty: "easy",
    q: "Cassava was introduced to Nigeria from:",
    options: ["India", "Brazil (South America)", "China", "Ethiopia"],
    answer: 1,
    explain: "Cassava originated in Brazil/Amazon and was introduced to Africa by Portuguese traders in the 16th century.",
  },
  {
    id: "n2", type: "mcq", topic: "nigeria", difficulty: "medium",
    q: "Which crop is INDIGENOUS to West Africa?",
    options: ["Cassava", "Maize", "Cowpea", "Tomato"],
    answer: 2,
    explain: "Cowpea (Vigna unguiculata) was domesticated in West/Central Africa. Nigeria is the world's largest producer.",
  },
  {
    id: "n3", type: "tf", topic: "nigeria", difficulty: "medium",
    q: "True or False: Oryza glaberrima is an African rice species indigenous to West Africa.",
    options: ["True", "False"],
    answer: 0,
    explain: "True. O. glaberrima was domesticated in the inland Niger Delta ~3,500 years ago, alongside the later-introduced O. sativa.",
  },
  {
    id: "n4", type: "scenario", topic: "nigeria", difficulty: "hard",
    q: "A Nigerian breeder wants to improve yam tuber dormancy. The richest source of useful alleles is most likely:",
    options: [
      "Asian Dioscorea germplasm",
      "West African wild Dioscorea relatives",
      "European yam landraces",
      "American sweet potato collections",
    ],
    answer: 1,
    explain: "West Africa is the yam centre of origin and hosts wild Dioscorea relatives carrying useful dormancy and disease-resistance alleles.",
  },

  // ===== MOLECULAR =====
  {
    id: "m1", type: "mcq", topic: "molecular", difficulty: "hard",
    q: "Which gene is famously associated with the architectural transition from teosinte to maize?",
    options: ["Tb1 (teosinte branched1)", "Rht1", "Sd1", "Waxy"],
    answer: 0,
    explain: "Tb1 represses lateral branching, giving maize its single-stalk architecture compared with bushy teosinte.",
  },
  {
    id: "m2", type: "tf", topic: "molecular", difficulty: "medium",
    q: "True or False: Domestication often leaves a 'selective sweep' signature reducing diversity at selected loci.",
    options: ["True", "False"],
    answer: 0,
    explain: "True. Strong selection during domestication reduces nucleotide diversity around targeted genes — a hallmark detectable by molecular scans.",
  },

  // ===== CONSERVATION =====
  {
    id: "c1", type: "mcq", topic: "conservation", difficulty: "easy",
    q: "Ex situ conservation of crop genetic resources is typified by:",
    options: ["On-farm landrace cultivation", "Gene banks and seed storage", "Protected forests", "Botanical gardens only"],
    answer: 1,
    explain: "Ex situ = off-site preservation, principally in gene banks (e.g., Svalbard, IITA, NACGRAB).",
  },
  {
    id: "c2", type: "fill", topic: "conservation", difficulty: "medium",
    q: "The international gene bank based in Ibadan, Nigeria, is the ____.",
    answer: "IITA",
    acceptable: ["international institute of tropical agriculture"],
    explain: "IITA holds the world's largest collections of cowpea, cassava and yam germplasm.",
  },
  {
    id: "c3", type: "scenario", topic: "conservation", difficulty: "hard",
    q: "A breeder loses a key landrace to drought. The fastest route to recovery is:",
    options: [
      "Resequencing modern cultivars",
      "Requesting accessions from a gene bank holding that landrace",
      "Importing unrelated cultivars",
      "Waiting for natural regeneration",
    ],
    answer: 1,
    explain: "Gene banks exist precisely for this purpose — ex situ accessions safeguard against loss in the field.",
  },
];

// Utility: filter + shuffle + adaptive selection
export function pickQuestions(opts: {
  topics?: Topic[];
  difficulty?: Difficulty | "mixed";
  count?: number;
}): Question[] {
  const { topics, difficulty = "mixed", count = 10 } = opts;
  let pool = QUESTION_BANK.slice();
  if (topics && topics.length) pool = pool.filter((q) => topics.includes(q.topic));
  if (difficulty !== "mixed") pool = pool.filter((q) => q.difficulty === difficulty);
  // Fisher–Yates shuffle
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  return pool.slice(0, Math.min(count, pool.length));
}
