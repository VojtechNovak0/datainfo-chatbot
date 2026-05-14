const express = require("express");
const axios = require("axios");
const pdfjsLib = require("pdfjs-dist/legacy/build/pdf.js");
require("dotenv").config();
const cheerio = require("cheerio");
const fs = require("fs");

const pdfUrls = [
  "http://erp.oznameni.datainfo.cz/wp-content/uploads/2024/09/Datainfo-Jak-na-vodne-a-stocne.pdf",
  "http://erp.oznameni.datainfo.cz/wp-content/uploads/2024/09/Datainfo-Jak-na-zalohy.pdf"
];

const WEB_CACHE_FILE = "./webIndex.json";
const PDF_CACHE_FILE = "./pdfIndex.json";
const BASE_URL = "https://help.datainfo.cz";
const MAX_PAGES = 20;
let discoveredPDFs = new Set();
let pipeline;

async function loadEmbedder() {
  const transformers = await import("@xenova/transformers");
  pipeline = transformers.pipeline;
}

function loadJson(file) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf-8"));
  } catch {
    return null;
  }
}

function saveJson(file, data) {
  fs.writeFileSync(file, JSON.stringify(data, null, 2));
}

let webIndex = [];
let visited = new Set();

const app = express();
app.use(express.json());
app.use(express.static("public"));

const GROQ_API_KEY = process.env.GROQ_API_KEY;

let chunks = [];
let embedder;

// ---------------- EMBEDDING ----------------
async function embed(text) {
  const output = await embedder(text, {
    pooling: "mean",
    normalize: true
  });

  return Array.from(output.data);
}

// ---------------- COSINE ----------------
function cosine(a, b) {
  if (!a || !b) return 0;

  let dot = 0, magA = 0, magB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }

  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

function splitText(text, size = 500, overlap = 100) {

  const chunks = [];

  for (let i = 0; i < text.length; i += size - overlap) {

    chunks.push(
      text.slice(i, i + size)
    );

  }

  return chunks;
}

function normalize(text) {
  return text
    .toLowerCase()
    .replace(/ě/g, "e")
    .replace(/š/g, "s")
    .replace(/č/g, "c")
    .replace(/ř/g, "r")
    .replace(/ž/g, "z")
    .replace(/ý/g, "y")
    .replace(/á/g, "a")
    .replace(/í/g, "i")
    .replace(/é/g, "e")
    .replace(/ů/g, "u")
    .replace(/ú/g, "u");
}

function score(text, query) {

  const t = normalize(text);
  const q = normalize(query);

  let score = 0;

  const queryWords = q
    .split(/\s+/)
    .filter(w => w.length > 2);

  for (const word of queryWords) {

    // přesná shoda
    if (t.includes(word)) {
      score += 3;
    }

    // částečná shoda
    if (
      word.length >= 4 &&
      t.includes(word.slice(0, 4))
    ) {
      score += 1;
    }

  }

  // bonus za celou frázi
  if (t.includes(q)) {
    score += 10;
  }

  return score;
}

function cleanContext(text) {
  return text
    .replace(/311XXX/g, "")
    .replace(/Systém integrované platební operace/g, "SIPO");
}

function getRelevantChunks(query, topK = 8) {
  return chunks
    .map(c => ({
      ...c,
      score: score(c.text, query)
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

// ---------------- PDF LOADER (PAGE BASED) ----------------
async function loadPDF(url) {

  const response = await axios.get(url, {
    responseType: "arraybuffer"
  });

  const data = new Uint8Array(response.data);

  const pdf = await pdfjsLib
    .getDocument({ data })
    .promise;

  const pages = [];

  for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {

    const page = await pdf.getPage(pageNum);

    const content = await page.getTextContent();

    const text = content.items
      .map(item => item.str)
      .join(" ");

    pages.push({
      text,
      page: pageNum
    });
  }

  return pages;
}

// ---------------- LOAD DOCS ----------------
async function loadDocs() {

  console.log("📄 Loading PDFs...");

  const cached = loadJson(PDF_CACHE_FILE);

  if (cached) {
    chunks = cached;
    console.log("📦 PDF CACHE LOADED:", chunks.length);
    return;
  }

  const tempChunks = [];

  for (const url of pdfUrls) {

    try {

      console.log("⬇️ Downloading:", url);

      const pages = await loadPDF(url);

      for (const pageData of pages) {

        tempChunks.push({
          text: pageData.text,
          source: url,
          page: pageData.page
        });

      }

      console.log("✅ PDF indexed:", url);

    } catch (err) {
      console.log("❌ PDF ERROR:", url);
    }
  }

  chunks = tempChunks;

  saveJson(PDF_CACHE_FILE, chunks);

  console.log("💾 PDF CACHE SAVED:", chunks.length);
}

// ---------------- GROQ ----------------
async function askAI(prompt) {
  const res = await axios.post(
    "https://api.groq.com/openai/v1/chat/completions",
    {
      model: "llama-3.1-8b-instant",
      messages: [
        {
          role: "system",
          content: `Jsi AI asistent pro vysvětlování dokumentace.

========================
ZÁKLADNÍ PRAVIDLA
========================

- odpovídej jasně a lidsky, ne citací bodů
- pokud najdeš relevantní sekci, vysvětli ji jako postup
- nikdy nevracej jen číslo bodu nebo název sekce
- ignoruj nadpisy jako odpověď

========================
FORMÁT ODPOVĚDI
========================

Pokud jde o postup:

1. krátké vysvětlení (max 1–2 věty)
2. kroky (max 5)
3. žádné opakování informací

Každý krok musí být na nový řádek a začínat číslem.

========================
ZÁKAZ OBSAHU
========================

Nikdy nezobrazuj:
- názvy sekcí
- názvy kapitol
- účetní účty (např. 311XXX)
- interní systémové kódy
- databázová ID
- IP adresy
- technické implementační detaily

Pokud se objeví v kontextu:
→ úplně je ignoruj
→ nevysvětluj je
→ nepoužívej je v odpovědi

========================
LOGIKA ODPOVĚDI
========================

- odpověď musí být pouze z poskytnutého kontextu
- pokud informace chybí:
  "Nenašel jsem to v dokumentaci."

========================
ZÁKAZ DUPLIKACE
========================

- nikdy neopakuj stejnou informaci
- žádné dvojité postupy
- žádné redundantní věty
`
        },
        { role: "user", content: prompt }
      ],
      temperature: 0.2
    },
    {
      headers: {
        Authorization: `Bearer ${GROQ_API_KEY}`,
        "Content-Type": "application/json"
      }
    }
  );

  return res.data.choices[0].message.content;
}

async function fetchPage(url) {
  const res = await fetch(url);
  const html = await res.text();

  const $ = cheerio.load(html);

  $("script, style, nav, footer, header, noscript").remove();

  const text = $("body").text().replace(/\s+/g, " ").trim();

  return text;
}

function extractLinks($) {

  const links = $("a")
    .map((i, el) => $(el).attr("href"))
    .get()
    .filter(Boolean)
    .map(href => {

      if (href.startsWith("http")) {
        return href;
      }

      return BASE_URL + href;
    });

  // PDF detekce
  links.forEach(link => {

    if (link.toLowerCase().includes(".pdf")) {

      discoveredPDFs.add(link);

      console.log("📄 PDF FOUND:", link);
    }

  });

  // jen interní odkazy
  return links.filter(link =>
    link.includes("help.datainfo.cz") ||
    link.includes(".pdf")
  );
}

function splitText(text, size = 800) {
  const chunks = [];

  for (let i = 0; i < text.length; i += size) {
    chunks.push(text.slice(i, i + size));
  }

  return chunks;
}

async function crawl(url) {
  if (visited.has(url)) return;
  if (visited.size >= MAX_PAGES) return;

  visited.add(url);

  try {
    const res = await fetch(url);
    const html = await res.text();

    const $ = cheerio.load(html);
         // PDF regex detection
    const pdfMatches = html.match(/https?:\/\/[^"' ]+\.pdf/gi) || [];

    pdfMatches.forEach(pdf => {

       discoveredPDFs.add(pdf);

       console.log("📄 PDF FOUND:", pdf);

    });

    $("script, style, nav, footer, header, noscript").remove();

    const text = $("body").text().replace(/\s+/g, " ").trim();

    const chunks = splitText(text, 800);

    for (const c of chunks) {
      webIndex.push({
        text: c,
        url
      });
    }

    const links = extractLinks($);

    for (const link of links.slice(0, 10)) {
      await crawl(link);
    }

  } catch (e) {
    console.log("crawl error:", url);
  }
}

async function loadWebKnowledge() {

  const cached = loadJson(WEB_CACHE_FILE);

  if (cached) {
    webIndex = cached;
    console.log("📦 WEB CACHE LOADED:", webIndex.length);
    return;
  }

  webIndex = [];
  visited.clear();

  await crawl(BASE_URL);

  saveJson(WEB_CACHE_FILE, webIndex);

  console.log("💾 WEB CACHE SAVED:", webIndex.length);
}

async function classifyIntent(message) {

  const res = await axios.post(
    "https://api.groq.com/openai/v1/chat/completions",
    {
      model: "llama-3.1-8b-instant",
      messages: [
        {
          role: "system",
          content: `
Rozhodni typ zprávy.

Možnosti:
- SMALLTALK
- DOCUMENT

SMALLTALK:
- pozdravy
- běžná konverzace
- poděkování
- krátké lidské reakce
- otázky typu "co ty"

DOCUMENT:
- otázky na dokumentaci
- návody
- postupy
- technické dotazy

Odpověz POUZE jedním slovem:
SMALLTALK
nebo
DOCUMENT
`
        },
        {
          role: "user",
          content: message
        }
      ],
      temperature: 0
    },
    {
      headers: {
        Authorization: `Bearer ${GROQ_API_KEY}`,
        "Content-Type": "application/json"
      }
    }
  );

  return res.data.choices[0].message.content.trim();
}

// ---------------- API ----------------
app.post("/api/chat", async (req, res) => {
  try {

    const message = req.body.message;
    const lower = message.toLowerCase();

    // krátké zprávy
    if (lower.length <= 1) {
      return res.json({
        answer: "Prosím napište konkrétní dotaz 🙂"
      });
    }

    // RAG SEARCH
    const relevant = getRelevantChunks(message, 8);

    console.log("===== TOP CHUNKS =====");

    relevant.forEach((r, i) => {
      console.log(i, "score:", r.score);
    });

    // FILTER (DŮLEŽITÉ)
    const filtered = relevant.filter(r => r.score >= 2);

    if (!filtered.length) {
      return res.json({
        answer: "Nenašel jsem to v dokumentaci."
      });
    }

    // LIMIT KONTEKSTU
    const context = cleanContext(
      filtered
        .slice(0, 5)
        .map(c => c.text)
        .join("\n\n")
    );

    // SYSTEM PROMPT (TVRDÝ RAG MODE)
    const answer = await askAI(`
Jsi RAG odpovídač nad dokumentací.

POUŽÍVEJ POUZE KONTEKST NÍŽE.
NIC NEVYMÝŠLEJ.

====================
PRAVIDLA
====================

- odpověď musí být pouze z poskytnutého textu
- nesmíš přidávat vlastní znalosti
- nesmíš domýšlet kroky
- pokud to není v textu → řekni: "Nenašel jsem to v dokumentaci."

====================
FORMÁT
====================

Pokud je postup:
- max 5 kroků
- každý krok na nový řádek
- bez opakování

====================
KONTEKST:
${context}

DOTAZ:
${message}
    `);

    return res.json({ answer });

  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: err.message });
  }
});

// ---------------- START ----------------
async function startServer() {

  const PORT = process.env.PORT || 3000;

  app.listen(PORT, () => {
    console.log("Server running on port", PORT);
  });

  try {

    console.log("Loading transformers...");
    await loadEmbedder(); // ← DŮLEŽITÉ

    console.log("Loading embedder model...");

    embedder = await pipeline(
      "feature-extraction",
      "Xenova/all-MiniLM-L6-v2"
    );

    console.log("Loading web knowledge...");
    await loadWebKnowledge();

    console.log("Loading PDFs...");
    await loadDocs();

    console.log("AI READY");

  } catch (err) {
    console.error(err);
  }
}

startServer();