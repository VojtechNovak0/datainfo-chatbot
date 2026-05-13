const express = require("express");
const axios = require("axios");
const pdfjsLib = require("pdfjs-dist/legacy/build/pdf.js");
const fs = require("fs");
require("dotenv").config();
const { pipeline } = require("@xenova/transformers");
const cheerio = require("cheerio");

const BASE_URL = "https://help.datainfo.cz";
const MAX_PAGES = 30;

let webIndex = [];
let visited = new Set();

const app = express();
app.use(express.json());
app.use(express.static("public"));

const GROQ_API_KEY = process.env.GROQ_API_KEY;

const files = [
  "./docs/Datainfo-Jak-na-zalohy.pdf",
  "./docs/Datainfo-Jak-na-vodne-a-stocne.pdf"
];

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

async function getRelevantChunks(query, topK = 8) {
  const queryVector = await embed(query);

  const scored = chunks.map(c => ({
    text: c.text,
    page: c.page,
    source: c.source,
    vector: c.vector,
    score: cosine(queryVector, c.vector)
  }));

  return scored
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
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

// ---------------- PDF LOADER (PAGE BASED) ----------------
async function loadPDF(filePath) {

  const data = new Uint8Array(
    fs.readFileSync(filePath)
  );

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

  for (const file of files) {

    const pages = await loadPDF(file);

    for (const pageData of pages) {

      const vector = await embed(pageData.text);

      chunks.push({
        text: pageData.text,
        source: file,
        page: pageData.page,
        vector
      });

    }

  }

  console.log("PDF načteny:", chunks.length);

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

TVÉ ÚKOLY:
- odpovídej vysvětlením, ne citací bodů
- nikdy nevracej jen číslo bodu (např. "bod 11")
- pokud najdeš relevantní sekci, ROZVEĎ ji do kroků
- spoj více vět z kontextu do smysluplného postupu
- odpověď má být 3–6 vět
- pokud je to postup, napiš ho jako kroky
- můžeš použít více částí textu z různých chunků
- ignoruj nadpisy jako hlavní odpověď
POKUD ODPOVĚĎ OBSAHUJE POSTUP:
- piš ho jako očíslované kroky
- každý krok na nový řádek
- začínej číslem (1., 2., 3.)
- nepoužívej dlouhé odstavce pro postupy

NEVRACEJ:
- "nachází se v bodě X"
- "bod X říká X"

MÍSTO TOHO:
- vysvětli co má uživatel udělat
- popiš postup
- dej kontext

Pokud odpověď není v dokumentaci:
"Nenašel jsem to v dokumentaci."
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
  return $("a")
    .map((i, el) => $(el).attr("href"))
    .get()
    .filter(Boolean)
    .map(href => href.startsWith("http") ? href : BASE_URL + href)
    .filter(url => url.includes("help.datainfo.cz"));
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
  webIndex = [];
  visited.clear();

  await crawl(BASE_URL);

  console.log("WEB INDEX READY:", webIndex.length);
}

// ---------------- API ----------------
app.post("/api/chat", async (req, res) => {
  try {

    const message = req.body.message;
    const lower = message.toLowerCase();

    // 1. krátké zprávy
    if (lower.length <= 1) {
      return res.json({
        answer: "Prosím napište konkrétní dotaz 🙂",
        sources: []
      });
    }

    // 2. RAG (TOTO JE HLAVNÍ ČÁST)
    const relevant = await getRelevantChunks(message, 8);

    console.log("===== TOP CHUNKS =====");

    relevant.forEach((r, i) => {
      console.log(i, "score:", r.score);
      console.log(r.text.slice(0, 150));
    });

    // 3. fallback
    if (!relevant.length || relevant[0].score < 0.25) {
      return res.json({
        answer: "V dokumentaci jsem k tomuto dotazu nenašel přesné informace.",
        sources: []
      });
    }

    // 4. context
    const context = relevant
      .map(r => `${r.text} (strana ${r.page})`)
      .join("\n\n");

    // 5. AI
    const answer = await askAI(`
Použij pouze tento kontext:

${context}

Dotaz: ${message}
`);

    return res.json({ answer });

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// ---------------- START ----------------
async function startServer() {

  embedder = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2"
  );

  const PORT = process.env.PORT || 3000;
  
  (async () => {
     await loadDocs();
     await loadWebKnowledge();

     app.listen(PORT, () => {
        console.log("Server ready");
     });
  })();
}

startServer();