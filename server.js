const express = require("express");
const axios = require("axios");
const pdfjsLib = require("pdfjs-dist/legacy/build/pdf.js");
require("dotenv").config();
const cheerio = require("cheerio");
const fs = require("fs");

const knownPDFs = [
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

function normalizeText(text) {

  return text
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "");
}

// ---------------- EMBEDDING ----------------
async function embed(text) {
  const output = await embedder(text, {
    pooling: "mean",
    normalizeText: true
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

function score(text, query) {

  const t = normalizeText(text);
  const q = normalizeText(query);

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
    score += 20;
  }

  return score;
}

function cleanContext(text) {
  return text
    .replace(/311XXX/g, "")
    .replace(/Systém integrované platební operace/g, "SIPO");
}

function getRelevantChunks(query, topK = 8) {

  const all = [
    ...chunks,
    ...webIndex
  ];

  return all
    .map(c => ({
      ...c,
      score: score(c.text, query)
    }))
    .filter(c => c.score > 0)
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

  // cache max 24h
  if (
    cached &&
    cached.timestamp &&
    Date.now() - cached.timestamp < 1000 * 60 * 60 * 24
  ) {

    chunks = cached.chunks;

    console.log("📦 PDF CACHE LOADED:", chunks.length);

    return;
  }

  const tempChunks = [];

  const pdfList = [
     ...new Set([
       ...knownPDFs,
       ...Array.from(discoveredPDFs)
    ])
  ];

  console.log("📚 PDFs found:", pdfList.length);

  for (const url of pdfList) {

    try {

      console.log("⬇️ Downloading PDF:", url);

      const pages = await loadPDF(url);

      for (const pageData of pages) {

        const smallerChunks = splitText(pageData.text, 500);

        for (const c of smallerChunks) {

          tempChunks.push({
            text: c,
            source: url,
            page: pageData.page
          });

        }

      }

      console.log("✅ PDF indexed:", url);

    } catch (err) {

      console.log("❌ PDF ERROR:", url);

    }
  }

  chunks = tempChunks;

  saveJson(PDF_CACHE_FILE, {
    timestamp: Date.now(),
    chunks
  });

  console.log("💾 PDF CACHE SAVED:", chunks.length);
}

// ---------------- GROQ ----------------
async function askAI(message, context) {

  const systemPrompt = `
Jsi AI asistent pro firemní dokumentaci.

PRAVIDLA:
- odpovídej stručně a přirozeně
- pokud máš kontext, použij ho
- pokud nemáš kontext, odpovídej obecně
- nikdy nepiš "nemám kontext"
- max 6 vět
- postupy piš jenom v bodech
- každý bod bude očíslovaný
`;

  const userContent = context
    ? `KONTEKST:\n${context}\n\nDOTAZ:\n${message}`
    : `DOTAZ:\n${message}`;

  try {
    const res = await axios.post(
      "https://api.groq.com/openai/v1/chat/completions",
      {
        model: "llama-3.1-8b-instant",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userContent }
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

  } catch (err) {

    if (err.response?.status === 429) {
      return "Server je přetížený, zkus to za chvíli 🙂";
    }

    throw err;
  }
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

  // cache 12 hodin
  if (
    cached &&
    cached.timestamp &&
    Date.now() - cached.timestamp < 1000 * 60 * 60 * 12
  ) {

    webIndex = cached.pages;

    console.log("📦 WEB CACHE LOADED:", webIndex.length);

    return;
  }

  console.log("🌐 Refreshing web knowledge...");

  webIndex = [];
  visited.clear();

  await crawl(BASE_URL);

  saveJson(WEB_CACHE_FILE, {
    timestamp: Date.now(),
    pages: webIndex
  });

  console.log("💾 WEB CACHE SAVED:", webIndex.length);
}

function classifyIntent(message) {

  const lower = normalizeText(message);

  // ======================
  // SMALLTALK
  // ======================

  const smalltalk = [
  "ahoj",
  "cau",
  "cus",
  "ok",
  "okay",
  "diky",
  "dekuji",
  "jak se mas",
  "co ty",
  "a ty",
  "aha",
  "super",
  "fajn",
  "jasne",
  "jo"
];

  if (smalltalk.some(w => lower.includes(w))) {
    return {
      intent: "smalltalk"
    };
  }

  // ======================
  // RAG DOTAZY
  // ======================

  const ragWords = [
  "jak",
  "kde",
  "co",
  "nastavit",
  "vytvorit",
  "pridat",
  "sipo",
  "faktura",
  "zaloh",
  "odberatel",
  "vodomer",
  "platba",
  "odecet",
  "aplikace"
];

  if (ragWords.some(w => lower.includes(w))) {
    return {
      intent: "rag"
    };
  }

  // ======================
  // FALLBACK AI
  // ======================

  return {
    intent: "fallback"
  };
}

async function askSmalltalkAI(message) {

  const response = await axios.post(
    "https://api.groq.com/openai/v1/chat/completions",
    {
      model: "llama-3.1-8b-instant",
      max_tokens: 40,
      temperature: 0.7,

      messages: [
        {
          role: "system",
          content: `
Jsi přátelský chatbot.

Pravidla:
- odpovídej krátce
- maximálně 1 věta
- buď přirozený
- můžeš být lehce vtipný
- odpovídej česky
- NEpiš dlouhé odpovědi
`
        },
        {
          role: "user",
          content: message
        }
      ]
    },
    {
      headers: {
        Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
        "Content-Type": "application/json"
      }
    }
  );

  return response.data.choices[0].message.content;
}

// ---------------- API ----------------
app.post("/api/chat", async (req, res) => {
  try {

    const message = req.body.message;
    const intent = classifyIntent(message);

    console.log("INTENT:", intent.intent);

    // ======================
    // 1) SMALLTALK (NO AI)
    // ======================
    if (intent.intent === "smalltalk") {

  const answer = await askSmalltalkAI(message);

  return res.json({
    answer
  });
}

    // ======================
    // 2) RAG (AI + DOCS)
    // ======================
    if (intent.intent === "rag") {

      const relevant = getRelevantChunks(message, 8) || [];

      const context = relevant.length
         ? relevant
            .slice(0, 3)
            .map(r => r.text.slice(0, 500))
            .join("\n\n")
         : "";

      const answer = await askAI(message, context);

      return res.json({ answer });
    }

    // ======================
    // 3) FALLBACK (AI ONLY)
    // ======================
    if (intent.intent === "fallback") {

      const answer = await askAI(message, null);

      return res.json({ answer });
    }

  } catch (err) {
    console.error(err);

    return res.json({
      answer: "Nastala chyba při komunikaci se serverem."
    });
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