const express = require("express");
const axios = require("axios");
require("dotenv").config();
const cheerio = require("cheerio");
const fs = require("fs");
const pdfParse = require("pdf-parse");


const knownPDFs = [
  "http://erp.oznameni.datainfo.cz//wp-content/uploads/2024/09/Datainfo-Jak-na-zalohy.pdf",
  "http://erp.oznameni.datainfo.cz//wp-content/uploads/2024/09/Datainfo-Jak-na-vodne-a-stocne.pdf"
];
const CHAT_LOG_FILE = "./chatLogs.json";
const WEB_CACHE_FILE = "./webIndex.json";
const PDF_CACHE_FILE = "./pdfIndex.json";
const BASE_URL = "https://help.datainfo.cz";
const MAX_PAGES = 25;
let discoveredPDFs = new Set();
let pipeline;

let webIndex = [];
let visited = new Set();

let docsLoaded = false;

let chunks = [];
let embedder;
let embeddedChunks = [];
const queryEmbeddingCache = {};
const queryCacheTTL = 1000 * 60 * 60; // 1 hodina
const embeddingCache = {};

let aiReady = false;
let webReady = false;
let pdfReady = false;

let aiBootPhase = "starting";

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

function saveChatLog(data) {

  let logs = [];

  try {
    logs = JSON.parse(
      fs.readFileSync(CHAT_LOG_FILE, "utf8")
    );
  } catch {}

  logs.push(data);

  fs.writeFileSync(
    CHAT_LOG_FILE,
    JSON.stringify(logs, null, 2)
  );
}

const app = express();
app.use(express.json());
app.use(express.static("public"));

const GROQ_API_KEY = process.env.GROQ_API_KEY;

function normalizeText(text) {
  return text
    .toLowerCase()
    .trim()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "");
}

// ---------------- EMBEDDING ----------------
async function embed(text) {
  const key = normalizeText(text);

  if (embeddingCache[key]) {
    return embeddingCache[key];
  }

  const out = await embedder(text, {
    pooling: "mean",
    normalize: true
  });

  const vec = Array.from(out.data);
  embeddingCache[key] = vec;

  return vec;
}

function cosineSim(a, b) {
  let dot = 0;
  let magA = 0;
  let magB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }

  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

async function buildEmbeddings() {
  console.log("🧠 Building embeddings...");

  if (!chunks || chunks.length === 0) {
    console.log("⚠️ No chunks found, skipping embeddings");
    return;
  }

  const embedded = [];

  for (const chunk of chunks) {
    try {
      const vector = await embed(chunk.text);

      embedded.push({
        ...chunk,
        embedding: vector
      });

    } catch (err) {
      console.log("❌ Embedding failed for chunk");
    }
  }

  embeddedChunks = embedded;
  chunks = embedded;

  console.log("✅ Embeddings ready:", chunks.length);

  saveJson(PDF_CACHE_FILE, {
    timestamp: Date.now(),
    chunks
  });

  console.log("💾 Embeddings saved to cache");
}

async function getRelevantChunks(query, topK = 5) {

  const key = normalizeText(query);
  const now = Date.now();

  // 1) CACHE HIT
  if (
    queryEmbeddingCache[key] &&
    (now - queryEmbeddingCache[key].timestamp < queryCacheTTL)
  ) {
    const qVec = queryEmbeddingCache[key].vector;

    return embeddedChunks
      .map(c => ({
        ...c,
        score: cosine(qVec, c.embedding)
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  // 2) NOVÝ EMBEDDING
  try {
    const qVec = await embed(query);

    // uložit do cache
    queryEmbeddingCache[key] = {
      vector: qVec,
      timestamp: now
    };

    return embeddedChunks
      .map(c => ({
        ...c,
        score: cosine(qVec, c.embedding)
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);

  } catch (err) {
    console.log("RAG error:", err.message);
    return [];
  }
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

function splitText(text, size = 800, overlap = 150) {
  const chunks = [];

  for (let i = 0; i < text.length; i += size - overlap) {
    chunks.push(text.slice(i, i + size));
  }

  return chunks;
}

function cleanContext(text) {
  return text
    .replace(/311XXX/g, "")
    .replace(/Systém integrované platební operace/g, "SIPO");
}

// ---------------- PDF LOADER (PAGE BASED) ----------------
async function loadPDF(url) {
  try {
    console.log("⬇️ DOWNLOADING:", url);

    const response = await axios.get(url, {
      responseType: "arraybuffer",
      timeout: 30000,
      headers: {
        "User-Agent": "Mozilla/5.0"
      }
    });

    console.log("📦 DOWNLOAD SIZE:", response.data.length);

    const buffer = Buffer.from(response.data);

    const pdfParse = require("pdf-parse");

    const data = await pdfParse(buffer);

    console.log("📄 PDF TEXT LENGTH:", data.text?.length);

    return [{
      text: data.text,
      page: 1
    }];

  } catch (err) {
    console.log("❌ PDF LOAD FAILED:", url);

    console.log("FULL ERROR:");
    console.log(err); // <- TOTO JE KLÍČ

    return [];
  }
}

// ---------------- LOAD DOCS ----------------
async function loadDocs() {
  if (docsLoaded) {
    console.log("⚠️ loadDocs already executed, skipping");
    return;
  }

  docsLoaded = true;
  console.log("📄 Loading PDFs...");
  
  const cached = loadJson(PDF_CACHE_FILE);
  
  if (cached?.chunks?.length > 0) {
     chunks = cached.chunks;

     // DŮLEŽITÉ:
     embeddedChunks = cached.chunks;

     console.log("📦 PDF CACHE LOADED:", chunks.length);

     return true; // <- přidej return TRUE
  }

  const pdfList = [
    ...new Set([
      ...knownPDFs,
      ...Array.from(discoveredPDFs)
    ])
  ];

  console.log("📚 PDFs found:", pdfList.length);

  const tempChunks = [];

  for (const url of pdfList) {
    try {
      console.log("⬇️ PDF:", url);

      const pages = await loadPDF(url);

      for (const p of pages) {

        const parts = splitText(p.text, 500);

        for (const c of parts) {
          tempChunks.push({
            text: c,
            source: url,
            page: p.page
          });
        }
      }

    } catch (e) {
      console.log("❌ PDF error:", url);
    }
  }

  chunks = tempChunks;

  console.log("💾 PDF SAVED:", chunks.length);
  console.log("PDF LIST:", pdfList);
}

// ---------------- GROQ ----------------
async function askAI(message, context) {

  const systemPrompt = `
Jsi AI asistent pro firemní dokumentaci.

PRAVIDLA:
- odpovídej POUZE z kontextu
- pokud je v kontextu odpověď, MUSÍŠ ji použít
- pokud je odpověď částečně v kontextu, slož ji z něj
- nikdy neříkej, že informace není v dokumentaci, pokud je v kontextu
- pokud kontext neobsahuje žádnou relevantní informaci, odpověz: "V dokumentaci to není uvedeno"
- NEvymýšlej žádné kroky ani menu
- pokud kontext obsahuje postup, kopíruj ho co nejpřesněji
- max 6 vět
- postupy piš jen v bodech
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
    const res = await axios.get(url, {
       timeout: 15000,
       headers: { "User-Agent": "Mozilla/5.0" }
    });

    const html = res.data;

    const $ = cheerio.load(html);
         // PDF regex detection
    const pdfMatches = html.match(/https?:\/\/[^"' ]+\.pdf/gi) || [];

    pdfMatches.forEach(pdf => {

       discoveredPDFs.add(pdf);

       console.log("📄 PDF FOUND:", pdf);

    });

    $("script, style, nav, footer, header, noscript").remove();

    const text = $("body").text().replace(/\s+/g, " ").trim();

    const chunks = splitText(text, 200);

    for (const c of chunks) {
       const vector = await embed(c);

       webIndex.push({
          text: c,
          url,
          embedding: vector
       });
    }

    const links = extractLinks($);

    for (const link of links.slice(0, 10)) {
      await crawl(link);
    }

  } catch (e) {
       console.log("❌ CRAWL FAILED:", url);
       console.log("❌ ERROR NAME:", e.name);
       console.log("❌ ERROR MESSAGE:", e.message);
       console.log("❌ FULL:", e);
  }
}

async function loadWebKnowledge() {

  const cached = loadJson(WEB_CACHE_FILE);

  // cache 12 hodin
  if (
      cached &&
      cached.pages &&
      Array.isArray(cached.pages) &&
      cached.pages.length > 0 &&
      cached.timestamp &&
      Date.now() - cached.timestamp < 1000 * 60 * 60 * 12
  ) {

     webIndex = cached.pages;
     embeddedChunks = cached.pages.map(p => ({
        ...p,
        embedding: p.embedding || []
     }));

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
  webReady = true;
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

app.get("/health", (req, res) => {
  res.status(200).send("ok");
});

// ---------------- API ----------------
app.post("/api/chat", async (req, res) => {
  try {

    const message = req.body.message;

    let ragTimeout = false;

    if (!aiReady) {
        const intent = classifyIntent(message);

        // SMALLTALK funguje hned
        if (intent.intent === "smalltalk") {
           const answer = await askSmalltalkAI(message);
           return res.json({ answer });
        }

        // rychlá fallback odpověď
        return res.json({
           answer: "⚡ Systém se načítá, ale už můžeš psát — odpovím hned jak bude AI plně ready."
        });
    }

    const intent = classifyIntent(message);

    console.log("INTENT:", intent.intent);

    // ======================
    // 1) SMALLTALK (NO AI)
    // ======================
    if (intent.intent === "smalltalk") {

  const answer = await askSmalltalkAI(message);
  
  saveChatLog({
     time: new Date().toLocaleString("cs-CZ", {
        timeZone: "Europe/Prague"
     }),
     ip: req.ip,
     question: message,
     answer
  });
  
  return res.json({
    answer
  });
}

    // ======================
    // 2) RAG (AI + DOCS)
    // ======================
    if (intent.intent === "rag") {

  if (ragTimeout) {
     const answer = await askAI(message, null);
     return res.json({ answer });
  }
  
  ragTimeout = true;
  setTimeout(() => ragTimeout = false, 2000);

  let relevant = await getRelevantChunks(message, 5);
  
  if (!relevant.length) {
     const answer = await askAI(message, null);
     return res.json({ answer });
  }

  console.log("RELEVANT:", relevant.map(r => r.score));

  const context = relevant
    .filter(r => r.score > 0.25)
    .map(r => r.text)
    .join("\n\n");

  console.log("CONTEXT LENGTH:", context.length);

  const answer = await askAI(message, context || null);

  saveChatLog({
     time: new Date().toLocaleString("cs-CZ", {
        timeZone: "Europe/Prague"
     }),
     ip: req.ip,
     question: message,
     answer
  });

  return res.json({ answer });
}

    // ======================
    // 3) FALLBACK (AI ONLY)
    // ======================
    if (intent.intent === "fallback") {

      const answer = await askAI(message, null);

      saveChatLog({
         time: new Date().toLocaleString("cs-CZ", {
            timeZone: "Europe/Prague"
         }),
         ip: req.ip,
         question: message,
         answer
      });

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

  console.log("⚡ Server started instantly");

  setTimeout(() => {
      initAI();
  }, 0);
}

async function initAI() {
  try {
    console.log("🧠 Loading AI in background...");
    aiBootPhase = "loading_embedder";

    await loadEmbedder();

    embedder = await pipeline(
      "feature-extraction",
      "Xenova/all-MiniLM-L6-v2"
    );

    aiBootPhase = "loading_web";
    await loadWebKnowledge();

    aiBootPhase = "loading_pdf";
    await loadDocs();

    aiBootPhase = "building_embeddings";
    await buildEmbeddings();

    aiReady = true;
    aiBootPhase = "ready";

    console.log("✅ AI READY");
  } catch (err) {
    console.error("❌ AI init failed:", err);
    aiBootPhase = "error";
  }
}

startServer();