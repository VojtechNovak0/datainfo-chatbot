const express = require("express");
const axios = require("axios");
const pdfjsLib = require("pdfjs-dist/legacy/build/pdf.js");
const fs = require("fs");
require("dotenv").config();

const { pipeline } = require("@xenova/transformers");

const app = express();
app.use(express.json());
app.use(express.static("public"));

const GROQ_API_KEY = process.env.GROQ_API_KEY;

const PDFS = [
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

// ---------------- PDF LOADER (PAGE BASED) ----------------
async function loadPDF(filePath) {
  const data = new Uint8Array(fs.readFileSync(filePath));
  const pdf = await pdfjsLib.getDocument(data).promise;

  const pages = [];

  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();

    const text = content.items.map(i => i.str).join(" ");

    pages.push({
      page: i,
      text,
      source: filePath
    });
  }

  return pages;
}

// ---------------- LOAD DOCS ----------------
async function loadDocs() {
  chunks = [];

  for (const file of PDFS) {
    console.log("Loading:", file);

    const pages = await loadPDF(file);

    for (const p of pages) {
      const blocks = splitText(p.text);

     for (const block of blocks) {

       if (!block.trim()) continue;

       const vector = await embed(block);

       chunks.push({
          text: block,
          page: p.page,
          source: file,
          embedding: vector
       });
     }
    }
  }

  console.log("TOTAL CHUNKS:", chunks.length);
}

// ---------------- SEARCH ----------------
async function getRelevantChunks(query, top = 3) {
  const queryVec = await embed(query);

  return chunks
    .map(c => ({
      ...c,
      score: cosine(queryVec, c.embedding)
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, top)
    .filter(c => c.score > 0.25);
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
          content: `
Jsi chatbot odpovídající pouze pomocí citací z dokumentace.

Pravidla:
- odpovídej POUZE přesnými větami přímo z PDF souborů
- nic nepřepisuj vlastními slovy
- nic neparafrázuj
- necituj informace, které nejsou v kontextu
- vždy uveď číslo strany

Formát odpovědi:

"CITACE Z DOKUMENTU" (strana X)

Pokud odpověď v kontextu není:
"V dokumentech jsem to nenašel."
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

// ---------------- API ----------------
app.post("/api/chat", async (req, res) => {
  try {

    console.log("➡️ NEW REQUEST:", req.body.message);

    const message = req.body.message;
    
    const lower = message.toLowerCase();

    if (
       lower.includes("děkuji") ||
       lower.includes("díky")
    ) {
       return res.json({
          answer: "Rádo se stalo 🙂",
          sources: []
       });
    }

    if (
       lower.includes("super") ||
       lower.includes("perfektní") ||
       lower.includes("skvělé")
    ) {
       return res.json({
          answer: "Jsem rád, že to pomohlo 🙂",
          sources: []
       });
    }

    if (
       lower.includes("ahoj") ||
       lower.includes("čau") ||
       lower.includes("čus")
    ) {
       return res.json({
          answer: "Ahoj 👋",
          sources: []
       });
    }
    
    const relevant = await getRelevantChunks(message, 3);

    const context = relevant
      .map(r => `${r.text} (strana ${r.page})`)
      .join("\n\n");

    const answer = await askAI(`
DOKUMENTACE:
${context}

DOTAZ:
${message}
`);

    res.json({ answer });

  } catch (err) {
    console.error("❌ BACKEND ERROR:", err);
    console.error(err.stack);

    res.status(500).json({
      error: err.message
    });
  }
});

// ---------------- START ----------------
async function startServer() {

  embedder = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2"
  );

  const PORT = process.env.PORT || 3000;
  
  app.listen(PORT, () => {
    console.log(`Server běží na portu ${PORT}`);
    await loadDocs(); // 🔥 nejdřív data
    console.log("📄 PDF načteny");
  });
}

startServer();