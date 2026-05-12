const express = require("express");
const axios = require("axios");
const pdfParse = require("pdf-parse");
const { pipeline } = require("@xenova/transformers");

let embedder;

require("dotenv").config();
const app = express();
app.use(express.json());
app.use(express.static("public"));

const GROQ_API_KEY = process.env.GROQ_API_KEY;

const PDFS = [
  "https://erp.oznameni.datainfo.cz/wp-content/uploads/2024/09/Datainfo-Jak-na-zalohy.pdf",
  "https://erp.oznameni.datainfo.cz/wp-content/uploads/2024/09/Datainfo-Jak-na-vodne-a-stocne.pdf"
];

let chunks = [];

// jednoduché dělení textu
function splitText(text, size = 1000) {
  const result = [];
  for (let i = 0; i < text.length; i += size) {
    result.push(text.slice(i, i + size));
  }
  return result;
}

function cosine(a, b) {
  let dot = 0, magA = 0, magB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }

  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

async function downloadPDF(url, retries = 3) {
  for (let i = 0; i < retries; i++) {
    try {
      return await axios.get(url, {
        responseType: "arraybuffer",
        timeout: 30000
      });
    } catch (err) {
      console.log(`Retry ${i + 1} failed`);
      if (i === retries - 1) throw err;
    }
  }
}

// načtení PDF
async function loadDocs() {

  const files = [
    "./docs/Datainfo-Jak-na-zalohy.pdf",
    "./docs/Datainfo-Jak-na-vodne-a-stocne.pdf"
  ];

  for (const filePath of files) {

    const dataBuffer = fs.readFileSync(filePath);
    const data = await pdfParse(dataBuffer);

    const splitChunks = splitText(data.text, 400);

    for (const chunk of splitChunks) {
      chunks.push({
        text: chunk,
        source: filePath
      });
    }
  }

  console.log("📄 PDF načteny z lokálních souborů");
}

// jednoduché skórování relevance
async function getRelevantChunks(query, top = 4) {

  const queryVector = await embed(query);

  return chunks
    .map(c => ({
      ...c,
      score: cosine(queryVector, c.embedding)
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, top);
}

async function embed(text) {
  const output = await embedder(text, {
    pooling: "mean",
    normalize: true
  });

  return Array.from(output.data);
}

// volání Groq API
async function askAI(prompt) {

  try {

    const response = await axios.post(
      "https://api.groq.com/openai/v1/chat/completions",
      {
        model: "llama-3.1-8b-instant",
        messages: [
          {
            role: "system",
            content: `
Odpovídej výhradně z poskytnutého kontextu.
Pokud v kontextu není odpověď, řekni: "V dokumentech jsem to nenašel."
Odpovědi formuluj stručně a cituj jen informace z textu.
`
          },
          {
            role: "user",
            content: prompt
          }
        ],
        temperature: 0.2
      },
      {
        headers: {
          "Authorization": `Bearer ${GROQ_API_KEY}`,
          "Content-Type": "application/json"
        }
      }
    );

    return response.data.choices[0].message.content;

  } catch (err) {

    console.log("GROQ ERROR:");

    if (err.response) {
      console.log(err.response.data);
    } else {
      console.log(err.message);
    }

    return "Chyba AI odpovědi.";
  }
}

app.post("/api/chat", async (req, res) => {
  const { message } = req.body;

  const relevant = await getRelevantChunks(message, 4);

  const context = relevant
    .map(r => r.text)
    .join("\n\n");

  const prompt = `
DOKUMENTACE:
${context}

DOTAZ:
${message}
`;

  const answer = await askAI(prompt);

  const sources = [...new Set(relevant.map(r => r.source))];

  res.json({
    answer,
    sources
  });
});
app.listen(process.env.PORT || 3000, () => {
  console.log("Server běží");

  loadDocs().catch(err => {
    console.error("PDF load failed:", err.message);
  });
});