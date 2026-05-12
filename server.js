const express = require("express");
const axios = require("axios");
const pdfParse = require("pdf-parse");

require("dotenv").config();
const app = express();
app.use(express.json());
app.use(express.static("public"));

const GROQ_API_KEY = process.env.GROQ_API_KEY;

const PDFS = [
  "http://erp.oznameni.datainfo.cz/wp-content/uploads/2024/09/Datainfo-Jak-na-zalohy.pdf",
  "http://erp.oznameni.datainfo.cz/wp-content/uploads/2024/09/Datainfo-Jak-na-vodne-a-stocne.pdf"
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

// načtení PDF
async function loadDocs() {

  for (const url of PDFS) {

    const res = await axios.get(url, {
      responseType: "arraybuffer"
    });

    const data = await pdfParse(res.data);

    const splitChunks = splitText(data.text);

    for (const chunk of splitChunks) {

      chunks.push({
        text: chunk,
        source: url
      });

    }

  }

  console.log("📄 Dokumenty načteny");
}

// jednoduché skórování relevance
function getRelevantChunks(query, top = 4) {

  const words = query.toLowerCase().split(" ");

  return chunks
    .map(item => {

      let score = 0;

      for (const word of words) {

        if (item.text.toLowerCase().includes(word)) {
          score++;
        }

      }

      return {
        text: item.text,
        source: item.source,
        score
      };

    })
    .sort((a, b) => b.score - a.score)
    .slice(0, top);

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
            content:
              "Odpovídej pouze z dokumentace. Pokud odpověď neznáš, napiš: Tohle v dokumentaci nemám."
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

  const relevant = getRelevantChunks(message, 4);

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

  const sources = [...new Set(
    relevant.map(r => r.source)
  )];

  res.json({
     answer,
     sources
  });
});
app.listen(3000, async () => {
  await loadDocs();
  console.log("Server běží na http://localhost:3000");
});