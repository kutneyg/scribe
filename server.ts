import express from "express";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { WebSocketServer, WebSocket } from "ws";
import { GoogleGenAI, Modality } from "@google/genai";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = Number(process.env.PORT) || 3000;

app.use(express.json({ limit: "30mb" }));

function createGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error(
      "GEMINI_API_KEY is not configured. Please add your API key in Settings > Secrets."
    );
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

function formatApiError(err: unknown): { status: number; message: string } {
  const rawMessage =
    err instanceof Error ? err.message : "Unexpected transcription error.";
  if (rawMessage.includes("API_KEY_INVALID") || rawMessage.includes("400")) {
    return {
      status: 400,
      message:
        "Invalid Gemini API key. Please verify your key in Settings > Secrets.",
    };
  }
  if (rawMessage.includes("PERMISSION_DENIED") || rawMessage.includes("403")) {
    return {
      status: 403,
      message:
        "Gemini API permission denied. Please check your API key in Settings > Secrets.",
    };
  }
  if (rawMessage.includes("RESOURCE_EXHAUSTED") || rawMessage.includes("429")) {
    return {
      status: 429,
      message:
        "Gemini API rate limit or quota exceeded. Please try again shortly or check your plan in Settings > Secrets.",
    };
  }
  return { status: 500, message: rawMessage };
}

app.get("/", (_req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

const ALLOWED_AUDIO_MIME_PREFIXES = [
  "audio/",
  "video/webm",
  "video/mp4",
];

app.post("/api/transcribe", async (req, res) => {
  try {
    const { audio, mimeType, context } = req.body ?? {};

    if (typeof audio !== "string" || audio.trim().length === 0) {
      res.status(400).json({ error: "Missing audio payload." });
      return;
    }

    const normalizedMime =
      typeof mimeType === "string" && mimeType.trim().length > 0
        ? mimeType.split(";")[0].trim().toLowerCase()
        : "audio/webm";

    const isAllowedMime = ALLOWED_AUDIO_MIME_PREFIXES.some((prefix) =>
      normalizedMime.startsWith(prefix)
    );
    if (!isAllowedMime) {
      res.status(400).json({ error: `Unsupported audio MIME type: ${normalizedMime}` });
      return;
    }

    const ai = createGeminiClient();

    const contextHint =
      context === "live"
        ? "This is a career advising dialogue between an advisor and a student."
        : "This is a post-appointment career advising dictation by an advisor.";

    const response = await ai.models.generateContent({
      model: "gemini-3.5-transcribe",
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: normalizedMime,
              data: audio,
            },
          },
          {
            text: `Transcribe this audio verbatim with accurate punctuation, capitalization, and natural sentence spacing. ${contextHint} Return only the transcript text without extra commentary.`,
          },
        ],
      },
    });

    const transcript = (response.text ?? "").trim();
    res.json({ transcript });
  } catch (err) {
    console.error("Audio transcription error:", err);
    const { status, message } = formatApiError(err);
    res.status(status).json({ error: message });
  }
});

const server = http.createServer(app);
const wss = new WebSocketServer({ server, path: "/ws/transcribe" });

wss.on("connection", async (clientWs) => {
  let ai: GoogleGenAI;
  try {
    ai = createGeminiClient();
  } catch (err) {
    if (clientWs.readyState === WebSocket.OPEN) {
      clientWs.send(
        JSON.stringify({
          type: "error",
          message: err instanceof Error ? err.message : "Missing API key.",
        })
      );
      clientWs.close();
    }
    return;
  }

  let sessionReady = false;
  let closing = false;
  const pendingAudio: string[] = [];
  let session: Awaited<ReturnType<GoogleGenAI["live"]["connect"]>> | null = null;
  let stopTimer: ReturnType<typeof setTimeout> | null = null;

  const closeSession = () => {
    if (closing) return;
    closing = true;
    if (stopTimer) {
      clearTimeout(stopTimer);
      stopTimer = null;
    }
    try {
      session?.close();
    } catch {
      // Ignore close errors if already closed
    }
  };

  try {
    session = await ai.live.connect({
      model: "gemini-3.5-transcribe-live",
      config: {
        responseModalities: [Modality.TEXT],
        inputAudioTranscription: {
          languageCodes: [],
        },
      },
      callbacks: {
        onopen: () => {
          sessionReady = true;
          if (clientWs.readyState === WebSocket.OPEN) {
            clientWs.send(JSON.stringify({ type: "ready" }));
          }
          while (pendingAudio.length > 0 && !closing) {
            const chunk = pendingAudio.shift();
            if (chunk && session) {
              session.sendRealtimeInput({
                audio: { data: chunk, mimeType: "audio/pcm;rate=16000" },
              });
            }
          }
        },
        onmessage: (message) => {
          if (clientWs.readyState !== WebSocket.OPEN) return;
          const content = message.serverContent;

          if (content?.interimInputTranscription?.text) {
            clientWs.send(
              JSON.stringify({
                type: "interim",
                text: content.interimInputTranscription.text,
              })
            );
          }

          if (content?.inputTranscription?.text) {
            clientWs.send(
              JSON.stringify({
                type: "final",
                text: content.inputTranscription.text,
              })
            );
          }
        },
        onerror: (err) => {
          console.error("Gemini Live transcription error:", err);
          if (clientWs.readyState === WebSocket.OPEN) {
            const formatted = formatApiError(err);
            clientWs.send(
              JSON.stringify({
                type: "error",
                message: formatted.message,
              })
            );
          }
        },
        onclose: () => {
          if (clientWs.readyState === WebSocket.OPEN) {
            clientWs.send(JSON.stringify({ type: "closed" }));
            clientWs.close();
          }
        },
      },
    });
  } catch (err) {
    console.error("Failed to connect to Gemini Live transcription:", err);
    if (clientWs.readyState === WebSocket.OPEN) {
      const formatted = formatApiError(err);
      clientWs.send(
        JSON.stringify({
          type: "error",
          message: formatted.message,
        })
      );
      clientWs.close();
    }
    return;
  }

  clientWs.on("message", (raw) => {
    let parsed: { type?: string; audio?: unknown };
    try {
      parsed = JSON.parse(raw.toString());
    } catch {
      return;
    }

    if (parsed.type === "stop") {
      stopTimer = setTimeout(() => {
        closeSession();
      }, 400);
      return;
    }

    if (typeof parsed.audio === "string" && parsed.audio.length > 0 && !closing) {
      if (sessionReady && session) {
        session.sendRealtimeInput({
          audio: { data: parsed.audio, mimeType: "audio/pcm;rate=16000" },
        });
      } else if (pendingAudio.length < 150) {
        pendingAudio.push(parsed.audio);
      }
    }
  });

  clientWs.on("close", () => {
    closeSession();
  });

  clientWs.on("error", () => {
    closeSession();
  });
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(`Server listening on http://0.0.0.0:${PORT}`);
});
