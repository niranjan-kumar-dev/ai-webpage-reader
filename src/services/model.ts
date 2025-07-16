import { ChatOllama } from "@langchain/ollama";
import { ChatOpenAI } from "@langchain/openai";
import * as dotenv from "dotenv";

dotenv.config();

export function getChatModel() {
  const llmProvider = process.env.LLM_PROVIDER || "ollama";

  if (llmProvider === "openai") {
    return getOpenAIModel();
  } else {
    return getOllamaModel();
  }
}

function getOllamaModel() {
  return new ChatOllama({
    model: process.env.LLM_MODEL || "llama3.2",
    temperature: parseFloat(process.env.LLM_TEMPERATURE || "0.7"),
  });
}

function getOpenAIModel() {
  return new ChatOpenAI({
    model: process.env.LLM_MODEL || "gpt-3.5-turbo",
    temperature: parseFloat(process.env.LLM_TEMPERATURE || "0.7"),
  });
}
