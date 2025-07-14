export const runtime = "nodejs";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { RetrievalQAChain } from "langchain/chains";
import { Document } from "langchain/document";
import { PuppeteerWebBaseLoader } from "@langchain/community/document_loaders/web/puppeteer";
import * as puppeteer from "puppeteer";
import * as cheerio from "cheerio";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import * as dotenv from "dotenv";
import { getChatModel } from "./model";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";

dotenv.config();

export async function proccessURLToPostgresqlVectorStore(): Promise<void> {
  // const model = getChatModel();
  // if (!model) {
  //   throw new Error("No chat model found. Please check your configuration.");
  // }
  // console.log("[AI WebReader] Model loaded");
  // const prompt = ChatPromptTemplate.fromTemplate(`
  //    Answer the user's question
  //    Context: {context}
  //    question: {input}

  //    `);

  // const chain = await createStuffDocumentsChain({
  //   llm: model,
  //   prompt: prompt,
  // });
  const url = "https://www.gmrtranscription.com/";
  const loader = new PuppeteerWebBaseLoader(url, {
    launchOptions: {
      headless: "new",
    },
    async evaluate(page: puppeteer.Page, browser: puppeteer.Browser) {
      try {
        await page.goto(url, { waitUntil: "networkidle0" });
        const textContent = await page.evaluate(() => {
          const bodyElement = document.querySelector("body");
          return bodyElement ? bodyElement.innerText : "";
        });
        await browser.close();
        return textContent || "";
      } catch (error) {
        console.error("Error during page evaluation:", error);
        await browser.close();
        return "";
      }
    },
  });
  console.log("Loading URL to Docs");
  const urlDocs = await loader.load();
  const pageContent = urlDocs[0].pageContent;
  console.log("Page content loaded:", pageContent);
  // Load the HTML content into a Cheerio document
  const $ = cheerio.load(pageContent);
  // Extract text from the body`
  $("script, style").remove(); // Remove script and style tags`

  //Further clean-up using regular expressions (exmple)
  const cleanedText = $("body")
    .html()
    ?.replace(/<style[^>]*>.*<\/style>/, " ");

  // Load the cleaned HTML into again to extract text
  const cleaned$ = cheerio.load(cleanedText!);
  const textContent = cleaned$("body").text();
  console.log("Extracted text:", textContent);
  const docs = textContent.replace(/[^\x20-\x7E]+/g, " ");
  console.log("Cleaned text:", docs);

  // create document instance
  const documents = [new Document({ pageContent: docs })];
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 50,
  });
  const splitDocuments = await splitter.splitDocuments(documents);
  console.log("Split documents:", splitDocuments);
  const embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HUGGINGFACEHUB_API_KEY, // Defaults to process.env.HUGGINGFACEHUB_API_KEY
    model: process.env.HUGGINGFACEHUB_API_MODEL, // Defaults to `BAAI/bge-base-en-v1.5` if not provided
    provider: "auto", // Falls back to auto selection mechanism within Hugging Face's inference API if not provided
  });

  // const batchSize = 10; // Adjust batch size as needed
  // const allDocs = [];
  // for (let i = 0; i < splitDocuments.length; i += batchSize) {
  //   const batch = splitDocuments.slice(i, i + batchSize);
  //   allDocs.push(...batch);
  // }
  if (!splitDocuments.length) {
    throw new Error("No documents to embed.");
  }
  const vectorStore = await FaissStore.fromDocuments(
    splitDocuments,
    embeddings
  );
  //const vectorStore = await FaissStore.fromDocuments(allDocs, embeddings);
  await vectorStore.save("./vector_store-url");
  console.log("Vector store created and saved");
}

/*Using a Faiss Vector store in chain */

export async function useFaissVectorStore(userPrompt: string): Promise<void> {
  const model = getChatModel();
  if (!model) {
    throw new Error("No chat model found. Please check your configuration.");
  }
  console.log("[AI WebReader] Model loaded");

  const promptTemplate = ChatPromptTemplate.fromTemplate(`
     Answer the user's question
     Context: {context}
     question: {input}
  `);

  const combineDocumentsChain = await createStuffDocumentsChain({
    llm: model,
    prompt: promptTemplate,
  });

  const embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HUGGINGFACEHUB_API_KEY, // Defaults to process.env.HUGGINGFACEHUB_API_KEY
    model: process.env.HUGGINGFACEHUB_API_MODEL, // Defaults to `BAAI/bge-base-en-v1.5` if not provided
    provider: "auto", // Falls back to auto selection mechanism within Hugging Face's inference API if not provided
  });

  const vectorStore = await FaissStore.load("./vector_store-url", embeddings);

  // Create a retriever from the vector store
  const retriever = vectorStore.asRetriever({
    searchType: "similarity",
    k: 1, // Number of documents to retrieve
  });
  const retrievalChain = await createRetrievalChain({
    retriever: retriever,
    combineDocsChain: combineDocumentsChain,
  });

  const response = await retrievalChain.invoke({ input: userPrompt });
  console.log("Response:", response.answer);
}

export default { proccessURLToPostgresqlVectorStore, useFaissVectorStore };
