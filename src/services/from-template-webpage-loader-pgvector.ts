// Utility function to get PGVectorStore config
export function getPgVectorStoreConfig(): {
  postgresConnectionOptions: PoolConfig;
  tableName: string;
  columns: {
    vectorColumnName: string;
    contentColumnName: string;
    metadataColumnName: string;
  };
  distanceStrategy: DistanceStrategy;
} {
  return {
    postgresConnectionOptions: {
      type: "postgres",
      host: "localhost",
      port: 5432,
      user: "postgres",
      password: "root",
      database: "gmrt_webpages",
    } as PoolConfig,
    tableName: "pages",
    columns: {
      vectorColumnName: "embedding",
      contentColumnName: "content",
      metadataColumnName: "metadata",
    },
    distanceStrategy: "cosine" as DistanceStrategy,
  };
}
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
import { Pool } from "pg";

// Create and export a single Pool instance for use and cleanup
export const pool = new Pool({
  host: "localhost",
  port: 5432,
  user: "postgres",
  password: "root",
  database: "gmrt_webpages",
});
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import {
  PGVectorStore,
  DistanceStrategy,
} from "@langchain/community/vectorstores/pgvector";
import { PoolConfig } from "pg";
dotenv.config();

export async function proccessURLToPgVectorStore(): Promise<void> {
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

  const documentsWithMetadata = splitDocuments.map((doc, index) => {
    return new Document({
      pageContent: doc.pageContent,
      metadata: {
        source: url,
        chunkIndex: index,
        createdAt: new Date().toISOString(),
      },
    });
  });

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

  try {
    const vectorStore = await PGVectorStore.initialize(
      embeddings,
      getPgVectorStoreConfig()
    );
    console.log("Vector store initialized");

    await vectorStore.addDocuments(documentsWithMetadata);
    console.log("Documents added to vector store successfully");
  } catch (error) {
    console.error("Error saving vector store to PostgreSQL:", error);
  }

  // const vectorStore = await FaissStore.fromDocuments(
  //   splitDocuments,
  //   embeddings
  // );
  //const vectorStore = await FaissStore.fromDocuments(allDocs, embeddings);
  //await vectorStore.save("./vector_store-url");
  //console.log("Vector store created and saved");
}

/*Using a Faiss Vector store in chain */
export async function usePgVectorStore(userPrompt: string): Promise<void> {
  const model = getChatModel();
  if (!model) {
    throw new Error("No chat model found. Please check your configuration.");
  }
  //console.log("[AI WebReader] Model loaded");

  const promptTemplate = ChatPromptTemplate.fromTemplate(`
     Answer the user's question
     Context: {context}
     question: {input}
  `);

  const combineDocumentsChain = await createStuffDocumentsChain({
    llm: model,
    prompt: promptTemplate,
  });

  // Use a top-performing embedding model for best accuracy
  const embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HUGGINGFACEHUB_API_KEY,
    model: process.env.HUGGINGFACEHUB_API_MODEL, // Dimension: 768 (matches your DB)
    provider: "auto",
  });

  const vectorStore = await PGVectorStore.initialize(
    embeddings,
    getPgVectorStoreConfig()
  );

  // Tune retriever for more context (increase k)
  const retriever = vectorStore.asRetriever({
    searchType: "similarity",
    k: 3, // Return top 3 similar chunks for better context
  });

  const retrievalChain = await createRetrievalChain({
    retriever,
    combineDocsChain: combineDocumentsChain,
  });

  const response = await retrievalChain.invoke({ input: userPrompt });
  console.log("Response:", response.answer);
}

export default { proccessURLToPgVectorStore, usePgVectorStore };
