// Add global type for chat history
declare global {
  var _chatHistory: (AIMessage | HumanMessage)[] | undefined;
}
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
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
// Create and export a single Pool instance for use and cleanup
export const pool = new Pool({
  host: "localhost",
  port: 5432,
  user: "postgres",
  password: "root",
  database: "gmrt_webpages",
});
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import {
  PGVectorStore,
  DistanceStrategy,
} from "@langchain/community/vectorstores/pgvector";
import { PoolConfig } from "pg";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
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
  const url = "https://www.gmrtranscription.com/academic-transcription/";
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

  // Add a hash to each chunk for deduplication
  const crypto = await import("crypto");
  const documentsWithMetadata = splitDocuments.map((doc, index) => {
    const hash = crypto
      .createHash("sha256")
      .update(doc.pageContent)
      .digest("hex");
    return new Document({
      pageContent: doc.pageContent,
      metadata: {
        source: url,
        chunkIndex: index,
        createdAt: new Date().toISOString(),
        contentHash: hash,
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

    // Deduplication and upsert logic
    for (const doc of documentsWithMetadata) {
      const { pageContent, metadata } = doc;
      // Check if chunk exists by contentHash
      const existing = await pool.query(
        `SELECT id, content FROM pages WHERE metadata->>'contentHash' = $1`,
        [metadata.contentHash]
      );
      if (existing.rows.length === 0) {
        // New chunk, insert
        await vectorStore.addDocuments([doc]);
        console.log(`Inserted new chunk: ${metadata.chunkIndex}`);
      } else if (existing.rows[0].content !== pageContent) {
        // Chunk changed, update
        await pool.query(
          `UPDATE pages SET content = $1, embedding = $2, metadata = $3 WHERE id = $4`,
          [
            pageContent,
            await embeddings.embedQuery(pageContent),
            JSON.stringify(metadata),
            existing.rows[0].id,
          ]
        );
        console.log(`Updated chunk: ${metadata.chunkIndex}`);
      } else {
        // Chunk unchanged
        console.log(`Chunk unchanged: ${metadata.chunkIndex}`);
      }
    }
    console.log("Deduplication and update complete.");
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
export async function useConversationalRetrievalChain(
  userPrompt: string
): Promise<void> {
  const model = getChatModel();
  if (!model) {
    throw new Error("No chat model found. Please check your configuration.");
  }
  const promptTemplate = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are an assistant answering questions based on the provided context from web pages.
        Use only the information in {context} to answer the question below.

        - Format your answer using **Markdown** (e.g., headings, bullet points, bold text).
        - If the answer is not in the context, say "_I don't know based on the provided information._"
        - Be as concise and direct as possible. If the answer is a name or title, respond with only that.
        ### Chat History:
        {chat_history}
        ### Context:
        {context}

        ### Question:
        {input}

        ### Answer (in Markdown):
        `,
    ],
    new MessagesPlaceholder("chat_history"),
  ]);

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

  // Tune retriever for concise answers (return only top chunk)
  const retriever = vectorStore.asRetriever({
    searchType: "similarity",
    k: 1, // Return only the top similar chunk for direct answers
  });

  // Create a history-aware retriever to rephrase the user's query based on the context
  const retrievalPrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
    [
      "user",
      "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else.",
    ],
  ]);

  const historyAwareRetriever = await createHistoryAwareRetriever({
    llm: model,
    retriever,
    rephrasePrompt: retrievalPrompt,
  });
  const retrievalChain = await createRetrievalChain({
    retriever: historyAwareRetriever,
    combineDocsChain: combineDocumentsChain,
  });

  // Improved dynamic chat history management
  if (!globalThis._chatHistory) {
    globalThis._chatHistory = [];
  }

  // Optionally, clear history if user types a special command (e.g., "reset")
  if (userPrompt.trim().toLowerCase() === "reset") {
    globalThis._chatHistory = [];
    console.log("Chat history reset.");
    return;
  }

  // Add the user's message
  globalThis._chatHistory.push(new HumanMessage(userPrompt));

  // Only keep the last 3 pairs (HumanMessage/AIMessage) for context
  const historyPairs = [];
  for (let i = globalThis._chatHistory.length - 1; i >= 0; i -= 2) {
    if (globalThis._chatHistory[i] && globalThis._chatHistory[i - 1]) {
      historyPairs.unshift(globalThis._chatHistory[i - 1]);
      historyPairs.unshift(globalThis._chatHistory[i]);
    }
    if (historyPairs.length >= 6) break;
  }

  const response = await retrievalChain.invoke({
    input: userPrompt,
    chat_history: historyPairs,
  });
  // Fallback if answer is empty
  let answer =
    response.answer && response.answer.trim()
      ? response.answer
      : "_I don't know based on the provided information._";
  globalThis._chatHistory.push(new AIMessage(answer));
  console.log("Response:", answer);
}

export default { proccessURLToPgVectorStore, useConversationalRetrievalChain };
