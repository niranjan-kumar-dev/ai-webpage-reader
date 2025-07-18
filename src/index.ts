import { TextLoader } from "langchain/document_loaders/fs/text";
import { DocxLoader } from "@langchain/community/document_loaders/fs/docx";

import { fromTemplateExample } from "./services/from-template_AI_Assistant";
import { fromTemplateWebUrl } from "./services/from-template-weburl";
import proccessURLToPostgresqlVectorStore, {
  useFaissVectorStore,
} from "./services/from-template-webpage-loader";

import * as readline from "readline";
import {
  usePgVectorStore,
  pool,
  proccessURLToPgVectorStore,
} from "./services/from-template-webpage-loader-pgvector";
import {
  proccessURLToConversationalPgVectorStore,
  useConversationalRetrievalChain,
} from "./services/from-template-conversational-retrieval-chain";
import { scrapeSitemapAndSaveToDatabase } from "./services/scrapeContextualContent";

async function askQuestion(query: string): Promise<string> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  return new Promise((resolve) => {
    rl.question(query, (ans) => {
      rl.close();
      resolve(ans);
    });
  });
}

(async () => {
  // const { v4: uuidv4 } = require("uuid");
  // await scrapeSitemapAndSaveToDatabase(
  //   "https://www.gmrtranscription.com/sitemap.xml",
  //   uuidv4(),
  //   uuidv4(),
  //   uuidv4()
  // );
  //await proccessURLToConversationalPgVectorStore();
  while (true) {
    const userInput = await askQuestion(
      "Enter your question (or type 'exit' to quit): "
    );
    if (
      userInput.trim().toLowerCase() === "exit" ||
      userInput.trim().toLowerCase() === "quit"
    ) {
      console.log("Goodbye!");
      break;
    }
    console.log("Thinking...");
    await usePgVectorStore(userInput);
    //await useConversationalRetrievalChain(userInput);
  }
  // Cleanly close the PostgreSQL pool and exit
  await pool.end();
  process.exit(0);
})();

// const loader = new TextLoader("./src/data/sample.txt");
// loader
//   .load()
//   .then((docs) => {
//     console.log(docs[0].pageContent);
//   })
//   .catch((error) => {
//     console.error("Error loading documents:", error);
//   });

// const documentLoader = new DocxLoader("./src/data/Article.docx");

// (async () => {
//   const docs = await documentLoader.load();
//   console.log(docs[0].pageContent);
// })();
