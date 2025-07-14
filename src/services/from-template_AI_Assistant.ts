import * as readline from "readline";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { getChatModel } from "./model";
import { StringOutputParser } from "@langchain/core/output_parsers";

function askQuestion(query: string): Promise<string> {
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

export async function fromTemplateExample() {
  const model = getChatModel();

  if (!model) {
    throw new Error("No chat model found. Please check your configuration.");
  }

  const prompt = ChatPromptTemplate.fromMessages([
    {
      role: "system",
      content: "You are an AI assistant. Answer the user's question",
    },
    {
      role: "user",
      content: "{user_input}",
    },
  ]);

  const parser = new StringOutputParser();
  const chain = prompt.pipe(model).pipe(parser);

  console.log("Type 'exit' or 'quit' to end the chat.\n");
  while (true) {
    const user_input = await askQuestion("You: ");
    if (
      user_input.trim().toLowerCase() === "exit" ||
      user_input.trim().toLowerCase() === "quit"
    ) {
      console.log("Goodbye!");
      break;
    }
    try {
      //const resolvedPrompt = await prompt.format({ user_input });
      const response = await chain.invoke({ user_input });
      console.log("AI:", response);
    } catch (error) {
      console.error("Error generating response:", error);
    }
  }
}
