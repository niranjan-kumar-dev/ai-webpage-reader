import { getChatModel } from "./model";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { StringOutputParser } from "@langchain/core/output_parsers";

async function extractWebContent(url: string): Promise<string> {
  const loader = new CheerioWebBaseLoader(url);
  const docs = await loader.load();
  return docs.map((doc: { pageContent: string }) => doc.pageContent).join("\n");
}

async function processWebContent(url: string): Promise<any> {
  const webContent = await extractWebContent(url);
  const model = getChatModel();
  const prompt = `Extract the following information from the web page text and return it as a JSON object with the following structure:\n{\n  Title: string,\n  Content: string,\n  Footer: string\n}\nIf any part is missing, leave it as an empty string.\n\nWeb page text:\n${webContent}`;
  const parser = new StringOutputParser();
  const chain = parser.pipe(model);
  const response = await chain.invoke(prompt);

  try {
    return JSON.parse(
      typeof response.content === "string"
        ? response.content
        : JSON.stringify(response.content)
    );
  } catch {
    return response;
  }
}

export async function fromTemplateWebUrl(url: string): Promise<string> {
  try {
    const content = await processWebContent(url);
    return content;
  } catch (error) {
    console.error("Error loading web content:", error);
    throw new Error("Failed to load web content");
  }
}
