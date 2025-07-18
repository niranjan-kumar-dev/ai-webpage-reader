import puppeteer from "puppeteer";
import * as cheerio from "cheerio";
import { writeFile } from "fs/promises";
import axios from "axios";

// Load environment variables

import { v4 as uuidv4 } from "uuid";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Document } from "langchain/document";
import { OpenAIEmbeddings } from "@langchain/openai";
import pool from "../config/database"; // Adjusted the path to match the correct location
import { logger } from "../utils/logger";
import * as dotenv from "dotenv";
import path from "path";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";

// Ensure environment variables are loaded
dotenv.config({ path: path.join(__dirname, "../../.env") });
//dotenv.config();
interface StructuredBlock {
  heading: string;
  content: string;
  list: string[];
}

interface SitemapEntry {
  url: string;
  lastmod?: string;
  changefreq?: string;
  priority?: string;
}

interface ScrapedPage {
  url: string;
  title: string;
  blocks: StructuredBlock[];
  scrapedAt: string;
}

interface EmbeddingResult {
  id: string;
  url: string;
  title: string;
  content: string;
  embedding: number[];
  metadata: any;
}

interface ScrapeOptions {
  maxUrls?: number;
  concurrency?: number;
  urlFilter?: (url: string) => boolean;
  skipWords?: string[];
  skipExtensions?: string[];
}

interface EmbeddingOptions {
  userId?: string;
  botId?: string;
  dataSourceId?: string;
  saveToDatabase?: boolean;
  saveToFile?: boolean;
  outputFile?: string;
}

// Initialize text splitter and embeddings
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 80,
});

const embeddings = new HuggingFaceInferenceEmbeddings({
  apiKey: process.env.HUGGINGFACEHUB_API_KEY,
  model: process.env.HUGGINGFACEHUB_API_MODEL, // Dimension: 768 (matches your DB)
  provider: "auto",
});

// Default words to skip in URLs
const DEFAULT_SKIP_WORDS = [
  "thanks",
  "thank-you",
  "privacy",
  "privacy-policy",
  "terms",
  "terms-of-service",
  "terms-and-conditions",
  "legal",
  "cookie",
  "cookies",
  "sitemap",
  "robots",
  "admin",
  "login",
  "signup",
  "register",
  "cart",
  "checkout",
  "payment",
  "search",
  "error",
  "404",
  "maintenance",
  "coming-soon",
  "under-construction",
];

// Default file extensions to skip
const DEFAULT_SKIP_EXTENSIONS = [
  ".pdf",
  ".doc",
  ".docx",
  ".xls",
  ".xlsx",
  ".ppt",
  ".pptx",
  ".zip",
  ".rar",
  ".tar",
  ".gz",
  ".jpg",
  ".jpeg",
  ".png",
  ".gif",
  ".svg",
  ".mp4",
  ".mp3",
  ".avi",
  ".mov",
  ".wmv",
  ".xml",
  ".json",
  ".txt",
  ".csv",
];

// Helper function to check if URL should be skipped
function shouldSkipUrl(
  url: string,
  skipWords: string[],
  skipExtensions: string[]
): boolean {
  const urlLower = url.toLowerCase();

  // Check for skip words in URL path
  const hasSkipWord = skipWords.some(
    (word) =>
      urlLower.includes(`/${word}`) ||
      urlLower.includes(`-${word}`) ||
      urlLower.includes(`_${word}`) ||
      urlLower.includes(`${word}-`) ||
      urlLower.includes(`${word}_`) ||
      urlLower.endsWith(`/${word}`) ||
      urlLower.includes(`?${word}`) ||
      urlLower.includes(`&${word}`)
  );

  // Check for skip extensions
  const hasSkipExtension = skipExtensions.some((ext) => urlLower.endsWith(ext));

  return hasSkipWord || hasSkipExtension;
}

// Step 1: Parse sitemap to extract URLs
async function parseSitemap(sitemapUrl: string): Promise<SitemapEntry[]> {
  console.log(`🗺️  Parsing sitemap: ${sitemapUrl}`);

  try {
    const response = await axios.get(sitemapUrl);
    const $ = cheerio.load(response.data, { xmlMode: true });
    const urls: SitemapEntry[] = [];

    $("url").each((_, element) => {
      const loc = $(element).find("loc").text();
      const lastmod = $(element).find("lastmod").text();
      const changefreq = $(element).find("changefreq").text();
      const priority = $(element).find("priority").text();

      if (loc) {
        urls.push({
          url: loc,
          lastmod: lastmod || undefined,
          changefreq: changefreq || undefined,
          priority: priority || undefined,
        });
      }
    });

    // Handle sitemap index (sitemaps that reference other sitemaps)
    const sitemapRefs = $("sitemap loc")
      .map((_, el) => $(el).text())
      .get();

    if (sitemapRefs.length > 0) {
      console.log(`📂 Found ${sitemapRefs.length} nested sitemaps`);
      for (const nestedSitemapUrl of sitemapRefs) {
        try {
          const nestedUrls = await parseSitemap(nestedSitemapUrl);
          urls.push(...nestedUrls);
        } catch (error) {
          console.warn(
            `⚠️  Failed to parse nested sitemap: ${nestedSitemapUrl}`,
            error
          );
        }
      }
    }

    console.log(`✅ Found ${urls.length} URLs in sitemap`);
    return urls;
  } catch (error) {
    console.error(`❌ Error parsing sitemap: ${sitemapUrl}`, error);
    throw error;
  }
}

// Step 2: Load full HTML (JS-rendered)
async function loadHTML(url: string): Promise<{ html: string; title: string }> {
  const browser = await puppeteer.launch({
    headless: true,
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
      "--disable-web-security",
    ],
  });
  const page = await browser.newPage();

  try {
    // Set user agent to avoid bot detection
    await page.setUserAgent(
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    );

    // Set viewport
    await page.setViewport({ width: 1920, height: 1080 });

    // Try different wait strategies with longer timeout
    await page.goto(url, {
      waitUntil: "domcontentloaded",
      timeout: 60000,
    });

    // Wait a bit more for dynamic content
    await new Promise((resolve) => setTimeout(resolve, 2000));

    const html = await page.content();
    const title = await page.title();
    await browser.close();
    return { html, title };
  } catch (error) {
    await browser.close();
    throw error;
  }
}

// Step 3: Parse and structure content
function extractStructuredContent(html: string): StructuredBlock[] {
  const $ = cheerio.load(html);
  const result: StructuredBlock[] = [];
  const seenHeadings = new Set<string>(); // Track duplicate headings

  // Skip common footer/header selectors
  const skipSelectors = [
    "header",
    "nav",
    ".header",
    ".navigation",
    ".nav",
    ".copyright",
    ".social-media",
    ".social-links",
    "script",
    "style",
    "noscript",
  ];

  // Remove unwanted elements
  skipSelectors.forEach((selector) => {
    $(selector).remove();
  });

  const container = $("main").length ? $("main") : $("body");
  let currentBlock: StructuredBlock | null = null;

  container.find("h1, h2, h3, h4, p, ul, ol, a, span").each((_, el) => {
    const tag = $(el)[0].tagName.toLowerCase();
    let text = $(el).text().trim().replace(/\s+/g, " ");

    // Skip if text is too short or contains unwanted content
    if (!text || text.length < 3) return;

    // Skip JavaScript code, copyright, and other unwanted content
    if (
      text.includes("document.getElementById") ||
      text.includes("setAttribute") ||
      text.includes("getTime()") ||
      text.includes("Δ") ||
      text.includes("All Rights Reserved") ||
      text.includes("Powered by") ||
      text.includes("Copyright 20") ||
      text.startsWith("Thank you for sharing referral")
    ) {
      return;
    }

    // Clean up common patterns
    text = text.replace(/\s*Δ\s*/g, "").trim();
    text = text.replace(/\s*\n\s*/g, "\n").trim();

    if (tag.startsWith("h")) {
      // Skip duplicate headings
      const headingLower = text.toLowerCase();
      if (seenHeadings.has(headingLower)) {
        return;
      }
      seenHeadings.add(headingLower);

      // Skip common unwanted headings
      if (
        headingLower.includes("copyright") ||
        headingLower.includes("powered by") ||
        headingLower.includes("all rights reserved")
      ) {
        return;
      }

      // Start new block
      if (currentBlock !== null) {
        result.push(currentBlock);
      }
      currentBlock = {
        heading: text,
        content: "",
        list: [],
      };
    } else if (tag === "p") {
      if (!currentBlock) {
        currentBlock = {
          heading: "",
          content: "",
          list: [],
        };
      }

      // Skip if paragraph is too short or contains unwanted content
      if (text.length < 10) return;

      currentBlock.content += (currentBlock.content ? "\n\n" : "") + text;
    } else if (tag === "ul" || tag === "ol") {
      if (!currentBlock) {
        currentBlock = {
          heading: "",
          content: "",
          list: [],
        };
      }

      $(el)
        .find("li")
        .each((_, li) => {
          const liText = $(li).text().trim().replace(/\s+/g, " ");
          if (liText && liText.length > 5 && currentBlock) {
            currentBlock.list.push(liText);
          }
        });

      // After processing a list, check if there are any related paragraphs nearby
      // Look for paragraphs that might be related to this section
      const nextElements = $(el).nextAll("p").slice(0, 2);
      nextElements.each((_, nextEl) => {
        const nextText = $(nextEl).text().trim().replace(/\s+/g, " ");
        if (nextText && nextText.length > 20 && currentBlock) {
          // Add this paragraph to current block if it's not already captured
          if (!currentBlock.content.includes(nextText)) {
            currentBlock.content +=
              (currentBlock.content ? "\n\n" : "") + nextText;
          }
        }
      });
    }
  });

  // Add the last block if it has content
  if (currentBlock !== null) {
    result.push(currentBlock);
  }

  // Filter out blocks with empty or very short content
  return result.filter(
    (block) =>
      block.heading.length > 0 &&
      (block.content.length > 10 || block.list.length > 0)
  );
}

// Step 4: Enhanced scrape single page with embedding support
async function scrapePageWithEmbedding(
  url: string,
  options: EmbeddingOptions = {}
): Promise<ScrapedPage | null> {
  try {
    console.log(`🔍 Scraping with embedding: ${url}`);
    const { html, title } = await loadHTML(url);
    const blocks = extractStructuredContent(html);

    const scrapedPage: ScrapedPage = {
      url,
      title,
      blocks,
      scrapedAt: new Date().toISOString(),
    };

    // If embedding options are provided, process and save to database
    if (
      options.saveToDatabase &&
      options.userId &&
      options.botId &&
      options.dataSourceId
    ) {
      await processPageForEmbedding(scrapedPage, options);
    }

    return scrapedPage;
  } catch (error) {
    console.error(`❌ Failed to scrape ${url}:`, error);
    return null;
  }
}

// Step 5: Process scraped page content for embedding and database storage
async function processPageForEmbedding(
  page: ScrapedPage,
  options: EmbeddingOptions
): Promise<void> {
  try {
    // Convert structured blocks to clean text content
    const cleanContent = convertBlocksToText(page.blocks);

    if (!cleanContent.trim() || cleanContent.length < 50) {
      logger.warn(`Insufficient content for embedding: ${page.url}`);
      return;
    }

    // Create a document with the clean content
    const document = new Document({
      pageContent: cleanContent,
      metadata: {
        source: page.url,
        title: page.title,
        scrapedAt: page.scrapedAt,
        contentType: "structured_web_content",
        totalBlocks: page.blocks.length,
      },
    });

    // Split into chunks
    const chunks = await textSplitter.splitDocuments([document]);

    logger.info(`Split content into ${chunks.length} chunks for ${page.url}`);

    // Process each chunk
    for (const chunk of chunks) {
      const content = chunk.pageContent;

      // Skip very short chunks
      if (content.trim().length < 50) continue;

      // Generate embedding
      const embeddingVector = await embeddings.embedQuery(content);

      // Enhanced metadata
      const enhancedMetadata = {
        ...chunk.metadata,
        url: page.url,
        title: page.title,
        source: "structured_scraping",
        contentType: "web_page",
        chunkIndex: chunks.indexOf(chunk),
        totalChunks: chunks.length,
        extractedAt: page.scrapedAt,
        processingMethod: "structured_blocks",
        originalLength: content.length,
      };

      // Store in database
      const embeddingId = uuidv4();
      await pool.query(
        "INSERT INTO embeddings (id, user_id, bot_id, data_source_id, content, embedding, metadata) VALUES ($1, $2, $3, $4, $5, $6, $7)",
        [
          embeddingId,
          options.userId,
          options.botId,
          options.dataSourceId,
          content,
          JSON.stringify(embeddingVector),
          JSON.stringify(enhancedMetadata),
        ]
      );
    }

    logger.info(
      `Successfully processed ${chunks.length} chunks for ${page.url}`
    );
  } catch (error) {
    logger.error(`Error processing page for embedding ${page.url}:`, error);
    throw error;
  }
}

// Step 6: Convert structured blocks to clean text content
function convertBlocksToText(blocks: StructuredBlock[]): string {
  return blocks
    .map((block) => {
      let text = "";

      // Add heading if present
      if (block.heading) {
        text += `${block.heading}\n\n`;
      }

      // Add content if present
      if (block.content) {
        text += `${block.content}\n\n`;
      }

      // Add list items if present
      if (block.list.length > 0) {
        text += block.list.map((item) => `• ${item}`).join("\n") + "\n\n";
      }

      return text;
    })
    .join("")
    .trim();
}

// Step 7: Enhanced sitemap processing with embedding support
async function scrapeSitemapWithEmbedding(
  sitemapUrl: string,
  options: ScrapeOptions & EmbeddingOptions = {}
): Promise<ScrapedPage[]> {
  const {
    maxUrls = 50,
    concurrency = 3,
    urlFilter,
    skipWords = DEFAULT_SKIP_WORDS,
    skipExtensions = DEFAULT_SKIP_EXTENSIONS,
    saveToDatabase = false,
    userId,
    botId,
    dataSourceId,
  } = options;

  console.log(`🚀 Starting sitemap scraping with embedding support...`);

  try {
    // Update status to processing if database integration is enabled
    if (saveToDatabase && dataSourceId) {
      await pool.query(
        "UPDATE embeddings SET metadata = COALESCE(metadata, '{}') || $1 WHERE id = $2",
        [JSON.stringify({ status: "processing" }), dataSourceId]
      );
    }

    // Parse sitemap
    const sitemapEntries = await parseSitemap(sitemapUrl);
    let urls = sitemapEntries.map((entry) => entry.url);

    console.log(`📊 Found ${urls.length} total URLs in sitemap`);

    // Apply filtering
    const originalCount = urls.length;
    urls = urls.filter((url) => !shouldSkipUrl(url, skipWords, skipExtensions));
    console.log(
      `🚫 Filtered out ${
        originalCount - urls.length
      } URLs with skip words/extensions`
    );

    if (urlFilter) {
      const beforeCustomFilter = urls.length;
      urls = urls.filter(urlFilter);
      console.log(
        `🔍 Custom filter applied: ${
          beforeCustomFilter - urls.length
        } additional URLs filtered out`
      );
    }

    if (maxUrls && urls.length > maxUrls) {
      urls = urls.slice(0, maxUrls);
      console.log(`📊 Limited to first ${maxUrls} URLs`);
    }

    // Scrape pages with embedding support
    const scrapedPages = await scrapePagesWithEmbedding(urls, concurrency, {
      userId,
      botId,
      dataSourceId,
      saveToDatabase,
    });

    // Update status to completed if database integration is enabled
    if (saveToDatabase && dataSourceId) {
      await pool.query(
        "UPDATE embeddings SET metadata = COALESCE(metadata, '{}') || $1 WHERE id = $2",
        [
          JSON.stringify({
            status: "completed",
            totalPages: scrapedPages.length,
          }),
          dataSourceId,
        ]
      );
    }

    console.log(
      `✅ Completed sitemap scraping with ${scrapedPages.length} pages processed`
    );
    return scrapedPages;
  } catch (error) {
    // Update status to failed if database integration is enabled
    if (saveToDatabase && dataSourceId) {
      const errorMessage =
        error instanceof Error ? error.message : "Unknown error";
      await pool.query(
        "UPDATE embeddings SET metadata = COALESCE(metadata, '{}') || $1 WHERE id = $2",
        [
          JSON.stringify({ status: "failed", error: errorMessage }),
          dataSourceId,
        ]
      );
    }

    logger.error(`Error in sitemap scraping with embedding:`, error);
    throw error;
  }
}

// Step 8: Scrape multiple pages with embedding support
async function scrapePagesWithEmbedding(
  urls: string[],
  concurrencyLimit: number = 3,
  embeddingOptions: EmbeddingOptions = {}
): Promise<ScrapedPage[]> {
  const results: ScrapedPage[] = [];

  for (let i = 0; i < urls.length; i += concurrencyLimit) {
    const batch = urls.slice(i, i + concurrencyLimit);
    console.log(
      `📦 Processing batch ${Math.floor(i / concurrencyLimit) + 1}/${Math.ceil(
        urls.length / concurrencyLimit
      )} with embedding support`
    );

    const batchPromises = batch.map((url) =>
      scrapePageWithEmbedding(url, embeddingOptions)
    );
    const batchResults = await Promise.all(batchPromises);

    const validResults = batchResults.filter(
      (result) => result !== null
    ) as ScrapedPage[];
    results.push(...validResults);

    // Add delay between batches
    if (i + concurrencyLimit < urls.length) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }

  return results;
}

// Step 9: Original scrape single page function (kept for backward compatibility)
async function scrapePage(url: string): Promise<ScrapedPage | null> {
  try {
    console.log(`🔍 Scraping: ${url}`);
    const { html, title } = await loadHTML(url);
    const blocks = extractStructuredContent(html);

    return {
      url,
      title,
      blocks,
      scrapedAt: new Date().toISOString(),
    };
  } catch (error) {
    console.error(`❌ Failed to scrape ${url}:`, error);
    return null;
  }
}

// Step 5: Scrape multiple pages with concurrency control
async function scrapePages(
  urls: string[],
  concurrencyLimit: number = 3
): Promise<ScrapedPage[]> {
  const results: ScrapedPage[] = [];

  for (let i = 0; i < urls.length; i += concurrencyLimit) {
    const batch = urls.slice(i, i + concurrencyLimit);
    console.log(
      `📦 Processing batch ${Math.floor(i / concurrencyLimit) + 1}/${Math.ceil(
        urls.length / concurrencyLimit
      )}`
    );

    const batchPromises = batch.map((url) => scrapePage(url));
    const batchResults = await Promise.all(batchPromises);

    const validResults = batchResults.filter(
      (result) => result !== null
    ) as ScrapedPage[];
    results.push(...validResults);

    // Add delay between batches to be respectful
    if (i + concurrencyLimit < urls.length) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }

  return results;
}

// Step 6: Enhanced main orchestrator for sitemap processing
async function scrapeSitemapAndSaveStructuredContent(
  sitemapUrl: string,
  outputFile: string,
  options: ScrapeOptions = {}
): Promise<void> {
  const {
    maxUrls = 50,
    concurrency = 3,
    urlFilter,
    skipWords = DEFAULT_SKIP_WORDS,
    skipExtensions = DEFAULT_SKIP_EXTENSIONS,
  } = options;

  console.log(`🚀 Starting sitemap scraping process...`);

  // Parse sitemap
  const sitemapEntries = await parseSitemap(sitemapUrl);
  let urls = sitemapEntries.map((entry) => entry.url);

  console.log(`📊 Found ${urls.length} total URLs in sitemap`);

  // Apply built-in skip filters
  const originalCount = urls.length;
  urls = urls.filter((url) => !shouldSkipUrl(url, skipWords, skipExtensions));
  console.log(
    `🚫 Filtered out ${
      originalCount - urls.length
    } URLs with skip words/extensions`
  );

  // Apply custom URL filter if provided
  if (urlFilter) {
    const beforeCustomFilter = urls.length;
    urls = urls.filter(urlFilter);
    console.log(
      `🔍 Custom filter applied: ${
        beforeCustomFilter - urls.length
      } additional URLs filtered out`
    );
  }

  console.log(`✅ Final URL count after all filters: ${urls.length}`);

  // Limit URLs if specified
  if (maxUrls && urls.length > maxUrls) {
    urls = urls.slice(0, maxUrls);
    console.log(`📊 Limited to first ${maxUrls} URLs`);
  }

  // Scrape pages
  const scrapedPages = await scrapePages(urls, concurrency);

  // Save results
  const output = {
    sitemapUrl,
    scrapedAt: new Date().toISOString(),
    totalUrls: urls.length,
    successfullyScraped: scrapedPages.length,
    skippedWords: skipWords,
    skippedExtensions: skipExtensions,
    pages: scrapedPages,
  };

  await writeFile(outputFile, JSON.stringify(output, null, 2), "utf-8");
  console.log(`✅ Saved ${scrapedPages.length} pages to ${outputFile}`);
}

// Step 7: Original single-page scraper (kept for backward compatibility)
// Step 7: Original single-page scraper (kept for backward compatibility)
async function scrapeAndSaveStructuredContent(
  url: string,
  outputFile: string
): Promise<void> {
  console.log(`🔍 Scraping structured content from: ${url}`);
  const { html } = await loadHTML(url);
  const structuredData = extractStructuredContent(html);
  await writeFile(outputFile, JSON.stringify(structuredData, null, 2), "utf-8");
  console.log(`✅ Saved structured content to ${outputFile}`);
}

// Helper function to convert structured blocks to text content
function blocksToText(blocks: StructuredBlock[]): string {
  return blocks
    .map((block) => {
      let text = block.heading ? `${block.heading}\n` : "";
      if (block.content) {
        text += `${block.content}\n`;
      }
      if (block.list.length > 0) {
        text += block.list.map((item) => `• ${item}`).join("\n") + "\n";
      }
      return text;
    })
    .join("\n")
    .trim();
}

// Function to generate embeddings
async function generateEmbedding(text: string): Promise<number[]> {
  try {
    const embeddings = new HuggingFaceInferenceEmbeddings({
      apiKey: process.env.HUGGINGFACEHUB_API_KEY,
      model: process.env.HUGGINGFACEHUB_API_MODEL, // Dimension: 768 (matches your DB)
      provider: "auto",
    });

    const embedding = await embeddings.embedQuery(text);
    return embedding;
  } catch (error) {
    logger.error("Error generating embedding:", error);
    throw error;
  }
}

// Function to save content and embedding to database
async function saveToDatabase(
  url: string,
  title: string,
  content: string,
  embedding: number[],
  dataSourceId: string,
  userId: string,
  botId: string,
  metadata: any = {}
): Promise<string> {
  try {
    // Create enhanced metadata that includes URL and title
    const enhancedMetadata = {
      ...metadata,
      url,
      title,
      source: "scraped_content",
      contentType: "structured_web_content",
    };

    const result = await pool.query(
      `INSERT INTO embeddings (id, user_id, bot_id, data_source_id, content, embedding, metadata, created_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
       RETURNING id`,
      [
        uuidv4(),
        userId,
        botId,
        dataSourceId,
        content,
        embedding, // Pass as array, not string
        JSON.stringify(enhancedMetadata),
      ]
    );

    return result.rows[0].id;
  } catch (error) {
    logger.error("Error saving to database:", error);
    throw error;
  }
}

// Function to process a single page and save to database
async function scrapePageAndSaveToDatabase(
  url: string,
  dataSourceId: string,
  userId: string,
  botId: string
): Promise<EmbeddingResult | null> {
  try {
    logger.info(`Processing URL: ${url}`);

    // Scrape the page
    const scrapedPage = await scrapePage(url);
    if (!scrapedPage) {
      logger.warn(`Failed to scrape page: ${url}`);
      return null;
    }

    // Convert blocks to text content
    const content = blocksToText(scrapedPage.blocks);

    if (!content || content.trim().length < 50) {
      logger.warn(`Insufficient content for URL: ${url}`);
      return null;
    }

    // Generate embedding
    const embedding = await generateEmbedding(content);

    // Save to database
    const id = await saveToDatabase(
      url,
      scrapedPage.title,
      content,
      embedding,
      dataSourceId,
      userId,
      botId,
      {
        blockCount: scrapedPage.blocks.length,
        scrapedAt: scrapedPage.scrapedAt,
      }
    );

    logger.info(`Successfully processed and saved URL: ${url} with ID: ${id}`);

    return {
      id,
      url,
      title: scrapedPage.title,
      content,
      embedding,
      metadata: {
        blockCount: scrapedPage.blocks.length,
        scrapedAt: scrapedPage.scrapedAt,
      },
    };
  } catch (error) {
    logger.error(`Error processing URL ${url}:`, error);
    return null;
  }
}

// Main function to scrape sitemap and save to database
async function scrapeSitemapAndSaveToDatabase(
  sitemapUrl: string,
  dataSourceId: string,
  userId: string,
  botId: string,
  options: ScrapeOptions = {}
): Promise<{
  totalUrls: number;
  successfullyProcessed: number;
  skippedUrls: number;
  results: EmbeddingResult[];
}> {
  const {
    maxUrls = 50,
    concurrency = 2,
    urlFilter,
    skipWords = DEFAULT_SKIP_WORDS,
    skipExtensions = DEFAULT_SKIP_EXTENSIONS,
  } = options;

  logger.info(`Starting sitemap processing for: ${sitemapUrl}`);

  try {
    // Update data source status to processing
    await pool.query(
      "UPDATE embeddings SET metadata = COALESCE(metadata, '{}') || $1 WHERE id = $2",
      [JSON.stringify({ status: "processing" }), dataSourceId]
    );

    // Parse sitemap
    const sitemapEntries = await parseSitemap(sitemapUrl);
    let urls = sitemapEntries.map((entry) => entry.url);

    logger.info(`Found ${urls.length} total URLs in sitemap`);

    // Apply built-in skip filters
    const originalCount = urls.length;
    urls = urls.filter((url) => !shouldSkipUrl(url, skipWords, skipExtensions));
    const skippedByFilters = originalCount - urls.length;

    logger.info(
      `Filtered out ${skippedByFilters} URLs with skip words/extensions`
    );

    // Apply custom URL filter if provided
    if (urlFilter) {
      const beforeCustomFilter = urls.length;
      urls = urls.filter(urlFilter);
      logger.info(
        `Custom filter applied: ${
          beforeCustomFilter - urls.length
        } additional URLs filtered out`
      );
    }

    // Limit URLs if specified
    if (maxUrls && urls.length > maxUrls) {
      urls = urls.slice(0, maxUrls);
      logger.info(`Limited to first ${maxUrls} URLs`);
    }

    logger.info(
      `Processing ${urls.length} URLs with concurrency: ${concurrency}`
    );

    // Process URLs in batches
    const results: EmbeddingResult[] = [];

    for (let i = 0; i < urls.length; i += concurrency) {
      const batch = urls.slice(i, i + concurrency);
      logger.info(
        `Processing batch ${Math.floor(i / concurrency) + 1}/${Math.ceil(
          urls.length / concurrency
        )}`
      );

      const batchPromises = batch.map((url) =>
        scrapePageAndSaveToDatabase(url, dataSourceId, userId, botId)
      );

      const batchResults = await Promise.all(batchPromises);
      const validResults = batchResults.filter(
        (result) => result !== null
      ) as EmbeddingResult[];

      results.push(...validResults);

      // Add delay between batches to be respectful
      if (i + concurrency < urls.length) {
        await new Promise((resolve) => setTimeout(resolve, 2000));
      }
    }

    // Update data source status to completed
    await pool.query(
      "UPDATE embeddings SET metadata = COALESCE(metadata, '{}') || $1 WHERE id = $2",
      [
        JSON.stringify({
          status: "completed",
          totalUrls: urls.length,
          successfullyProcessed: results.length,
          completedAt: new Date().toISOString(),
        }),
        dataSourceId,
      ]
    );

    logger.info(
      `Completed sitemap processing. Processed: ${results.length}/${urls.length} URLs`
    );

    return {
      totalUrls: urls.length,
      successfullyProcessed: results.length,
      skippedUrls: skippedByFilters,
      results,
    };
  } catch (error) {
    // Update data source status to failed
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error";
    await pool.query(
      "UPDATE embeddings SET metadata = COALESCE(metadata, '{}') || $1 WHERE id = $2",
      [
        JSON.stringify({
          status: "failed",
          error: errorMessage,
          failedAt: new Date().toISOString(),
        }),
        dataSourceId,
      ]
    );

    logger.error(`Error processing sitemap ${sitemapUrl}:`, error);
    throw error;
  }
}

// Debug function to test sitemap parsing
async function debugSitemap(sitemapUrl: string): Promise<void> {
  console.log(`🔍 Debugging sitemap: ${sitemapUrl}`);

  try {
    const sitemapEntries = await parseSitemap(sitemapUrl);
    console.log(`📊 Total URLs found: ${sitemapEntries.length}`);

    if (sitemapEntries.length > 0) {
      console.log(`📋 First few URLs:`);
      sitemapEntries.slice(0, 5).forEach((entry, index) => {
        console.log(`  ${index + 1}. ${entry.url}`);
      });
    }

    // Test filtering
    const urls = sitemapEntries.map((entry) => entry.url);
    const filteredUrls = urls.filter(
      (url) => !shouldSkipUrl(url, DEFAULT_SKIP_WORDS, DEFAULT_SKIP_EXTENSIONS)
    );
    console.log(`🚫 URLs after filtering: ${filteredUrls.length}`);

    if (filteredUrls.length > 0) {
      console.log(`✅ First few filtered URLs:`);
      filteredUrls.slice(0, 5).forEach((url, index) => {
        console.log(`  ${index + 1}. ${url}`);
      });
    }
  } catch (error) {
    console.error(`❌ Error debugging sitemap:`, error);
  }
}

// Export functions for use in other modules
export {
  // Core scraping functions
  parseSitemap,
  loadHTML,
  extractStructuredContent,
  scrapePage,
  scrapePages,

  // Database and embedding functions
  blocksToText,
  generateEmbedding,
  saveToDatabase,
  scrapePageAndSaveToDatabase,
  scrapeSitemapAndSaveToDatabase,

  // Legacy functions for backward compatibility
  scrapeSitemapAndSaveStructuredContent,
  scrapeAndSaveStructuredContent,

  // Utility functions
  shouldSkipUrl,
  debugSitemap,
  DEFAULT_SKIP_WORDS,
  DEFAULT_SKIP_EXTENSIONS,

  // Types
  type ScrapeOptions,
  type StructuredBlock,
  type SitemapEntry,
  type ScrapedPage,
  type EmbeddingResult,
};
