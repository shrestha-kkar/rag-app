import { CohereClient } from "cohere-ai";
import axios from "axios";
import dotenv from "dotenv";
import * as cheerio from "cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { index } from "./utils/pinecone";

dotenv.config({ override: true });

const cohere = new CohereClient({
  token: process.env.COHERE_API_KEY!,
});

async function scrapeWebsite(url: string): Promise<string>{
    const response = await axios.get(url, {
  headers: {
    'User-Agent':
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',
    Accept:
      'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    Connection: 'keep-alive',
  },

  timeout: 30000,
});

    const $ = cheerio.load(response.data);

    $('script').remove();
    $('style').remove();
    $('noscript').remove();

    const text = $('p, h1, h2, h3, li, title')
  .map((_, el) => $(el).text())
  .get()
  .join(' ')
  .replace(/\s+/g, ' ')
  .trim();

    return text;
}

async function ingestWebsite(url: string) {
    try {
        console.log(`\nScraping: ${url}\n`);

        const websiteText = await scrapeWebsite(url);

        console.log(`Website scraped successfully`);

        //split into chunks

        const splitter = new RecursiveCharacterTextSplitter({
            chunkSize: 500,
            chunkOverlap: 100,
        });

        const chunks = await splitter.splitText(websiteText);

        console.log(`Created ${chunks.length} chunks`);

        //generate embeddings
        const embedResponse = await cohere.embed({
            texts: chunks,
            model: "embed-english-v3.0",
            inputType: "search_document",
            embeddingTypes: ['float'],
        });

        const embeddings =
  (embedResponse.embeddings as any).float as number[][];

        //Create pinecone vectors
        const vectors = chunks.map((chunk,i) => ({
            id: `chunk-${Date.now()}-${i}`,
            values: embeddings[i],
            metadata: {
                text: chunk,
                source: url,
                chunkNumber: i,
            },
        }));

        //upload vectors to pinecone
        await index.upsert({
            records: vectors,
        });

        console.log(`Ingested ${vectors.length} vectors into Pinecone`);
    } catch (error) {
        console.error("Error occurred while ingesting website:", error);
    }
}

ingestWebsite('https://www.daiict.ac.in/');
