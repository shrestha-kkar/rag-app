import { CohereClient } from "cohere-ai";
import axios from "axios";
import dotenv from "dotenv";
import * as cheerio from "cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { index } from "./utils/pinecone";

dotenv.config();

const cohere = new CohereClient({
  token: process.env.COHERE_API_KEY!,
});

async function scrapeWebsite(url: string){
    const response = await axios.get(url);

    const $ = cheerio.load(response.data);

    $('script').remove();
    $('style').remove();
    $('noscript').remove();

    const text = $("body").text();
    return text
        .replace(/\s+/g, " ")
        .trim();
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
            model: "embed-english-v2.0",
            inputType: "search_document",
            embeddingTypes: ['float'],
        });

        const embeddings = embedResponse.embeddings as number[][];

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
