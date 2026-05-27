import { CohereClient } from 'cohere-ai';
import axios from 'axios';
import dotenv from 'dotenv';
import * as cheerio from 'cheerio';

import { RecursiveCharacterTextSplitter }
    from '@langchain/textsplitters';

import { index } from './utils/pinecone';

dotenv.config({ override: true });

const cohere = new CohereClient({
    token: process.env.COHERE_API_KEY!,
});

const MAX_PAGES = 30;

async function scrapeWebsite(
    startUrl: string
): Promise<string> {

    const visited = new Set<string>();

    const texts: string[] = [];

    async function crawl(
        url: string,
        depth = 0
    ) {

        // limits
        if (
            visited.has(url) ||
            depth > 1 ||
            visited.size >= MAX_PAGES
        ) {
            return;
        }

        visited.add(url);

        try {

            console.log(`Crawling: ${url}`);

            const response = await axios.get(url, {
                headers: {
                    'User-Agent':
                        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',

                    Accept:
                        'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',

                    'Accept-Language':
                        'en-US,en;q=0.5',

                    Connection: 'keep-alive',
                },

                timeout: 30000,
            });

            const $ = cheerio.load(response.data);

            // remove junk
            $('script').remove();
            $('style').remove();
            $('noscript').remove();
            $('svg').remove();
            $('img').remove();

            // extract clean text
            const pageText = $(
                'p, h1, h2, h3, h4, li, title'
            )
                .map((_, el) => $(el).text())
                .get()
                .join(' ')
                .replace(/\s+/g, ' ')
                .trim();

            texts.push(pageText);

            // get links
            const links = $('a')
                .map((_, el) => $(el).attr('href'))
                .get();

            for (const link of links) {

                if (!link) continue;

                try {

                    const absoluteUrl = new URL(
                        link,
                        url
                    ).href;

                    const startDomain =
                        new URL(startUrl).hostname;

                    const currentDomain =
                        new URL(absoluteUrl).hostname;

                    // skip junk urls
                    if (
                        absoluteUrl.includes('#') ||
                        absoluteUrl.includes('.pdf') ||
                        absoluteUrl.includes('.jpg') ||
                        absoluteUrl.includes('.png') ||
                        absoluteUrl.includes('.jpeg') ||
                        absoluteUrl.includes('.svg') ||
                        absoluteUrl.includes('calendar') ||
                        absoluteUrl.includes('event') ||
                        absoluteUrl.includes('?')
                    ) {
                        continue;
                    }

                    // internal links only
                    if (
                        currentDomain === startDomain
                    ) {
                        await crawl(
                            absoluteUrl,
                            depth + 1
                        );
                    }

                } catch {
                    continue;
                }
            }

        } catch (error) {

            console.log(
                `Failed to crawl: ${url}`
            );
        }
    }

    await crawl(startUrl);

    return texts.join('\n\n');
}

async function ingestWebsite(
    url: string
) {

    try {

        console.log(`\nScraping: ${url}\n`);

        const websiteText =
            await scrapeWebsite(url);

        console.log(
            `Website scraped successfully`
        );

        // split into chunks
        const splitter =
            new RecursiveCharacterTextSplitter({
                chunkSize: 500,
                chunkOverlap: 100,
            });

        const chunks =
            await splitter.splitText(
                websiteText
            );

        console.log(
            `Created ${chunks.length} chunks`
        );

        // generate embeddings in batches
        const embeddings: number[][] = [];

        const batchSize = 96;

        for (
            let i = 0;
            i < chunks.length;
            i += batchSize
        ) {

            const batch = chunks.slice(
                i,
                i + batchSize
            );

            console.log(
                `Embedding batch ${i / batchSize + 1
                }`
            );

            const embedResponse =
                await cohere.embed({
                    texts: batch,

                    model:
                        'embed-english-v3.0',

                    inputType:
                        'search_document',

                    embeddingTypes: ['float'],
                });

            const batchEmbeddings =
                (embedResponse.embeddings as any)
                    .float as number[][];

            embeddings.push(
                ...batchEmbeddings
            );

            // wait 15 seconds
            await new Promise((resolve) =>
                setTimeout(resolve, 15000)
            );
        }

        // create pinecone vectors
        const vectors = chunks.map(
            (chunk, i) => ({
                id:
                    `chunk-${Date.now()}-${i}`,

                values: embeddings[i],

                metadata: {
                    text: chunk,
                    source: url,
                    chunkNumber: i,
                },
            })
        );

        // upload to pinecone in batches
        const pineconeBatchSize = 100;

        for (
            let i = 0;
            i < vectors.length;
            i += pineconeBatchSize
        ) {

            const batch = vectors.slice(
                i,
                i + pineconeBatchSize
            );

            console.log(
                `Uploading batch ${i / pineconeBatchSize + 1
                }`
            );

            await index.upsert({
                records: batch,
            });
        }

        console.log(
            `Ingested ${vectors.length} vectors into Pinecone`
        );

    } catch (error) {

        console.error(
            'Error occurred while ingesting website:',
            error
        );
    }
}

ingestWebsite(
    'https://www.daiict.ac.in/'
);