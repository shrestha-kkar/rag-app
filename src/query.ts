import { CohereClient } from 'cohere-ai';
import dotenv from 'dotenv';
import OPENAI from 'openai';
import { index } from './utils/pinecone';

dotenv.config({ override: true });

const cohere = new CohereClient({
    token: process.env.COHERE_API_KEY!,
});

//AIcredits API client
const client = new OPENAI({
    apiKey: process.env.AICREDITS_API_KEY!,
    baseURL: process.env.AICREDITS_BASE_URL!,

    timeout: 60000,
});

async function askQuestion(question: string) {
    try {
        console.log(`\nQuestion: ${question}\n`);

        //generate query embedding
        const embedResponse = await cohere.embed({
            texts: [question],
            model: "embed-english-v3.0",
            inputType: "search_query",
        });

        const queryEmbeddings = embedResponse.embeddings as number[][];

        const queryVector = queryEmbeddings[0];

        //search pinecone
        // let searchResult: any;

        // for (let i = 0; i < 3; i++) {
        //     try {
        //         searchResult = await index.query({
        //             vector: queryVector,
        //             topK: 8,
        //             includeMetadata: true,
        //         });

        //         break;
        //     } catch (err) {
        //         console.log(`Retry ${i + 1}...`);
        //     }
        // }

        const searchResult = await index.query({
            vector: queryVector,
            topK: 8,
            includeMetadata: true,
        });

        //extract context
        const context = searchResult?.matches?.map((match: any) => match.metadata?.text)
            .join("\n\n") || "";

        console.log(`Retrieved Context\n${context}\n`);

        //RAG prompt

        const prompt = `
You are a helpful AI assistant.

Answer the question using the provided context.

If the answer is partially available,
infer carefully from the context.

If the answer truly does not exist,
say:
"I could not find that information."

Context:
${context}

Question:
${question}
`;

        //generate answer using AIcredits
        const completion = await client.chat.completions.create({
            model: 'openai/gpt-4o-mini',
            messages: [
                {
                    role: 'user',
                    content: prompt,
                }
            ],

            temperature: 0.2,
            max_tokens: 150,
        });

        console.log('\nFinal Answer:\n\n');

        console.log(
            completion.choices[0].message.content
        );
    } catch (error) {
        console.error('Error querying RAG:', error);
    }
}

// askQuestion("What does this website talk about?");

// cli input
const readline = require('readline').createInterface({
    input: process.stdin,
    output: process.stdout,
});

readline.question('Ask a question: ', async (question: string) => {
    await askQuestion(question);
    readline.close();
});