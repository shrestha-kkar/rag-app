import { CohereClient } from 'cohere-ai';
import dotenv from 'dotenv';
import OPENAI from 'openai';
import { index } from './utils/pinecone';

dotenv.config();

const cohere = new CohereClient({
    token: process.env.COHERE_API_KEY!,
});

//AIcredits API client
const client = new OPENAI({
    apiKey: process.env.AICREDITS_API_KEY!,
    baseURL: process.env.AICREDITS_BASE_URL!,
});

async function askQuestion(question: string) {
    try {
        console.log(`\nQuestion: ${question}\n`);

        //generate query embedding
        const embedResponse = await cohere.embed({
            texts: [question],
            model: "embed-english-v2.0",
            inputType: "search_query",
        });

        const queryEmbeddings = embedResponse.embeddings as number[][];

        const queryVector = queryEmbeddings[0];

        //search pinecone
        const searchResult = await index.query({
            vector:queryVector,
            topK: 5,
            includeMetadata: true,
        });

        //extract context
        const context = searchResult.matches?.map((match: any) => match.metadata?.text)
        .join("\n\n");

        console.log(`Retrieved Context\n${context}\n`);

        //RAG prompt

        const prompt = `
        You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer is not present in context, say:
"I could not find that information."

Context:
${context}

Question:
${question}
`;

        //generate answer using AIcredits
        const completion = await client.chat.completions.create({
            model:'google/gemini-2.0-flash',
            messages:[
                {
                    role:'user',
                    content: prompt,
                }
            ],

            temperature: 0.2,
            max_tokens: 150,
        });

        console.log('\nFinal Answer:\n');

        console.log(
            completion.choices[0].message.content
        );
    } catch (error) {
        console.error('Error querying RAG:', error);
    }
}

askQuestion("What does this website talk about?");

//cli input
// const readline = require('readline').createInterface({
//     input: process.stdin,
//     output: process.stdout,
// });

// readline.question('Ask a question: ', async (question) => {
//     await askQuestion(question);
//     readline.close();
// });