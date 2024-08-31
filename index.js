const fs = require('fs');
const express = require('express');

const z = require('zod');

const openAI = require('openai');

const {zodResponseFormat} = require('openai/helpers/zod');


const { Pinecone } = require('@pinecone-database/pinecone');

const dotenv = require('dotenv');

dotenv.config();



const QuestionFormat = z.object({
  questions: z.array(z.string()),
})

const openai = new openAI({
  apiKey: OPENAI_API_KEY,
});


const readWikipediaText = async (isClean = false) => {
  let wikipediaText;
  if (isClean) {
    wikipediaText = fs.readFileSync('./clean-science.txt', 'utf8');
  } else {
    wikipediaText = fs.readFileSync('./science.txt', 'utf8');
  }
  return wikipediaText;
};

const cleanScienceText = async () => {
  const wikipediaText = await readWikipediaText();
  const lines = wikipediaText.split('\n');
  const cleanLines = lines.map((line) => {
    const cleanLine = line.replace(/[^a-zA-Z ]/g, "").toLocaleLowerCase();
    return cleanLine;
  });
  const cleanText = cleanLines.join('\n');
  await fs.writeFileSync('./clean-science.txt', cleanText);
};


const createChunk = async () => {
  const CHUNK_SIZE = 6;
  const wikipediaText = await readWikipediaText(true);
  const lines = wikipediaText.split('\n').filter((line) => line.length > 0)
  const chunks = [];

  for (let i = 0; i < lines.length; i += CHUNK_SIZE) {
    const chunk = lines.slice(i, i + CHUNK_SIZE);
    chunks.push(chunk.join('\n'));
  }

  return chunks;

}

const createOpenAIEmbeddings = async (chunk) => {
  const embeddings = await openai.embeddings.create({
    input: chunk,
    model: 'text-embedding-3-small',
    dimensions: 1024,
  });
  return embeddings;
};

const createEmbeddingForChunks = async () => {
  const chunks = await createChunk();
  const embeddings = [];

  for (const chunk of chunks) {
    const embedding = await createOpenAIEmbeddings(chunk);
    embeddings.push({
      chunk,
      embedding,
    });
  }


  return embeddings;
};

const uploadEmbeddings = async (embeddings, index) => {
  const embeddingsToUpload = [];
  for (let i = 0; i < embeddings.length; i += 1) {
    const embedding = embeddings[i];
    embeddingsToUpload.push({
      id: `vec_${i}`,
      values: embedding.embedding.data[0].embedding,
      metadata: {
        text: embedding.chunk,
      },
    });
  }
  console.log(embeddingsToUpload);
  await index.upsert(embeddingsToUpload);
}

const findSimilar = async (index, vector) => {
  const requestQuery = {
    vector,
    topK: 1,
    includeValues: false,
    includeMetadata: true,
  }

  const response = await index.query(requestQuery);
  return response;
};

const createOpenAIResponse = async (question, context, content) => {



  const chatCompletion = await openai.beta.chat.completions.parse({
    messages: [
      { role: 'system', content: `${content} \n ${context}` },
      { role: 'user', content: question },
    ],
    model: 'gpt-4o-2024-08-06',
    response_format: zodResponseFormat(QuestionFormat, "event"),
  });
  return chatCompletion.choices;
};




const pinecone = new Pinecone({
  apiKey: PINE_CONE_API_KEY,
});

const pineconeIndexName = PINE_CONE_INDEX;

const pineConeNameSpace = PINE_CONE_NAMESPACE;

const start = async () => {

  const index = await pinecone.index(pineconeIndexName).namespace(pineConeNameSpace);

  await cleanScienceText();
  const embeddings = await createEmbeddingForChunks();
  await uploadEmbeddings(embeddings, index);
  const question = `Generate 5 descriptive type questions for an entrance Test. Do not include options and answer. Expected Output Example: ["Question 1","Question 2", "Question 3"]`;
  const questionEmbedding = await createOpenAIEmbeddings(question);
  // console.log(questionEmbedding.data[0].embedding);
  const similarVectors = await findSimilar(index, questionEmbedding.data[0].embedding);
  // console.dir({ similarVectors }, { depth: null });
  const gptResponse = await createOpenAIResponse(question, similarVectors.matches[0].metadata.text, "Based on the provided context here give the answer to the question");
  return gptResponse[0].message.parsed;
};

// start();

// cleanScienceText();

// createEmbeddingForChunks();

// uploadEmbeddings(createEmbeddingForChunks(), );



const validate = async (question, answer) => {
  const index = await pinecone.index(pineconeIndexName).namespace(pineConeNameSpace);

  await cleanScienceText();
  const embeddings = await createEmbeddingForChunks();
  await uploadEmbeddings(embeddings, index);
  const questionPrompt = `You Being a Teacher analyze the given answer by a student and award a score between 0 and 5 to the following question. Based on the quality of the answer. Below is the Question Which you have asked the student:\n${question}\n\nAnd Below is the answer given by the Student:\n${answer}`;
  const questionEmbedding = await createOpenAIEmbeddings(questionPrompt);
  // console.log(questionEmbedding.data[0].embedding);
  const similarVectors = await findSimilar(index, questionEmbedding.data[0].embedding);
  // console.dir({ similarVectors }, { depth: null });
  const gptResponse = await createOpenAIResponse(question, similarVectors.matches[0].metadata.text);
  return gptResponse[0].message.parsed;
}



const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.get('/question', async (req, res) => {
  const response = await start();
  res.status(200).json(response);
});

app.post('/validate', async (req, res) => {
  const { answers } = req.body;
  for (let i =0; i< answers.length; i++) {
    const {question, answer} = answers[i];
    console.log(question, answer);
    const currentScore = await  validate(question, answer);
    console.log(currentScore);
  }
  res.status(200).json("Please Complete API");
});

app.listen(3005, '0.0.0.0', () => {
  console.log('Listening.');
});