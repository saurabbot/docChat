import { OpenAI } from "langchain/llms/openai";
// import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { CSVLoader } from 'langchain/document_loaders/fs/csv'
import { FaissStore } from "langchain/vectorstores/faiss";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'


import * as dotenv from 'dotenv'
dotenv.config()


export const ingestDocs = async () => {
    const loader = new CSVLoader('sample_1.csv')
    const docs = await loader.load()
    console.log('docs have been loaded')

    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200
    })
    const docOutput = await textSplitter.splitDocuments(docs)
    let vectorStore = await FaissStore.fromDocuments(
        docOutput,
        new OpenAIEmbeddings(),
    )
    console.log('saving....')
    const directory = "/Users/saurabh/codebag/privateGPT/";
    await vectorStore.save(directory);
    console.log('saved!')

}
ingestDocs()