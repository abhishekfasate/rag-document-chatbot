from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import shutil
import pypdf
from constants import embedding_model, persist_directory, colletion_name, file_path
class RAG_INGETION:

    def __init__(self,path):
        self.folder_path = path
        self.splitter =RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model)


    def read_and_chunk(self):
        all_chunks = []
        all_metadata = []

        # Itration Over all files
        for filename in os.listdir(self.folder_path):
            filename_path = os.path.join(self.folder_path,filename)

            # This will read text file only and information realated to user for question and answers
            if os.path.isfile(filename_path) and filename.endswith("txt"):
                with open(filename_path,"r") as file:
                    content = file.read()

                chunks = self.splitter.split_text(content)
                all_chunks.extend(chunks)
                all_metadata.extend([{"source": filename}] * len(chunks))

            # This will read Resume only or whatever the data in .pdf file and chunk that data
            elif os.path.isfile(filename_path) and filename.endswith("pdf"):
                reader = pypdf.PdfReader(filename_path)
                pdf_text = []

                for page_number in range(len(reader.pages)):
                    page_data = reader.pages[page_number]
                    text = page_data.extract_text()
                    pdf_text.append(text)

                content = "\n".join(pdf_text)

                chunks = self.splitter.split_text(content)
                all_chunks.extend(chunks)
                all_metadata.extend([{"source": filename}] * len(chunks))

        # with open("chunk_data_review.txt", "w") as file:   
                #     for data in all_chunks:
                #         file.writelines(str(data))
                #         file.write("\n\n")

        return all_chunks, all_metadata

    def store_in_db(self, all_chunks, all_metadata):
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
            print(f"deleting existing database {persist_directory} to prevent duplication.")

        vector_store = Chroma.from_texts(
            texts= all_chunks,
            metadatas= all_metadata,
            collection_name= colletion_name,
            embedding= self.embedding,
            persist_directory= persist_directory
        )

        print("Data Added in Chroma Vector DB...")
        return vector_store
    
# Object Creation
ingestion_pipeline = RAG_INGETION(file_path)

# Break the File in to chunks
chunks, metadata = ingestion_pipeline.read_and_chunk()

# Chunks Converted to vector
vector_db = ingestion_pipeline.store_in_db(chunks, metadata)
