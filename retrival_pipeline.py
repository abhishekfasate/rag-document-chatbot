from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from constants import colletion_name, persist_directory, embedding_model

class RAG_RETRIVAL:
    def __init__(self):
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = Chroma(
            collection_name=colletion_name,
            persist_directory=persist_directory,
            embedding_function=self.embedding
        )
        self.llm = ChatOllama(model="llama3.1",temperature=0.3)

    def retrival_pipeline(self, question):

        # We are passing question directly langchain_chroma will embeded the question under the hood
        relevent_docs = self.vector_store.similarity_search(question, k = 5)

        # We are just taking out the relevent docs content from relevent_docs 
        context_text = "\n\n".join(doc.page_content for doc in relevent_docs)

        return context_text

    def query(self, question, context):
        message = [
                    SystemMessage(
                        content=f"You are assistent of Abhishek Rajesh Fasate. You are expert Resume Viewer and document assistant.\n"
                                f"Answer the users question using ONLY the provided documents context below. \n" 
                                f"If you dont know the answer say I dont know. Do not make things up\n"
                                f"---Document Context--\n{context}"
                    ),
                    HumanMessage(
                        content={question}
                    )
                ]
        return message

    def call_llm(self, question):
        context = self.retrival_pipeline(question=question)
        message = self.query(question=question,context=context)

        response = self.llm.invoke(message)
        return response.content
if __name__ == "__main__":
    retrival = RAG_RETRIVAL()
    exit_prompt = ["exit", "bye", "quite", "/q"]
    print(f"Type {exit_prompt} to close chat")

    while True:
        question = input("You :")

        if not question:
            continue 

        if question.lower() in exit_prompt:
            print("Exit Trigger by User. Thank You!!")
            break

        print("\n 👀 Searching in Documents:")
        try:
            answer = retrival.call_llm(question=question)
            print(f"\nAbhishek : {answer}")

        except Exception as e:
            print(f"Error in Retrival Pipeline: {e}")


        