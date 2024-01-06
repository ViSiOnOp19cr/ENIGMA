from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from medical_keywords import medical_keywords

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response, sources = qa_result({'query': query})
    if not sources:
        return "Hey, I am a medical chatbot. I cannot provide information outside the data I have."
    return response

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    user_input = message.content.lower()  # Convert the input to lowercase for easier keyword matching
    chain = cl.user_session.get("chain")

    # Debug print
    print("Received user input:", user_input)

    # Checks if the input contains any of the medical keywords
    if any(keyword in user_input for keyword in medical_keywords):
        print("Medical keyword found. Processing with chain.ainvoke...")

        # If medical keywords are found, proceed with the chatbot's normal response
        try:
            # Replace acall with ainvoke
            res = await chain.ainvoke(user_input)
            answer = res["result"]
            print("Received answer from chain.ainvoke:", answer)
        except Exception as e:
            print("Error during chain.ainvoke:", e)
            answer = "Sorry, an error occurred while processing your request."

    else:
        # If no medical keywords are found, respond with the standard out-of-scope message
        print("No medical keywords found. Sending standard message.")
        answer = "Hey, I am a medical chatbot. I cannot provide information on that."

    # Send the response message
    await cl.Message(content=answer).send()
    print("Response sent.")