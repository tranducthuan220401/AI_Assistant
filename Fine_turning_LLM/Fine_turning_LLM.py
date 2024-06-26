from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms.ctransformers import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

model_dir = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "data/db_faiss"
eb_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
gpt4all_kwargs = {'allow_download': 'True'}
embedding_model  = GPT4AllEmbeddings(
    model_name=eb_name,
    gpt4all_kwargs=gpt4all_kwargs
)

def create_db_from_files(pdf_data_path):
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local("data/db_faiss")
    return db

def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01
    )
    return llm

def creat_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=["context", "question"])
    return prompt

def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = db.as_retriever(search_kwargs = {"k":4}, max_tokens_limit=1024),
        return_source_documents = False,
        chain_type_kwargs= {'prompt': prompt}

    )
    return llm_chain

def read_vectors_db():
    db = FAISS.load_local(vector_db_path, embedding_model,allow_dangerous_deserialization = True)
    return db

create_db_from_files("data")

db = read_vectors_db()
llm = load_llm(model_dir)

template = """<|im_start|>system\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
prompt = creat_prompt(template)

llm_chain  = create_qa_chain(prompt, llm, db)

def get_response(query):
    response = llm_chain.invoke({"query": query})
    
    return response['result']

if __name__ == "__main__":
    t = True
    while t:
        question = input("input:")
        if question == "bye":
            t = False
        else:
            response = llm_chain.invoke({"query": question})
            print(response['result'])