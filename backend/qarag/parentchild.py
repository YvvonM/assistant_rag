import os

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

from langchain_community.document_loaders import CSVLoader
from langchain.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever

## Text Splitting & embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain_openai.embeddings import OpenAIEmbeddings

loader = CSVLoader('/home/yvvonmajala/Documents/ten/backend/qarag/data1.csv', encoding="utf-8")
docs = loader.load()

docs[0]

import uuid
# This text splitter is used to create the parent documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2700)

# This text splitter is used to create the child documents
# It should create documents smaller than the parent
child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

# The vectorstore to use to index the child chunks
vectorstore = Chroma(
  collection_name=f"split_parents_{str(uuid.uuid4())}",
  embedding_function=OpenAIEmbeddings(),
  persist_directory="./chroma_db"
)

# The storage layer for the parent documents
store = InMemoryStore()

retriever1 = ParentDocumentRetriever(
  vectorstore=vectorstore,
  docstore=store,
  child_splitter=child_splitter,
  parent_splitter=parent_splitter,
)

def add_documents_in_batches(retriever, docs, max_batch_size=41666):
    batch = []
    current_batch_size = 0

    for doc in docs:
        doc_size = sum(len(segment) for segment in doc)

        # If an individual document is larger than the max batch size, split it further
        if doc_size > max_batch_size:
            # Create chunks of the document
            current_doc = []
            current_doc_size = 0

            for segment in doc:
                segment_size = len(segment)

                if current_doc_size + segment_size > max_batch_size:
                    # Add the current chunk to the batch
                    batch.append(current_doc)
                    retriever.add_documents(batch)
                    batch = []
                    current_doc = [segment]
                    current_doc_size = segment_size
                else:
                    current_doc.append(segment)
                    current_doc_size += segment_size

            # Add the last chunk of the document
            if current_doc:
                batch.append(current_doc)
                retriever.add_documents(batch)
                batch = []
        else:
            if current_batch_size + doc_size > max_batch_size:
                retriever.add_documents(batch)
                batch = [doc]
                current_batch_size = doc_size
            else:
                batch.append(doc)
                current_batch_size += doc_size

    if batch:
        retriever.add_documents(batch)

# Add documents in smaller batches
add_documents_in_batches(retriever1, docs)

child_retriever = vectorstore.as_retriever()

child_retriever.get_relevant_documents("Data Engineering Job")

retriever1.invoke('Data Engineering Job')

from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

chat_model = ChatOpenAI(
  model="gpt-3.5-turbo",
  verbose=True,
  callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

from langchain.prompts import PromptTemplate

# Prompt
template = """[INST] <<SYS>> Use the following pieces of context to answer the question at the end.
Be interactive with the person. Let one answer lead to another question to the person. Ask about a persons' cv skills and qualifications. Based on the skills and qualifications, recommend a job from the context provided.
You may also recommend other skills the person needs but doesn't have in order to land a certain job.
Answers should be long explaining the concept very well.

{context}
Question: {question}
Helpful Answer:[/INST
]"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

# QA chain
from langchain.chains import RetrievalQA

child_qa_chain = RetrievalQA.from_chain_type(
  chat_model,
  retriever=child_retriever,
  chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
  return_source_documents=True
)

qa_chain = RetrievalQA.from_chain_type(
  chat_model,
  retriever=retriever1,
  chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
  return_source_documents=True
)

# Retrieve and generate using the relevant snippets of the blog.

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
retriever = retriever1
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("recommend to me a software engineering role")

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

from langchain_core.messages import HumanMessage

def get_answer_from_rag_chain(question, chat_history):
    ai_msg = rag_chain.invoke({"input": question, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=question), ai_msg["answer"]])

    return ai_msg["answer"], chat_history


