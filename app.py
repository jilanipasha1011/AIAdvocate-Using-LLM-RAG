# Add imports at top
import os
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# ------------------ Configuration ------------------
VECTORSTORE_PATH = "faiss_index"
CASE_MEMORY_PATH = "case_memory_index"
GROQ_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Initialize Groq LLM
llm = ChatGroq(temperature=0, model=GROQ_MODEL, api_key=os.getenv("GROQ_API_KEY"))


# ------------------ Load Main Legal Vectorstore ------------------
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ------------------ Load / Create Case Memory Vectorstore ------------------
@st.cache_resource
def load_case_memory():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if os.path.exists(CASE_MEMORY_PATH):
        case_memory = FAISS.load_local(CASE_MEMORY_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        case_memory = FAISS.from_texts(["dummy"], embeddings)
        case_memory.delete([case_memory.index_to_docstore_id[0]])
    return case_memory

case_memory = load_case_memory()

def add_to_case_memory(summary: str, metadata: dict):
    case_memory.add_texts([summary], metadatas=[metadata])
    case_memory.save_local(CASE_MEMORY_PATH)

def search_case_memory(query: str, k: int = 3):
    docs = case_memory.similarity_search(query, k=k)
    return docs

# ------------------ Web Search Tool ------------------
search_tool = TavilySearchResults(max_results=3, tavily_api_key=os.getenv("TAVILY_API_KEY"))

# ------------------ Case Memory Tool ------------------
def retrieve_past_cases(query: str) -> str:
    docs = search_case_memory(query)
    if not docs:
        return "No similar past cases found."
    return "\n\n".join([f"Case {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

case_memory_tool = Tool(
    name="Retrieve Past Cases",
    func=retrieve_past_cases,
    description="Useful when you need to recall how similar legal issues were handled in the past. Input should be a description of the current legal problem."
)

tools = [
    Tool(name="Web Search", func=search_tool.run, description="Useful for recent legal updates, news, or information not found in the local law database."),
    case_memory_tool
]

# ------------------ RAG Chain ------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    verbose=True,
)

# ------------------ System Prompt (Master Prompt) ------------------
system_prompt = """You are Supreme Court advocate an experienced Indian legal advisor with practical courtroom knowledge.

You specialize in Indian civil, criminal, property, and procedural law.

When a user provides a case or legal question:
If user admits committing a crime and asks how to avoid consequences, do not provide evasion strategies.
Instead:

Refuse to assist with wrongdoing

Encourage lawful resolution

Suggest consulting a lawyer for rights clarification

Promote accountability

STEP 1: Understand and summarize the facts clearly.
- Identify parties involved
- Identify location (jurisdiction)
- Identify dates and key events
- Identify type of dispute (civil, criminal, property, family, etc.)

STEP 2: Classify the legal nature of the issue:
- Is it civil, criminal, or both?
- Is urgent relief required?
- Is there risk of FIR, arrest, or injunction?

STEP 3: Provide applicable Indian legal provisions:
- Cite relevant IPC, CrPC, CPC, or other Acts where applicable.
- Explain sections in simple language.
- Do not just list sections — explain how they apply to THIS case.

STEP 4: Give practical step-by-step action plan:
- Immediate actions
- Legal remedies available
- Documents required
- Whether FIR, legal notice, stay order, injunction, or police protection is appropriate

STEP 5: Highlight legal risks and mistakes to avoid.

STEP 6: Provide strategic advice based on situation:
- If user is complainant
- If user is accused
- If matter is sub judice

IMPORTANT:
- Never give generic productivity or case management advice.
- Always tailor response to the specific facts given.
- Use structured headings.
- Be practical and realistic, not theoretical.
- If recent law updates are required, use web search.
- If statutory interpretation is needed, use RAG database first.
- Provide the strongest possible legal strategy in favor of the client. 
- Think and respond like a senior Supreme Court advocate defending the client’s
- interests. Always prioritize protecting the client’s 
- legal position and suggest the fastest lawful way to resolve, weaken,
- or exit the case. Provide winning strategies based on Indian law, 
- procedural safeguards, and available remedies. If any detail is unclear, 
- give conditional strategic options rather than guessing, 
- and clearly mention assumptions while still focusing on the client’s defense.

Tone:
Professional, clear, structured, practical, and solution-focused.

"""  # (full text from Step 1)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    system_message=system_prompt,
    verbose=True,
    handle_parsing_errors=True
)

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="MyAIAdvocate", page_icon="⚖️")
st.title("⚖️ MyAIAdvocate - Your Personal Legal Advisor")
st.markdown("Upload your case file (optional) and ask your legal questions. I'll retrieve relevant Indian laws and suggest strategies.")

# Sidebar for file upload (same as before)
with st.sidebar:
    st.header("Upload Case File")
    uploaded_file = st.file_uploader("Upload a text or PDF file describing your case", type=["txt", "pdf"])
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            from pypdf import PdfReader
            reader = PdfReader(uploaded_file)
            case_text = "".join([page.extract_text() for page in reader.pages])
        else:
            case_text = uploaded_file.read().decode("utf-8")
        st.session_state["case_text"] = case_text
        st.success("File uploaded. You can now ask questions about this case.")
    
    st.divider()
    st.info("This tool provides general legal information based on Indian law. Not a substitute for professional legal advice.")

# Main chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your legal question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    context = ""
    if "case_text" in st.session_state:
        context = f"**Case Facts:**\n{st.session_state['case_text']}\n\n"

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                full_query = f"{context}**Question:** {prompt}" if context else prompt
                response = agent.run(full_query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

                # Automatically store case summary if facts were provided
                if context:
                    summary = f"Facts: {st.session_state['case_text'][:500]}...\nQuery: {prompt}\nAdvice: {response[:500]}..."
                    metadata = {
                        "timestamp": str(datetime.now()),
                        "user_query": prompt,
                        "case_type": "unknown"  # you could add classification later
                    }
                    add_to_case_memory(summary, metadata)
            except Exception as e:
                error_msg = f"Sorry, an error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})