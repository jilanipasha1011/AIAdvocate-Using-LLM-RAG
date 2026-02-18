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

#  Configuration 
VECTORSTORE_PATH = "faiss_index"
CASE_MEMORY_PATH = "case_memory_index"
GROQ_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Environment Checks 
required_env_vars = {
    "GROQ_API_KEY": "Groq API key is required for the LLM.",
    "TAVILY_API_KEY": "Tavily API key is required for web search."
}
missing_vars = []
for var, msg in required_env_vars.items():
    if not os.getenv(var):
        missing_vars.append(f"{var}: {msg}")
if missing_vars:
    st.error("Missing required environment variables:\n" + "\n".join(missing_vars))
    st.stop()

# Initialize Groq LLM
llm = ChatGroq(temperature=0, model=GROQ_MODEL, api_key=os.getenv("GROQ_API_KEY"))

# Load Main Legal Vectorstore 
@st.cache_resource
def load_vectorstore():
    if not os.path.exists(VECTORSTORE_PATH):
        st.error(f"Main legal vectorstore not found at '{VECTORSTORE_PATH}'. Please run RAG_System.py first to create it.")
        st.stop()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

try:
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
except Exception as e:
    st.error(f"Failed to load main vectorstore: {e}")
    st.stop()

# Load / Create Case Memory Vectorstore 
@st.cache_resource
def load_case_memory():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if os.path.exists(CASE_MEMORY_PATH):
        try:
            case_memory = FAISS.load_local(CASE_MEMORY_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.warning(f"Could not load case memory index: {e}. Creating a new one.")
            case_memory = FAISS.from_texts(["dummy"], embeddings)
            # Remove the dummy entry so the index is empty but initialized
            dummy_id = list(case_memory.index_to_docstore_id.values())[0]
            case_memory.delete([dummy_id])
            case_memory.save_local(CASE_MEMORY_PATH)
    else:
        # Initialize with a dummy and then delete it to get an empty index
        case_memory = FAISS.from_texts(["dummy"], embeddings)
        dummy_id = list(case_memory.index_to_docstore_id.values())[0]
        case_memory.delete([dummy_id])
        case_memory.save_local(CASE_MEMORY_PATH)
    return case_memory

case_memory = load_case_memory()

def add_to_case_memory(summary: str, metadata: dict):
    global case_memory
    case_memory.add_texts([summary], metadatas=[metadata])
    case_memory.save_local(CASE_MEMORY_PATH)

def search_case_memory(query: str, k: int = 3):
    docs = case_memory.similarity_search(query, k=k)
    return docs

# Web Search Tool 
try:
    search_tool = TavilySearchResults(max_results=3, tavily_api_key=os.getenv("TAVILY_API_KEY"))
except Exception as e:
    st.error(f"Failed to initialize Tavily search tool: {e}")
    st.stop()

# RAG Tool (Main Legal Document Retriever) 
def retrieve_legal_docs(query: str) -> str:
    """Retrieve relevant legal documents from the main Indian law database."""
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "No relevant legal documents found in the database."
    return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

legal_rag_tool = Tool(
    name="Legal Document Retriever",
    func=retrieve_legal_docs,
    description="Retrieves relevant sections from Indian legal statutes, case law, and commentaries based on a legal query. Always use this tool first to ground your advice in Indian law."
)

# Case Memory Tool 
def retrieve_past_cases(query: str) -> str:
    docs = search_case_memory(query)
    if not docs:
        return "No similar past cases found."
    return "\n\n".join([f"Case {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

case_memory_tool = Tool(
    name="Retrieve Past Cases",
    func=retrieve_past_cases,
    description="Retrieves relevant past legal cases based on a description of the current issue. Use this tool to understand how similar legal matters were previously handled and resolved."
)

# Assemble Tools 
tools = [
    legal_rag_tool,          # Now the agent can query the main vectorstore
    case_memory_tool,
    Tool(
        name="Web Search",
        func=search_tool.run,
        description="Retrieves up-to-date legal information, recent court rulings, regulatory changes, and news that may not be available in the internal case database."
    )
]

# System Prompt (Master Prompt)
system_prompt = """
- You are a Supper AI Advocate like Supreme Court Advocate trained on Indian legal data and Your creater is Jilani pasha and courtroom transcripts, so your advice is deeply rooted in practical Indian legal experience, not just theoretical knowledge.

You are a senior Supreme Court of India advocate with extensive courtroom experience across Indian civil, criminal, constitutional, property, and procedural law.

You ONLY provide advice based on Indian law.
Never refer to foreign legal systems, US/UK law, or international frameworks unless specifically asked for comparison.

Your role is to think, analyze, and respond like a seasoned Supreme Court litigator defending the client’s interests within the Indian legal system.

----------------------------------------------------
SAFETY RULE
----------------------------------------------------
If a user admits committing a crime and asks how to avoid arrest, destroy evidence, evade police, or escape legal consequences:
- Refuse to assist in wrongdoing.
- Do not provide evasion tactics.
- Encourage lawful resolution.
- Advise consulting a qualified advocate immediately.
- Promote accountability and legal compliance.

----------------------------------------------------
MANDATORY ANALYSIS FRAMEWORK
----------------------------------------------------

STEP 1: FACT SUMMARY
- Clearly summarize the facts provided.
- Identify parties involved.
- Identify state and jurisdiction (India-specific).
- Identify timeline and key events.
- Identify nature of dispute (civil, criminal, constitutional, property, matrimonial, commercial, etc.).

STEP 2: LEGAL CLASSIFICATION (INDIAN LAW ONLY)
- Civil, criminal, quasi-criminal, or constitutional?
- Risk of FIR under CrPC?
- Risk of arrest or need for anticipatory bail?
- Urgent need for stay, injunction, or writ?

STEP 3: APPLICABLE INDIAN STATUTES
Cite and explain relevant Indian laws such as:
- Bharatiya Nyaya Sanhita (BNS) / IPC (where applicable)
- Bharatiya Nagarik Suraksha Sanhita (BNSS) / CrPC
- CPC
- Indian Evidence Act
- Constitution of India
- State-specific laws (if relevant)

Do NOT just list sections.
Explain how the provisions apply directly to the given facts.

STEP 4: STRATEGIC LEGAL PLAN
Provide a practical and court-ready action plan:
- Immediate steps
- Correct legal remedy (FIR, anticipatory bail, quashing under Section 482, writ petition under Article 226/32, civil suit, injunction, etc.)
- Required documents
- Jurisdiction strategy (trial court, High Court, Supreme Court)

STEP 5: RISKS & DEFENSIVE STRATEGY
- Identify legal vulnerabilities.
- Mention limitation issues.
- Highlight procedural mistakes to avoid.
- Provide defense strategy if user is accused.
- Provide prosecution strategy if user is complainant.

----------------------------------------------------
IMPORTANT INSTRUCTIONS
----------------------------------------------------
- Always tailor advice to the exact facts given.
- Use structured headings.
- Be strategic, realistic, and courtroom-oriented.
- Focus on protecting the client’s legal position.
- Provide the fastest lawful remedy available under Indian law.
- If facts are unclear, provide conditional strategies and clearly state assumptions.
- Do not give theoretical textbook explanations.

Tone: Authoritative, structured, strategic, and practical — like a senior Supreme Court advocate arguing for the client.
"""

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the agent with tools and system message
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    system_message=system_prompt,  # This sets the initial system message
    verbose=True,
    handle_parsing_errors=True
)

# Force the system prompt to be the first message in the prompt template
# This ensures it's always used, even if memory overwrites it.
agent.agent.llm_chain.prompt.messages[0].prompt.template = system_prompt

# Streamlit UI 
st.set_page_config(page_title="MyAIAdvocate", page_icon="⚖️")
st.title("⚖️ MyAIAdvocate - Your Personal Legal Advisor")
st.markdown("Upload your case file (optional) and ask your legal questions. I'll retrieve relevant Indian laws and suggest strategies.")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Case File")
    uploaded_file = st.file_uploader("Upload a text or PDF file describing your case", type=["txt", "pdf"])
    if uploaded_file is not None:
        try:
            if uploaded_file.type == "application/pdf":
                # Import pypdf only when needed
                try:
                    from pypdf import PdfReader
                except ImportError:
                    st.error("pypdf library is required to read PDF files. Please install it with: pip install pypdf")
                    st.stop()
                reader = PdfReader(uploaded_file)
                case_text = "".join([page.extract_text() for page in reader.pages])
            else:
                case_text = uploaded_file.read().decode("utf-8")
            st.session_state["case_text"] = case_text
            st.success("File uploaded. You can now ask questions about this case.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    
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
                        "case_type": "unknown"
                    }
                    add_to_case_memory(summary, metadata)
            except Exception as e:
                error_msg = f"Sorry, an error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})