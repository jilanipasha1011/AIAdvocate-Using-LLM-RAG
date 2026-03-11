import os
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
VECTORSTORE_PATH  = "faiss_index"
CASE_MEMORY_PATH  = "case_memory_index"
GROQ_MODEL        = "llama-3.3-70b-versatile"
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"

# ── Environment Checks ────────────────────────────────────────────────────────
required_env_vars = {
    "GROQ_API_KEY":   "Groq API key is required for the LLM.",
    "TAVILY_API_KEY": "Tavily API key is required for web search.",
}
missing_vars = [f"{var}: {msg}" for var, msg in required_env_vars.items() if not os.getenv(var)]
if missing_vars:
    st.error("Missing required environment variables:\n" + "\n".join(missing_vars))
    st.stop()

# ── Initialize Groq LLM ───────────────────────────────────────────────────────
llm = ChatGroq(
    temperature=0,
    model=GROQ_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
)

# ── Load Main Legal Vectorstore ───────────────────────────────────────────────
@st.cache_resource
def load_vectorstore():
    if not os.path.exists(VECTORSTORE_PATH):
        st.error(
            f"Main legal vectorstore not found at '{VECTORSTORE_PATH}'. "
            "Please run RAG_System.py first to create it."
        )
        st.stop()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

try:
    vectorstore = load_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 4})
except Exception as e:
    st.error(f"Failed to load main vectorstore: {e}")
    st.stop()

# ── Load / Create Case Memory Vectorstore ─────────────────────────────────────
@st.cache_resource
def load_case_memory():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if os.path.exists(CASE_MEMORY_PATH):
        try:
            return FAISS.load_local(CASE_MEMORY_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.warning(f"Could not load case memory index: {e}. Creating a new one.")

    # Initialise with a dummy entry, then delete it to get an empty-but-valid index
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

def search_case_memory(query: str, k: int = 3, score_threshold: float = 1.2):
    """Return top-k similar cases within the score_threshold L2 distance."""
    try:
        results = case_memory.similarity_search_with_score(query, k=k)
        docs = [doc for doc, score in results if score <= score_threshold]
        return docs if docs else []
    except Exception:
        return []

# ── Web Search Tool ───────────────────────────────────────────────────────────
try:
    search_tool = TavilySearchResults(max_results=3, tavily_api_key=os.getenv("TAVILY_API_KEY"))
except Exception as e:
    st.error(f"Failed to initialize Tavily search tool: {e}")
    st.stop()

# ── RAG Tool ──────────────────────────────────────────────────────────────────
def retrieve_legal_docs(query: str) -> str:
    """Retrieve relevant legal documents from the main Indian law database."""
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "No relevant legal documents found in the database."
    return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

legal_rag_tool = Tool(
    name="Legal Document Retriever",
    func=retrieve_legal_docs,
    description=(
        "Retrieves relevant sections from Indian legal statutes, case law, and "
        "commentaries based on a legal query. Always use this tool first to ground "
        "your advice in Indian law."
    ),
)

# ── Case Memory Tool ──────────────────────────────────────────────────────────
def retrieve_past_cases(query: str) -> str:
    docs = search_case_memory(query)
    if not docs:
        return "No similar past cases found."
    return "\n\n".join([f"Case {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

case_memory_tool = Tool(
    name="Retrieve Past Cases",
    func=retrieve_past_cases,
    description=(
        "Retrieves relevant past legal cases based on a description of the current issue. "
        "Use this tool to understand how similar legal matters were previously handled."
    ),
)

# ── Assemble Tools ────────────────────────────────────────────────────────────
tools = [
    legal_rag_tool,
    case_memory_tool,
    Tool(
        name="Web Search",
        func=search_tool.run,
        description=(
            "Retrieves up-to-date legal information, recent court rulings, regulatory "
            "changes, and news that may not be available in the internal case database."
        ),
    ),
]

# ── System Prompt ─────────────────────────────────────────────────────────────
system_prompt = """
- You are a Supper AI Advocate like Supreme Court Advocate trained on Indian legal data and Your creater is Jilani pasha and courtroom transcripts, so your advice is deeply rooted in practical Indian legal experience, not just theoretical knowledge.

You are a senior Supreme Court of India advocate with extensive courtroom experience across Indian civil, criminal, constitutional, property, and procedural law.

You ONLY provide advice based on Indian law.
Never refer to foreign legal systems, US/UK law, or international frameworks unless specifically asked for comparison.

Your role is to think, analyze, and respond like a seasoned Supreme Court litigator defending the client's interests within the Indian legal system.

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
- Focus on protecting the client's legal position.
- Provide the fastest lawful remedy available under Indian law.
- If facts are unclear, provide conditional strategies and clearly state assumptions.
- Do not give theoretical textbook explanations.

Tone: Authoritative, structured, strategic, and practical — like a senior Supreme Court advocate arguing for the client.
"""

# ── Persist memory across Streamlit reruns ────────────────────────────────────
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = ConversationSummaryBufferMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        max_token_limit=2000,
    )
memory = st.session_state.agent_memory

# ── Initialize Agent ──────────────────────────────────────────────────────────
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    system_message=system_prompt,
    verbose=True,
    handle_parsing_errors=True,
)
agent.agent.llm_chain.prompt.messages[0].prompt.template = system_prompt


# ── Helper: detect case type from text ───────────────────────────────────────
def detect_case_type(text: str) -> str:
    """Return a rough case-type label based on keyword matching."""
    text_lower = text.lower()
    keywords = {
        "criminal":      ["fir", "arrest", "bail", "murder", "theft", "rape", "ipc", "bns", "police", "crpc", "bnss"],
        "property":      ["land", "property", "encroachment", "possession", "title", "sale deed", "registry"],
        "matrimonial":   ["divorce", "maintenance", "alimony", "custody", "marriage", "dowry", "domestic violence"],
        "constitutional":["fundamental rights", "article", "writ", "high court", "supreme court", "public interest"],
        "commercial":    ["contract", "cheque bounce", "npa", "debt", "company", "trademark", "copyright"],
        "consumer":      ["consumer", "deficiency", "refund", "product", "service complaint"],
        "labour":        ["employment", "termination", "salary", "gratuity", "pf", "epf", "labour"],
    }
    for case_type, words in keywords.items():
        if any(w in text_lower for w in words):
            return case_type
    return "general"


# ── Core: process a query and generate a response ─────────────────────────────
def process_query(prompt: str):
    """Run the agent on a prompt, display the reply, and save to memory."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    context = ""
    if "case_text" in st.session_state:
        context = f"**Case Facts:**\n{st.session_state['case_text']}\n\n"

    with st.chat_message("assistant"):
        with st.spinner("⚖️ Thinking..."):
            try:
                full_query = f"{context}**Question:** {prompt}" if context else prompt
                response   = agent.run(full_query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

                # ── Save to case memory ──────────────────────────────────────
                combined_text    = (st.session_state.get("case_text", "") + " " + prompt + " " + response)
                case_type        = detect_case_type(combined_text)
                facts_snippet    = st.session_state.get("case_text", "No file uploaded")[:1000]
                structured_summary = (
                    f"[Case Type: {case_type.upper()}]\n"
                    f"Facts: {facts_snippet}\n"
                    f"User Query: {prompt}\n"
                    f"Legal Advice: {response[:800]}"
                )
                if len(response.strip()) > 40:
                    metadata = {
                        "timestamp":     str(datetime.now()),
                        "user_query":    prompt,
                        "case_type":     case_type,
                        "has_case_file": bool(st.session_state.get("case_text")),
                    }
                    add_to_case_memory(structured_summary, metadata)

            except Exception as e:
                error_msg = f"Sorry, an error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


# ══════════════════════════════════════════════════════════════════════════════
#  Streamlit UI
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="MyAIAdvocate", page_icon="⚖️", layout="wide")
st.title("⚖️ MyAIAdvocate — Your Personal Advocate for Indian Law")
st.markdown("Type your legal question below. I'll retrieve relevant Indian laws and suggest strategies.")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # 1. File Upload
    st.header("📁 Upload Case File")
    uploaded_file = st.file_uploader(
        "Upload a .txt or .pdf file describing your case", type=["txt", "pdf"]
    )
    if uploaded_file is not None:
        try:
            if uploaded_file.type == "application/pdf":
                try:
                    from pypdf import PdfReader
                except ImportError:
                    st.error("Install pypdf: pip install pypdf")
                    st.stop()
                reader    = PdfReader(uploaded_file)
                case_text = "".join([page.extract_text() for page in reader.pages])
            else:
                case_text = uploaded_file.read().decode("utf-8")
            st.session_state["case_text"] = case_text
            st.success("✅ File uploaded successfully.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    st.divider()

    # 2. Memory Panel
    st.header("🧠 Memory Status")
    mem = st.session_state.get("agent_memory")
    if mem:
        recent_msgs = len(mem.chat_memory.messages)
        has_summary = bool(getattr(mem, "moving_summary_buffer", ""))
        st.metric("Conversation turns stored", recent_msgs // 2)
        if has_summary:
            st.success("📝 Old turns summarised to save space")
            with st.expander("View running summary"):
                st.caption(mem.moving_summary_buffer)
        else:
            st.info("No summary yet — conversation is still short.")
    else:
        st.info("Memory not initialised yet.")

    try:
        stored_cases = case_memory.index.ntotal
        st.metric("Past cases stored", stored_cases)
    except Exception:
        pass

    st.divider()

    # 3. Clear Memory
    st.header("🗑️ Clear Memory")
    if st.button("Clear Conversation Memory", use_container_width=True):
        st.session_state.pop("agent_memory", None)
        st.session_state.pop("messages",     None)
        st.session_state.pop("case_text",    None)
        st.success("Memory cleared! Reloading...")
        st.rerun()

    st.divider()
    st.info("⚖️ General legal information only. Not a substitute for professional legal advice.")


# ── Chat History ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ── Text Input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask your legal question..."):
    process_query(prompt)