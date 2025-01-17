import os
import logging
import json
import uuid
from typing import Dict, List, Optional, Any, Union

from datetime import datetime

# FastAPI and related imports
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    Body,
    Query,
    Depends
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv

# LangChain / RAG Pipeline Imports (placeholder imports—adjust for your project)
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from bs4 import BeautifulSoup
import requests

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends

# Supabase
from supabase import create_client, Client

###############################################################################
#                            ENV & LOGGING SETUP
###############################################################################
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment!")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

###############################################################################
#                  SUPABASE CREDENTIALS & CLIENT INITIALIZATION
###############################################################################

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

###############################################################################
#                      OPTIONAL: CREATE TABLES / SCHEMA
###############################################################################
def create_db_schema() -> None:
    """
    You can run this function ONCE in a safe admin environment to create
    the necessary tables in your Supabase Postgres database (if they do not exist).
    """
    schema_sql = """
    -- Enable UUID generation if not enabled
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

    CREATE TABLE IF NOT EXISTS public.users (
      id uuid DEFAULT uuid_generate_v4() PRIMARY KEY,
      created_at timestamp with time zone DEFAULT now(),
      email text UNIQUE NOT NULL,
      password_hash text,
      full_name text,
      last_login timestamp with time zone,
      role text DEFAULT 'user'
    );

    CREATE TABLE IF NOT EXISTS public.chats (
      chat_id uuid DEFAULT uuid_generate_v4() PRIMARY KEY,
      user_id uuid REFERENCES public.users (id) ON DELETE CASCADE,
      created_at timestamp with time zone DEFAULT now(),
      title text,
      last_updated timestamp with time zone DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS public.chat_session (
      session_id uuid DEFAULT uuid_generate_v4() PRIMARY KEY,
      chat_id uuid REFERENCES public.chats (chat_id) ON DELETE CASCADE,
      created_at timestamp with time zone DEFAULT now(),
      updated_at timestamp with time zone DEFAULT now(),
      content jsonb DEFAULT '{}'::jsonb
    );

    CREATE TABLE IF NOT EXISTS public.logs (
      log_id uuid DEFAULT uuid_generate_v4() PRIMARY KEY,
      session_id uuid REFERENCES public.chat_session (session_id) ON DELETE CASCADE,
      timestamp timestamp with time zone DEFAULT now(),
      event_type text,
      details jsonb DEFAULT '{}'::jsonb
    );

    CREATE TABLE IF NOT EXISTS public.ai_thought_table (
      id uuid DEFAULT uuid_generate_v4() PRIMARY KEY,
      created_at timestamp with time zone DEFAULT now(),
      session_id uuid REFERENCES public.chat_session (session_id) ON DELETE CASCADE,
      thought_process text,
      decision_making jsonb DEFAULT '{}'::jsonb
    );
    """

    logging.info("Schema creation SQL:\n%s", schema_sql)
    # You can run this SQL in Supabase's SQL Editor, or use an RPC if you have one:
    # supabase_admin.rpc('execute_sql', {'q': schema_sql}).execute()
    # Or manually run it in your project's SQL editor.
    pass

###############################################################################
#                               FASTAPI APP
###############################################################################
app = FastAPI(
    title="RAG-GENAI-Women",
    version="1.0.0",
    description=(
        "A production-ready pipeline with session-based JSON storage, plus "
        "auth endpoints for SignUp, Login, and more. "
        "Supports multiple concurrent WebSocket connections (one per session)."
    )
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security scheme
security = HTTPBearer()
###############################################################################
#                          LLM & VECTOR STORE SETUP
###############################################################################
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model="gpt-4o")  # Example placeholder name
llm_decision_maker = ChatOpenAI(model="gpt-4o-mini")

vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings_model
)

def get_time_date() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_country_from_ip() -> str:
    # Stub; in production, do a real IP lookup
    return "India"

###############################################################################
#                                 WEB SEARCH TOOL
###############################################################################
# @tool
# def web_search_tool(query: str) -> Dict[str, Any]:
#     """
#     Perform a web search and return a single dictionary:
#        {"results": [...], "count": <int>}
#     """
#     from googlesearch import search

#     results = []
#     try:
#         for url in search(query, num_results=3):
#             try:
#                 resp = requests.get(url, timeout=10)
#                 soup = BeautifulSoup(resp.text, "html.parser")
#                 snippet = soup.get_text()
#                 results.append({"url": url, "content": snippet})
#             except Exception as e:
#                 logging.exception("Error fetching content from: %s", url)
#                 results.append({"url": url, "content": f"Error: {str(e)}"})
#     except Exception as e:
#         logging.error("Error performing search: %s", e)

#     return {"results": results, "count": len(results)}

###############################################################################
#                                RAG PIPELINE
###############################################################################
class State(TypedDict):
    question: str
    retrieved_context: List[Document]
    demographic_context: str
    web_search_needed: int
    web_search_results: List[dict]
    final_answer: str
    tone: str

def add_demographic_context(state: State):
    country = get_country_from_ip()
    timestamp = get_time_date()
    demo = f"User from {country} at {timestamp}"
    logging.info(f"[add_demographic_context] {demo}")
    return {"demographic_context": demo}

def retrieve(state: State):
    logging.info("[retrieve] Searching vector store...")
    user_query = state["question"]
    docs = vector_store.similarity_search(user_query)
    combined_text = "\n\n".join(doc.page_content for doc in docs)

    tone = state.get("tone", "detailed")
    sys_msg = (
        "You are an assistant extracting key points. Focus on relevant details. Think like a lawyer and search for relevant details."
        if tone != "casual" else
        "You are an assistant extracting key points in a conversational manner. Think like a lawyer and search for relevant details."
    )

    prompt = [
        {"role": "system", "content": sys_msg},
        {
            "role": "user",
            "content": (
                f"Query:\n{user_query}\n\nDocs:\n{combined_text}\n"
                "Extract relevant points."
            )
        }
    ]
    resp = llm.invoke(prompt)
    extracted = resp.content.strip()

    return {"retrieved_context": [Document(page_content=extracted, metadata={"source": "filtered"})]}

def decide_web_search(state: State):
    logging.info("[decide_web_search]")
    retrieved_text = "\n\n".join(doc.page_content for doc in state["retrieved_context"])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a decision-making assistant. "
                "Respond strictly with '1' if a web search is required, or '0' if not."
            )
        },
        {
            "role": "user",
            "content": f"Question:\n{state['question']}\n\nContext:\n{retrieved_text}"
        },
    ]

    response = llm_decision_maker.invoke(messages)
    decision = response.content.strip()

    logging.info(f"[decide_web_search] LLM decision raw: {decision}")

    try:
        return {"web_search_needed": int(decision)}
    except ValueError:
        logging.error(f"Invalid decision response: {decision}")
        raise ValueError(f"Unexpected LLM response for web search decision: {decision}")

web_search_tool = TavilySearchResults(max_results=1,tavily_api_key=TAVILY_API_KEY,include_answer=True)
def perform_web_search(state):
    need_search = state.get("web_search_needed", 0)
    if need_search == 1:
        logging.info("[perform_web_search] Searching the web...")

        # Construct the query from state
        query = f"{state['question']} ({state['demographic_context']})"

        #making sure query is only string
        query = str(query)

        #calling llm to extract only what needs to be searched
        prompt = [
            {"role": "system", "content": "The following is a conversation. Extract only the relevant part of the query that needs to be searched on the web."},
            {"role": "user", "content": f"{query}"}
        ]
        resp = llm.invoke(prompt)
        query = resp.content.strip()

        logging.info(f"[perform_web_search] Query: {query}")
        # Directly assign the list of results
        structured_results = web_search_tool.invoke({"query": query})

        logging.info(f"[perform_web_search] structured_results: {structured_results}")
        summarized_results = []
        for r in structured_results:
            logging.info(f"[perform_web_search] type of r is {type(r)}")
            content = r.get('content')
            
            sum_prompt = [
                {"role": "system", "content": "Summarize the content with short citation."},
                {"role": "user", "content": f"{content}\nURL: {r['url']}"}
            ]
            sum_resp = llm.invoke(sum_prompt)
            summarized_results.append({
                "url": r.get('url'),
                "summary": sum_resp.content.strip()
            })

        return {"web_search_results": summarized_results}
    else:
        logging.info("[perform_web_search] Skipping web search...")
        return {"web_search_results": []}

def consolidate(state: State):
    logging.info("[consolidate] Generating final answer...")
    retrieved_text = "\n\n".join(doc.page_content for doc in state["retrieved_context"])
    web_data = state.get("web_search_results", [])

    sources_text = "\n".join(
        f"URL: {r['url']}\nSummary: {r['summary']}" for r in web_data
    )
    tone = state.get("tone", "detailed")
    sys_msg = (
        "You are a precise assistant. Combine context and results into a final answer."
        if tone != "casual" else
        "You are a friendly assistant. Combine context and results in a final manner."
    )

    final_prompt = [
        {"role": "system", "content": sys_msg},
        {
            "role": "user",
            "content": (
                f"Question:\n{state['question']}\n\n"
                f"Retrieved:\n{retrieved_text}\n\n"
                f"Web:\n{sources_text}\n\n"
                "Give a comprehensive final answer."
            )
        }
    ]
    resp = llm.invoke(final_prompt)
    raw_ans = resp.content.strip()

    # Summarize for chat
    summ_prompt = [
        {
            "role": "system",
            "content": "Provide a concise version of the answer, preserving key details."
        },
        {
            "role": "user",
            "content": raw_ans
        }
    ]
    s_resp = llm.invoke(summ_prompt)
    chat_ans = s_resp.content.strip()

    final = {
        "crunched_summary": chat_ans,
        "full_answer": raw_ans,
        "sources": web_data if web_data else None,
        "source_type": (
            "Web + Retrieved" if web_data and retrieved_text
            else "Web" if web_data
            else "Retrieved"
        )
    }
    return {"final_answer": final}

###############################################################################
#                         PIPELINE GRAPH BUILD
###############################################################################
graph_builder = StateGraph(State).add_sequence([
    add_demographic_context,
    retrieve,
    decide_web_search,
    perform_web_search,
    consolidate
])
graph_builder.add_edge(START, "add_demographic_context")
graph_builder.add_edge("add_demographic_context", "retrieve")
graph_builder.add_edge("retrieve", "decide_web_search")
graph_builder.add_edge("decide_web_search", "perform_web_search")
graph_builder.add_edge("perform_web_search", "consolidate")
graph_builder.add_edge("consolidate", END)

pipeline_graph = graph_builder.compile()

###############################################################################
#                         SESSION-BASED JSON STORAGE
###############################################################################
SESSIONS_DIR = "sessions_data"
os.makedirs(SESSIONS_DIR, exist_ok=True)

def generate_session_id() -> str:
    return str(uuid.uuid4())

def get_session_file(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")

def load_session_from_json(session_id: str) -> dict:
    """Load or create session data from JSON."""
    path = get_session_file(session_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        data = {
            "session_id": session_id,
            "started_at": get_time_date(),
            "messages": []
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return data

def save_session_to_json(session_data: dict):
    session_id = session_data["session_id"]
    path = get_session_file(session_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)

def append_message(session_id: str, role: str, content: str):
    data = load_session_from_json(session_id)
    data["messages"].append({
        "role": role,
        "content": content,
        "timestamp": get_time_date()
    })
    save_session_to_json(data)

###############################################################################
#                              AUTH & USER MODELS
###############################################################################
class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class SignupResponse(BaseModel):
    user_id: Optional[str]
    message: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class LoginResponse(BaseModel):
    access_token: Optional[str]
    token_type: str = "bearer"
    user_id: Optional[str]
    message: str

class LogoutResponse(BaseModel):
    message: str

class Identity(BaseModel):
    provider: str
    identity_id: str
    created_at: Union[datetime, str]
    last_sign_in_at: Union[datetime, str]

class UserProfile(BaseModel):
    user_id: str
    email: str
    full_name: Optional[str]
    role: str
    created_at: datetime
    updated_at: Optional[datetime]
    last_sign_in_at: Optional[datetime]
    email_verified: bool
    phone_verified: bool
    is_anonymous: bool
    app_metadata: Dict[str, Union[str, List[str]]]
    user_metadata: Dict[str, Union[str, bool]]
    identities: List[Identity]


###############################################################################
#                              HTTP MODELS
###############################################################################
class AskRequest(BaseModel):
    user_input: str
    tone: Optional[str] = "detailed"

class AskResponse(BaseModel):
    session_id: str
    message: str

###############################################################################
#                             HTTP AUTH ENDPOINTS
###############################################################################
@app.post("/auth/signup", response_model=SignupResponse)
def signup(payload: SignupRequest):
    """
    Sign up a new user using Supabase Auth.
    Optionally store extra info (e.g., full_name) in your custom 'users' table.
    """
    # 1) Use Supabase Auth to create the user
    try:
        result = supabase_client.auth.sign_up(
            {
                "email": payload.email,
                "password": payload.password
            }
        )
    except Exception as e:
        logging.exception("[signup] Error from Supabase Auth sign_up")
        return SignupResponse(user_id=None, message=f"Sign up failed: {str(e)}")

    if result.user is None:
        # Possibly means "Confirm email" is enabled, user needs to verify
        return SignupResponse(
            user_id=None,
            message="User created, but email confirmation required."
        )

    # 2) The user is created in supabase.auth. We can optionally store extra data
    user_id = result.user.id
    full_name = payload.full_name if payload.full_name else ""
    now = datetime.utcnow()

    # Attempt to store in our custom 'users' table
    try:
        insert_res = supabase_admin.table("users").insert({
            "id": user_id,
            "email": payload.email,
            "password_hash": "N/A (Using Supabase Auth)",
            "full_name": full_name,
            "created_at": now.isoformat(),
            "last_login": None,
            "role": "user"
        }).execute()
        logging.info("[signup] Inserted custom user record: %s", insert_res.data)
    except Exception as e:
        logging.exception("[signup] Error inserting into 'users' table")

    return SignupResponse(user_id=user_id, message="Sign up successful.")

@app.post("/auth/login", response_model=LoginResponse)
def login(payload: LoginRequest):
    """
    Log in an existing user with Supabase Auth. 
    Return the access_token, which you can store on client side for usage, 
    or rely on same-site cookies if you have it configured.
    """
    try:
        result = supabase_client.auth.sign_in_with_password(
            {
                "email": payload.email,
                "password": payload.password
            }
        )
        if result.user is None:
            return LoginResponse(
                access_token=None,
                user_id=None,
                message="Login failed: invalid credentials or user not confirmed."
            )

        user_id = result.user.id
        access_token = result.session.access_token if result.session else None

        # We can track "last_login" in our custom table:
        now = datetime.utcnow()
        try:
            supabase_admin.table("users").update({
                "last_login": now.isoformat()
            }).eq("id", user_id).execute()
        except Exception as e:
            logging.exception("[login] Error updating last_login in 'users' table")

        return LoginResponse(
            access_token=access_token,
            user_id=user_id,
            message="Login success."
        )
    except Exception as e:
        logging.exception("[login] Error from Supabase Auth sign_in_with_password")
        return LoginResponse(
            access_token=None,
            user_id=None,
            message=f"Login error: {str(e)}"
        )

@app.post("/auth/logout", response_model=LogoutResponse)
def logout():
    """
    Invalidate the user's session if you are storing it on the server 
    or using persistent session management. For token-based approach,
    you can have the client discard the token and possibly call 
    supabase_client.auth.sign_out() as well.
    """
    try:
        # This will revoke the refresh token from Supabase's perspective
        supabase_client.auth.sign_out()
        return LogoutResponse(message="Logout successful.")
    except Exception as e:
        logging.exception("[logout] Error from Supabase Auth sign_out")
        raise HTTPException(status_code=500, detail="Logout failed.")


@app.get("/auth/me", response_model=UserProfile)
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Retrieve info about the currently logged-in user.
    """
    try:
        # Extract access token from Authorization header
        access_token = credentials.credentials

        # Retrieve user details using the access token
        user_response = supabase_client.auth.get_user(access_token)
        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="User not authenticated.")

        user = user_response.user

        # Optionally fetch additional data from your custom `users` table
        res = supabase_client.table("users").select("*").eq("id", user.id).single().execute()
        record = res.data

        # Construct the UserProfile response
        return UserProfile(
            user_id=user.id,
            email=user.email,
            full_name=record.get("full_name") if record else None,
            role=user.role,
            created_at=user.created_at,
            updated_at=user.updated_at,
            last_sign_in_at=user.last_sign_in_at,
            email_verified=user.user_metadata.get("email_verified", False),
            phone_verified=user.user_metadata.get("phone_verified", False),
            is_anonymous=user.is_anonymous,
            app_metadata=user.app_metadata,
            user_metadata=user.user_metadata,
            identities=[
                Identity(
                    provider=identity.provider,
                    identity_id=identity.identity_id,
                    created_at=str(identity.created_at) if isinstance(identity.created_at, datetime) else identity.created_at,
                    last_sign_in_at=str(identity.last_sign_in_at) if isinstance(identity.last_sign_in_at, datetime) else identity.last_sign_in_at,
                )
                for identity in user.identities
            ] if user.identities else []
        )
    except Exception as e:
        logging.exception("[get_current_user] Error retrieving user info")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/auth/confirm")
def confirm_email(
    access_token: str = Query(...), 
    refresh_token: str = Query(...), 
    expires_in: int = Query(...), 
    token_type: str = Query(...)
):
    """
    Endpoint to handle confirmation links sent via email.
    """
    try:
        # Use Supabase client to retrieve and confirm the user
        result = supabase_client.auth.get_user(access_token)
        if result.user:
            return {"status": "success", "message": "Email confirmed successfully.", "user": result.user}
        else:
            raise HTTPException(status_code=400, detail="Invalid or expired confirmation link.")
    except Exception as e:
        logging.exception("[confirm_email] Error during confirmation")
        raise HTTPException(status_code=500, detail=str(e))


###############################################################################
#                             HTTP ENDPOINTS
###############################################################################
@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "Service is healthy."}

@app.post("/ask", response_model=AskResponse)
def ask_endpoint(payload: AskRequest):
    """
    Optional endpoint to create a session or store the first user message
    before switching to WebSockets.
    """
    session_id = generate_session_id()
    user_input = payload.user_input
    append_message(session_id, "user", user_input)

    return AskResponse(
        session_id=session_id,
        message="Session created. Connect via WS to continue."
    )

@app.post("/reset")
def reset_session(session_id: str = Body(..., embed=True)):
    """
    Deletes the session JSON file, effectively resetting the conversation.
    """
    path = get_session_file(session_id)
    if os.path.exists(path):
        os.remove(path)
        return {"status": "ok", "message": f"Session {session_id} reset."}
    else:
        raise HTTPException(status_code=404, detail="Session not found.")

###############################################################################
#                          WEBSOCKET CONCURRENCY
###############################################################################
class ConnectionManager:
    """
    Manages EXACTLY ONE active WebSocket per session_id.
    If a new WebSocket for the same session_id arrives, 
    it closes the old connection first. 
    """
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        # If there's already an active socket for this session, close it
        if session_id in self.active_connections:
            old_ws = self.active_connections[session_id]
            logging.info(f"[WS] Closing old connection for session {session_id} to allow new one.")
            await old_ws.close(code=4000, reason="Replaced by a new connection")

        logging.info(f"[WS] Accepting WebSocket for session: {session_id}")
        await websocket.accept()

        self.active_connections[session_id] = websocket
        logging.info(f"[WS] Session {session_id} connected. "
                     f"Total active sessions: {len(self.active_connections)}")

    def disconnect(self, session_id: str, websocket: WebSocket):
        stored_ws = self.active_connections.get(session_id)
        if stored_ws is websocket:
            del self.active_connections[session_id]
            logging.info(f"[WS] Session {session_id} disconnected. "
                         f"Remaining active sessions: {len(self.active_connections)}")

    async def send_json(self, session_id: str, data: dict):
        ws = self.active_connections.get(session_id)
        if ws is not None:
            await ws.send_json(data)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: Optional[str] = Query(None),
    tone: str = Query("detailed")
):
    """
    WebSocket endpoint. 
    - The user can pass `session_id` and `tone` as query parameters, e.g.:
        ws://localhost:8000/ws?session_id=abc-123&tone=casual
    - Or omit `session_id` to generate one automatically.
    - Each message from client must be JSON with {"user_input": "..."}.
    """
    if not session_id:
        session_id = generate_session_id()
        logging.info(f"[WS] No session_id provided. Created new: {session_id}")

    await manager.connect(session_id, websocket)

    while True:
        try:
            data = await websocket.receive_json()
            user_input = data.get("user_input", "")
            append_message(session_id, "user", user_input)

            session_data = load_session_from_json(session_id)
            conversation_text = ""
            for msg in session_data["messages"]:
                role_name = msg["role"].capitalize()
                conversation_text += f"{role_name}: {msg['content']}\n"

            chain_state = {
                "question": conversation_text,
                "tone": tone
            }

            await manager.send_json(session_id, {
                "type": "status",
                "message": "I'm starting pipeline..."
            })

            try:
                async for step_result in pipeline_graph.astream(chain_state, stream_mode="values"):
                    if "demographic_context" in step_result:
                        await manager.send_json(session_id, {
                            "type": "status",
                            "message": f"I see Demographic: {step_result['demographic_context']}"
                        })
                    if "retrieved_context" in step_result:
                        excerpt = step_result["retrieved_context"][0].page_content[:60]
                        await manager.send_json(session_id, {
                            "type": "status",
                            "message": f"I Retrieved context: {excerpt}..."
                        })
                    if "web_search_needed" in step_result:
                        await manager.send_json(session_id, {
                            "type": "status",
                            "message": f"I think Web search is {'needed' if step_result['web_search_needed'] else 'not needed'}."
                        })
                    if "web_search_results" in step_result:
                        count = len(step_result["web_search_results"])
                        await manager.send_json(session_id, {
                            "type": "status",
                            "message": f"I found {count} web search results."
                        })
                    if "final_answer" in step_result:
                        final_ans = step_result["final_answer"]
                        short_answer = final_ans["crunched_summary"]
                        append_message(session_id, "assistant", short_answer)

                        await manager.send_json(session_id, {
                            "type": "final_answer",
                            "short_answer": short_answer,
                            "full_answer": final_ans["full_answer"],
                            "sources": final_ans["sources"],
                            "source_type": final_ans["source_type"]
                        })

            except Exception as e:
                logging.exception("[WS] Error during pipeline streaming.")
                await manager.send_json(session_id, {
                    "type": "error",
                    "message": str(e)
                })

        except WebSocketDisconnect:
            logging.info(f"[WS] Client disconnected for session {session_id}")
            manager.disconnect(session_id, websocket)
            break
        except Exception as e:
            logging.exception("[WS] Error reading JSON from WebSocket.")
            await manager.send_json(session_id, {
                "type": "error",
                "message": str(e)
            })
            # Not disconnecting immediately—client may continue with valid input

###############################################################################
#                         LOCAL DEV ENTRY POINT
###############################################################################
if __name__ == "__main__":
    import uvicorn
    # Uncomment if you want to log out or run the DDL
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
