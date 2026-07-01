# SHL Assessment Recommender

A production-quality **Conversational AI Agent** that recommends SHL Individual Test Solutions through a stateful REST API powered by RAG (Retrieval-Augmented Generation) and SQLite session persistence.

## Architecture

```
POST /chat {"session_id": "...", "message": "..."}
    │
    ▼
SessionService (SQLite / sessions.db) ── [Save & Load Session History]
    │
    ▼
IntentService (Groq LLaMA-3.3-70B)
    │
    ├── CLARIFY          → ClarificationService → one targeted question
    ├── RECOMMEND/REFINE → RetrievalService (ChromaDB) → RecommendationService
    ├── COMPARE          → ComparisonService (ChromaDB lookup + LLM)
    ├── OFF_TOPIC        → RefusalService (deterministic, no LLM)
    └── PROMPT_INJECTION → RefusalService (deterministic, no LLM)
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI + Uvicorn |
| Session Database | SQLite (`sessions.db` inside persistent data dir) |
| LLM | Groq `llama-3.3-70b-versatile` |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector DB | ChromaDB (persistent) |
| Catalog | SHL Product Catalog JSON (400+ assessments) |

## Step-by-Step Setup Guide

Follow these detailed steps to set up and run the project from scratch:

### 1. Clone the Repository & Navigate to Project Root
First, ensure you are in the project root directory:
```bash
cd shl-recommendation-system
```

### 2. Create a Virtual Environment (`.venv`)
Create a Python 3.12+ virtual environment to isolate the project's dependencies:
```bash
python3 -m venv .venv
```

### 3. Activate the Virtual Environment
Activate the environment depending on your operating system:
* **Linux / macOS**:
  ```bash
  source .venv/bin/activate
  ```
* **Windows (Command Prompt)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```
* **Windows (PowerShell)**:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

Once activated, your terminal prompt should show `(.venv)`.

### 4. Install Dependencies
Ensure `pip` is upgraded, then install all required packages:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure the Environment Secret Key
Copy the example environment template to `.env`:
```bash
cp .env.example .env
```
Open the `.env` file and replace the placeholder `GROQ_API_KEY` with your actual Groq key (you can create your API keys from the [Groq Console Keys Page](https://console.groq.com/keys)):
```env
GROQ_API_KEY=gsk_your_actual_key_here
PORT=8002
```

### 6. Build the Catalog (Downloads SHL Assessment Data)
Run the catalog build script. This will download and normalize the SHL assessment collection to `app/data/catalog.json`:
```bash
python scripts/build_catalog.py
```

### 7. Build the Vector Store (Generates Embeddings)
Generate document embeddings for all assessments and store them in the Chroma vector database:
```bash
python scripts/build_vectorstore.py
```

### 8. Start the FastAPI Server
Run the FastAPI web application with hot-reloading enabled:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

The API will now be live:
* **Base URL**: http://localhost:8002
* **Swagger API Documentation**: http://localhost:8002/docs
* **ReDoc API Documentation**: http://localhost:8002/redoc

---

## API Reference

### `GET /health`

```json
{"status": "ok"}
```

### `POST /chat`

Conversational endpoint. Conversation history is persisted automatically on the server in SQLite.

**First Request (Omit `session_id` to generate a new session):**
```json
{
  "message": "I need a Java test for senior developers"
}
```

**First Response (Returns generated `session_id`):**
```json
{
  "session_id": "c37415c5-0d43-4925-b1c8-6da53aea67ee",
  "reply": "Here are my top recommendations for senior Java developers...",
  "recommendations": [
    {
      "name": "Java (Advanced Level)",
      "url": "https://www.shl.com/products/product-catalog/view/java-advanced/",
      "test_type": "Knowledge & Skills"
    }
  ],
  "end_of_conversation": false
}
```

**Subsequent Requests (Pass the received `session_id` to maintain context):**
```json
{
  "session_id": "c37415c5-0d43-4925-b1c8-6da53aea67ee",
  "message": "Also include personality tests"
}
```

---

## Supported Intents

| Intent | Trigger | Response |
|--------|---------|----------|
| `CLARIFY` | Vague or incomplete request | One targeted clarification question |
| `RECOMMEND` | Clear role/skill requirements | 1–10 ranked assessments |
| `REFINE` | Update existing recommendations | Updated ranked list |
| `COMPARE` | "Compare X vs Y" | Side-by-side comparison |
| `OFF_TOPIC` | Outside SHL scope | Polite refusal |
| `PROMPT_INJECTION` | Jailbreak attempt | Polite refusal |

---

## Example Conversations

**Vague query (starts the session):**
```
User: "I need an assessment"
Agent: "What role are you hiring for? Knowing the position will help me 
        recommend the most relevant SHL assessments."
```

**Specific query:**
```
User: "Java tests for mid-level backend developers"
Agent: "Based on your requirements, I recommend:
        1. Java (Advanced Level) — covers OOP, concurrency, JVM internals
        2. ..."
```

**Refinement (using the same session):**
```
User: "Also include personality tests"
Agent: "Updated recommendations including personality assessments: ..."
```

**Comparison:**
```
User: "Compare OPQ32 vs Verify Numerical Reasoning"
Agent: "OPQ32 is a personality questionnaire (25 min) while Verify is 
        a cognitive ability test (17 min)..."
```

---

## Project Structure

```
app/
├── api/routes.py               # FastAPI endpoints
├── services/
│   ├── agent.py                # Central orchestrator (wires SQLite + RAG)
│   ├── session_service.py      # SQLite session database manager (NEW)
│   ├── intent_service.py       # Intent classifier
│   ├── conversation_service.py # State reconstruction
│   ├── retrieval_service.py    # ChromaDB retrieval
│   ├── recommendation_service.py # RAG recommendation
│   ├── comparison_service.py   # Assessment comparison
│   ├── clarification_service.py # Question generation
│   └── refusal_service.py      # Deterministic refusals
│   └── scraper.py              # Catalog loader
├── vectorstore/
│   ├── chroma.py               # ChromaDB wrapper
│   └── embedding.py            # HuggingFace embeddings
├── prompts/
│   ├── classifier_prompt.py    # Intent classification prompt
│   ├── recommendation_prompt.py # Recommendation prompt
│   ├── comparison_prompt.py    # Comparison prompt
│   └── clarification_prompt.py # Clarification prompt
├── models/
│   ├── request_models.py       # ChatRequest (session_id, message)
│   └── response_models.py      # ChatResponse (session_id, reply, recommendations)
├── data/
│   ├── catalog.json            # SHL assessment catalog
│   ├── chroma_db/              # Persistent ChromaDB
│   └── sessions.db             # SQLite session database (NEW)
├── config.py                   # Pydantic Settings
└── main.py                     # App entry point + startup database migrations
scripts/
├── build_catalog.py            # Download + process catalog
└── build_vectorstore.py        # Build ChromaDB index
tests/
├── conftest.py                 # Shared fixtures (uses isolated temp session DB)
└── test_chat.py                # E2E multi-turn session-based test scenarios
```

## Running Tests

```bash
# Run all tests (no API key required — uses mock LLM and temp sqlite DB)
pytest tests/ -v

# Run specific test class
pytest tests/test_chat.py::TestRecommendation -v
```

## Anti-Hallucination Guarantees

- All assessment URLs are validated against ChromaDB-retrieved documents
- The LLM is never asked to generate assessment names/URLs from memory
- Any URL not in the retrieved set is rejected post-generation
- Comparison responses only use content from fetched assessment documents

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | **Required.** Your Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHROMA_DB_PATH` | `app/data/chroma_db` | ChromaDB storage path |
| `SQLITE_DB_PATH` | `app/data/sessions.db`| SQLite session database path |
| `CATALOG_PATH` | `app/data/catalog.json` | SHL catalog JSON path |
| `PORT` | `8002` | Server port |
| `RETRIEVAL_TOP_K` | `10` | Max retrieval candidates |
