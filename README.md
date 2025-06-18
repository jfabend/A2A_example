# A2A (+ MCP) Example

This project demonstrates communication between agents using the **Agent-to-Agent (A2A)** protocol in combination with the **Model-Context-Protocol (MCP)**.

---

## 🔧 Components

- **database_agent.py**
  Database Agent — Counts inventory. LangGraph-based.

- **currency_agent.py**
  Currency Agent — Converts USD <-> EUR. Autogen-based.

- **host_agent.py**
  Host Agent - Delegates tasks to the other two agents. LangGraph ReAct Agent.

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Database Agent (in a 1st terminal)

```bash
python a2a_server_db_agent.py
```

The database agent starts and waits for instructions.

### 3. Run the Currency Agent (in a 2nd terminal)

```bash
python a2a_server_currency_agent.py
```

The currency agent starts and waits for instructions.

### 4. Run the host agent (in a 3rd terminal)

```bash
python host_agent.py
```

The host agent asks the user for input and communicates with the other agents via A2A.

---

## Example questions

How much is 20 Euros in USD?
How many different T-Shirts does our company sell?
