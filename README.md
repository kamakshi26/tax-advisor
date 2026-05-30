# Tax Advisor CLI

A domain-constrained conversational AI built with Python and Llama 3.2 (via Ollama).

## What it does
- Answers US federal tax questions based on 2026 tax data
- Returns structured JSON with thinking, answer, quote, and confidence
- Flags irrelevant questions automatically
- Maintains full conversation history across multiple turns

## Concepts demonstrated
- Prompt engineering — system prompts, grounding, chain of thought
- Structured outputs — reliable JSON parsing from LLM responses
- Conversation memory — manual context management
- Confidence scoring — medium/low signals for uncertain answers

## Tech
- Python
- Ollama (llama3.2) — runs fully locally, no API cost
- No external dependencies except `ollama` and `python-dotenv`

## Run locally
1. Install Ollama from ollama.com
2. Run `ollama pull llama3.2`
3. Clone this repo
4. Run `pip install ollama python-dotenv`
5. Run `python3 main.py`

## Sample output
\```
You: What is the standard deduction for a single filer in 2026?

Thinking: The information block states the standard deduction amounts...
Answer:   $16,100
Quote:    For the 2026 tax year, the standard deduction amounts are: Single filers: $16,100
Confidence: high
\```
