# P1 — Prompt Engineering Lab

## Overview
This project demonstrates how different prompt engineering techniques affect LLM output quality. The system extracts structured information from customer support emails using different prompting strategies and compares their performance.

## Techniques Compared
- Zero-shot prompting
- Few-shot prompting
- Structured output (Pydantic)

## Tech Stack
- Python
- Streamlit
- LangChain
- Ollama (llama3.1:8b)
- Pydantic

## Architecture
Email → Prompt Template → LLM → Structured Output → UI

## How to Run
pip install -r requirements.txt
ollama run llama3.1:8b
streamlit run app.py

## Example Use Case
Customer support email classification and information extraction.

## Key Learning
Structured output with schema enforcement makes LLM systems reliable and production-ready.