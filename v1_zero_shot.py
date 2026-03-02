 # v1_zero_shot.py
#
# P1 Prompt Engineering Lab — Version 1: Zero Shot Baseline
#
# What this script does:
#   1. Loads 5 test customer emails
#   2. Tests 3 different prompts against all 5 emails
#   3. Scores each result: valid JSON? all fields? urgency valid?
#   4. Prints a comparison table
#   5. Saves full results to outputs/v1_results.json
#
# Key LangChain concept introduced here: LCEL (LangChain Expression Language)
# The pipe operator | chains: prompt | model | (we parse output manually here)
# In v3 we add the output parser to the chain too

import json
import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from src.prompts import ALL_PROMPTS
from src.evaluator import score_response, summarize_prompt_results, EvalResult

# ─────────────────────────────────────────────────────────────────────────
# LOAD CONFIGURATION FROM .env FILE
#
# load_dotenv() reads the .env file and makes its values available
# via os.getenv(). This is the industry standard way to handle config.
# You never hardcode model names, URLs, or API keys in Python files.
# ─────────────────────────────────────────────────────────────────────────

load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1:8b")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

# ─────────────────────────────────────────────────────────────────────────
# RICH CONSOLE
#
# rich is a library for beautiful terminal output.
# Console() is the main object we use to print styled text and tables.
# In real projects, readable logs save hours of debugging time.
# ─────────────────────────────────────────────────────────────────────────

console = Console()

# ─────────────────────────────────────────────────────────────────────────
# TEST DATA — 5 emails covering different scenarios
# ─────────────────────────────────────────────────────────────────────────

TEST_EMAILS = [
    {
        "id": 1,
        "email": "Hi, I bought the UltraBoost X shoes last week, order #84521, and the left shoe sole is coming apart already. This is really frustrating. I need a replacement ASAP."
    },
    {
        "id": 2,
        "email": "Hello, just checking if my refund for order 77203 has been processed? I returned the blue jacket 2 weeks ago and still nothing in my account. Not urgent just want to know."
    },
    {
        "id": 3,
        "email": "WORST EXPERIENCE EVER. Package never arrived, order #99012, been waiting 3 weeks. I want my money back immediately or I am disputing with my bank."
    },
    {
        "id": 4,
        "email": "Hey can you help me find a gift for my mom? She likes gardening. Budget around $50. No order yet just browsing."
    },
    {
        "id": 5,
        "email": "My password reset email is not coming through. Email is jane@example.com. Not super urgent but would like to fix it today."
    }
]


# ─────────────────────────────────────────────────────────────────────────
# INITIALISE THE MODEL
#
# ChatOllama is LangChain's connector for Ollama.
# We create it ONCE and reuse it for every call.
#
# temperature=0  → deterministic output (same input = same output always)
#                  perfect for extraction tasks
#
# This is the ONLY place we mention Ollama or the model name.
# All the prompt and evaluation code does not know or care which
# model is running. That is good design — swap model by changing .env
# ─────────────────────────────────────────────────────────────────────────

llm = ChatOllama(
    model       = MODEL_NAME,
    base_url    = BASE_URL,
    temperature = TEMPERATURE
)


# ─────────────────────────────────────────────────────────────────────────
# THE CHAIN — LangChain's most important concept
#
# In LangChain, you build pipelines using the | (pipe) operator.
# This is called LCEL: LangChain Expression Language.
#
# prompt | llm  means:
#   1. Take the prompt template
#   2. Fill in the variables (like {email})
#   3. Send the filled prompt to the LLM
#   4. Get back the response
#
# The result of (prompt | llm).invoke({"email": some_text})
# is an AIMessage object. We get the text with .content
#
# Why is this better than calling the model directly?
# Because you can add more steps to the chain:
#   prompt | llm | output_parser    (we add this in v3)
#   prompt | llm | output_parser | validator  (even more steps later)
# Each step is composable. Swap any part without touching others.
# ─────────────────────────────────────────────────────────────────────────

def run_one_email(
    prompt_name: str,
    prompt_template,
    email_id: int,
    email_text: str
) -> EvalResult:
    """
    Runs one email through one prompt and returns a scored EvalResult.
    
    The chain: prompt_template | llm
    - prompt_template.format fills in {email}
    - llm sends it to Ollama and gets back a response
    - .content extracts the text from the response object
    - score_response evaluates the text
    """
    
    # Build the chain for this prompt
    # The | operator connects prompt → model into one pipeline
    chain = prompt_template | llm
    
    # Run the chain — fill in {email} and call the model
    # .invoke() runs the chain synchronously (waits for response)
    response = chain.invoke({"email": email_text})
    
    # response is an AIMessage object
    # response.content is the actual text string the model returned
    raw_output = response.content
    
    # Score the raw output using our evaluator
    result = score_response(
        email_id    = email_id,
        prompt_name = prompt_name,
        raw_output  = raw_output
    )
    
    return result


def run_prompt_against_all_emails(
    prompt_name: str,
    prompt_template
) -> list[EvalResult]:
    """
    Runs one prompt against all 5 test emails.
    Prints progress as it goes.
    Returns list of 5 EvalResult objects.
    """
    
    console.print(f"\n[bold blue]Testing: {prompt_name}[/bold blue]")
    console.print("─" * 50)
    
    results = []
    
    for test_case in TEST_EMAILS:
        
        email_id   = test_case["id"]
        email_text = test_case["email"]
        
        # Show progress
        console.print(f"  Email {email_id}: [dim]{email_text[:55]}...[/dim]")
        
        # Run the chain and get scored result
        result = run_one_email(
            prompt_name     = prompt_name,
            prompt_template = prompt_template,
            email_id        = email_id,
            email_text      = email_text
        )
        
        # Show immediate feedback
        score_color = "green" if result.total_score == 3 else \
                      "yellow" if result.total_score == 2 else "red"
        
        console.print(
            f"    Score: [{score_color}]{result.total_score}/3[/{score_color}]"
            f"  |  Status: {result.parse_status}"
        )
        
        # If it failed to parse, show what the model actually said
        if result.parse_status == "parse_failed":
            console.print(
                f"    [red]Raw output:[/red] {result.raw_output[:120]}"
            )
        
        results.append(result)
    
    return results


def print_comparison_table(all_summaries: list[dict]):
    """
    Uses rich to print a beautiful comparison table.
    Shows all 3 prompts side by side with scores.
    """
    
    # rich Table — this is how real engineers print structured data
    # to the terminal cleanly
    table = Table(
        title       = "Prompt Comparison — Zero Shot Baseline",
        show_header = True,
        header_style= "bold white on dark_blue"
    )
    
    # Add columns
    table.add_column("Prompt",        style="cyan",   width=28)
    table.add_column("Valid JSON",    style="white",  width=12, justify="center")
    table.add_column("All Fields",   style="white",  width=12, justify="center")
    table.add_column("Urgency OK",   style="white",  width=12, justify="center")
    table.add_column("Overall %",    style="bold",   width=12, justify="center")
    table.add_column("Parse Fails",  style="red",    width=14, justify="center")
    
    # Add one row per prompt
    for s in all_summaries:
        pct = s["overall_pct"]
        pct_color = "green" if pct >= 80 else "yellow" if pct >= 60 else "red"
        
        table.add_row(
            s["prompt_name"],
            s["valid_json"],
            s["all_fields"],
            s["urgency_valid"],
            f"[{pct_color}]{pct}%[/{pct_color}]",
            str(s["parse_failures"]) if s["parse_failures"] else "none"
        )
    
    console.print()
    console.print(table)


def save_results(
    all_results: list[EvalResult],
    all_summaries: list[dict]
):
    """
    Saves full results to outputs/v1_results.json
    We save everything so we can compare v1 vs v2 vs v3 later.
    """
    
    os.makedirs("outputs", exist_ok=True)
    
    # Convert EvalResult dataclasses to plain dicts for JSON serialization
    # vars() converts a dataclass instance to a dictionary
    results_as_dicts = [vars(r) for r in all_results]
    
    output = {
        "version":    "v1_zero_shot",
        "timestamp":  datetime.now().isoformat(),
        "model":      MODEL_NAME,
        "summaries":  all_summaries,
        "results":    results_as_dicts
    }
    
    filepath = "outputs/v1_results.json"
    
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    
    console.print(f"\n[green]Results saved to {filepath}[/green]")


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────

def main():
    
    console.print("\n[bold]P1 — Prompt Engineering Lab[/bold]")
    console.print("[bold]v1 — Zero Shot Baseline[/bold]")
    console.print(f"Model : [cyan]{MODEL_NAME}[/cyan]")
    console.print(f"Task  : Extract structured data from customer emails")
    
    all_results   = []
    all_summaries = []
    
    # Test every prompt against all 5 emails
    for prompt_name, prompt_template in ALL_PROMPTS.items():
        
        results = run_prompt_against_all_emails(prompt_name, prompt_template)
        summary = summarize_prompt_results(prompt_name, results)
        
        all_results.extend(results)
        all_summaries.append(summary)
    
    # Print the comparison table
    print_comparison_table(all_summaries)
    
    # Key questions to think about after running
    console.print("\n[bold yellow]Study These Questions:[/bold yellow]")
    console.print("  1. Which prompt scored highest and WHY?")
    console.print("  2. Which emails failed most often?")
    console.print("  3. What did the raw output look like when parse failed?")
    console.print("  4. What would YOU change in the prompt to fix the failures?")
    console.print("     → Answer these before moving to v2. v2 fixes the failures.")
    
    # Save everything
    save_results(all_results, all_summaries)


if __name__ == "__main__":
    main()