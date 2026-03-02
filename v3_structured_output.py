# v3_structured_output.py
#
# P1 Prompt Engineering Lab — Version 3: Pydantic Structured Output
#
# What is new in v3:
#   - llm.with_structured_output(EmailExtraction) replaces manual JSON parsing
#   - The model is forced to return data matching our Pydantic schema
#   - No more try_parse_json() — Pydantic handles everything
#   - Automatic retry if output does not match schema
#
# Key lesson: do not hope for correct output — enforce it

import json
import os
from datetime import datetime

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console
from rich.table import Table

from src.evaluator import EmailExtraction

# ── simple direct setup — no .env needed for local Ollama ─────────────────
console = Console()

llm = ChatOllama(
    model       = "llama3.1:8b",
    temperature = 0
)

# ── same 5 test emails ─────────────────────────────────────────────────────
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

# ── the prompt for v3 ──────────────────────────────────────────────────────
# Notice this prompt is simpler than v2.
# Why? Because with_structured_output sends the schema to the model
# automatically. The model already knows the fields and valid values
# from the Pydantic schema. We do not need to repeat them in the prompt.
# Less repetition = cleaner prompt = better results.

PROMPT_V3 = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a customer service data extraction assistant.
Extract key information from customer emails accurately and completely."""
    ),
    (
        "human",
        """Extract the structured information from this customer email:

{email}"""
    )
])

# ── the structured chain ───────────────────────────────────────────────────
#
# This is the key difference from v1 and v2:
#
# v1 and v2:  prompt | llm  → raw text → manual JSON parsing
# v3:         prompt | llm.with_structured_output(EmailExtraction)
#                           → EmailExtraction object directly
#
# with_structured_output does three things:
#   1. Tells the model the exact schema it must follow
#   2. Gets the response back
#   3. Validates and parses it into an EmailExtraction object
#
# You get back a proper Python object — not text, not a dict.
# Access fields like: result.urgency, result.product, result.order_number

structured_llm = llm.with_structured_output(EmailExtraction)
chain          = PROMPT_V3 | structured_llm


def run_one_email(email_id: int, email_text: str) -> dict:
    """
    Runs one email through the structured chain.
    Returns a result dictionary with the extracted data.

    Notice: no try_parse_json() needed.
    No JSON parsing at all.
    We just call chain.invoke() and get back an EmailExtraction object.
    """

    try:
        # Run the chain — returns an EmailExtraction object
        result = chain.invoke({"email": email_text})

        # result is now a proper Python object
        # result.customer_issue, result.product, etc are all typed fields
        return {
            "email_id":       email_id,
            "success":        True,
            "customer_issue": result.customer_issue,
            "product":        result.product,
            "order_number":   result.order_number,
            "urgency":        result.urgency,
            "action_needed":  result.action_needed,
            "error":          None
        }

    except Exception as e:
        # Only fails if model completely ignores the schema
        # Very rare with llama3.1:8b but we handle it anyway
        return {
            "email_id":       email_id,
            "success":        False,
            "customer_issue": None,
            "product":        None,
            "order_number":   None,
            "urgency":        None,
            "action_needed":  None,
            "error":          str(e)
        }


def print_results_table(results: list[dict]):
    """
    Prints a clean table showing extracted data for all 5 emails.
    This shows the ACTUAL extracted content — not just scores.
    Because with structured output, if it ran — it is correct.
    """

    table = Table(
        title        = "v3 Structured Output — Extracted Data",
        header_style = "bold white on dark_blue",
        show_header  = True
    )

    table.add_column("ID",            width=4,  justify="center")
    table.add_column("Issue",         width=22, style="cyan")
    table.add_column("Product",       width=16)
    table.add_column("Order",         width=10, justify="center")
    table.add_column("Urgency",       width=10, justify="center")
    table.add_column("Action",        width=20)
    table.add_column("Status",        width=8,  justify="center")

    for r in results:

        # Color the urgency value
        urgency = r["urgency"] or "—"
        if urgency == "high":
            urgency_str = "[red]high[/red]"
        elif urgency == "medium":
            urgency_str = "[yellow]medium[/yellow]"
        else:
            urgency_str = "[green]low[/green]"

        status = "[green]OK[/green]" if r["success"] else "[red]FAIL[/red]"

        table.add_row(
            str(r["email_id"]),
            r["customer_issue"] or "—",
            r["product"]        or "[dim]null[/dim]",
            r["order_number"]   or "[dim]null[/dim]",
            urgency_str,
            r["action_needed"]  or "—",
            status
        )

    console.print()
    console.print(table)


def save_results(results: list[dict]):
    """Saves v3 results to outputs folder."""

    os.makedirs("outputs", exist_ok=True)

    output = {
        "version":   "v3_structured_output",
        "timestamp": datetime.now().isoformat(),
        "model":     "llama3.1:8b",
        "results":   results
    }

    with open("outputs/v3_results.json", "w") as f:
        json.dump(output, f, indent=2)

    console.print("[green]Results saved to outputs/v3_results.json[/green]")


def main():

    console.print("\n[bold]P1 — Prompt Engineering Lab[/bold]")
    console.print("[bold]v3 — Pydantic Structured Output[/bold]")
    console.print("Model  : [cyan]llama3.1:8b[/cyan]")
    console.print("Change : with_structured_output replaces manual JSON parsing")

    results = []

    console.print("\n[bold blue]Running structured extraction...[/bold blue]")
    console.print("─" * 50)

    for test_case in TEST_EMAILS:

        email_id   = test_case["id"]
        email_text = test_case["email"]

        console.print(f"\n  Email {email_id}: [dim]{email_text[:55]}...[/dim]")

        result = run_one_email(email_id, email_text)

        if result["success"]:
            console.print(f"    urgency : {result['urgency']}")
            console.print(f"    product : {result['product']}")
            console.print(f"    order   : {result['order_number']}")
        else:
            console.print(f"    [red]Failed: {result['error'][:80]}[/red]")

        results.append(result)

    # Print the full extraction table
    print_results_table(results)

    # Key comparison point
    success_count = sum(1 for r in results if r["success"])
    console.print(f"\n[bold]Success rate: {success_count}/5[/bold]")

    console.print("\n[bold yellow]Compare v1 vs v2 vs v3:[/bold yellow]")
    console.print("  v1 zero-shot    : manually parse JSON, hope for valid output")
    console.print("  v2 few-shot     : better prompts, still manually parse JSON")
    console.print("  v3 structured   : schema enforced, no parsing, typed object")
    console.print("\n  v3 is how production systems work.")
    console.print("  You never manually parse LLM output in real projects.")

    save_results(results)


if __name__ == "__main__":
    main()