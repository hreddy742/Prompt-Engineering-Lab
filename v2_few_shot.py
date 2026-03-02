# v2_few_shot.py
#
# P1 Prompt Engineering Lab — Version 2: Few-Shot + Chain of Thought
#
# What changed from v1:
#   - Two new prompts: PROMPT_D (few-shot) and PROMPT_E (few-shot + CoT)
#   - We compare v2 prompts against the best v1 prompt (PROMPT_B)
#   - Everything else is identical — same chain, same evaluator, same emails
#
# Key lesson: same model, same code, better prompts = better results
# The ONLY thing that changed is what we said to the model

import json
import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from rich.console import Console
from rich.table import Table

from src.prompts import PROMPT_B, PROMPT_D, PROMPT_E
from src.evaluator import score_response, summarize_prompt_results, EvalResult

# ── load config ────────────────────────────────────────────────────────────
load_dotenv()

MODEL_NAME  = os.getenv("MODEL_NAME",      "llama3.1:8b")
BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

console = Console()

# ── same 5 test emails as v1 ───────────────────────────────────────────────
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

# ── initialise model — same as v1 ─────────────────────────────────────────
llm = ChatOllama(
    model       = MODEL_NAME,
    base_url    = BASE_URL,
    temperature = TEMPERATURE
)

# ── we only test these 3 prompts in v2 ────────────────────────────────────
# PROMPT_B = best from v1 (our baseline to beat)
# PROMPT_D = new few-shot prompt
# PROMPT_E = new few-shot + chain of thought prompt
PROMPTS_TO_TEST = {
    "PROMPT_B_v1_best":   PROMPT_B,
    "PROMPT_D_few_shot":  PROMPT_D,
    "PROMPT_E_cot":       PROMPT_E,
}


def run_one_email(prompt_name, prompt_template, email_id, email_text):
    """
    Runs one email through one prompt.
    Identical to v1 — chain pattern does not change.
    prompt | llm → response → score
    """
    chain    = prompt_template | llm
    response = chain.invoke({"email": email_text})

    return score_response(
        email_id    = email_id,
        prompt_name = prompt_name,
        raw_output  = response.content
    )


def run_prompt_against_all_emails(prompt_name, prompt_template):
    """
    Runs one prompt against all 5 emails.
    Prints live progress with scores.
    """

    console.print(f"\n[bold blue]Testing: {prompt_name}[/bold blue]")
    console.print("─" * 50)

    results = []

    for test_case in TEST_EMAILS:

        email_id   = test_case["id"]
        email_text = test_case["email"]

        console.print(f"  Email {email_id}: [dim]{email_text[:55]}...[/dim]")

        result = run_one_email(
            prompt_name     = prompt_name,
            prompt_template = prompt_template,
            email_id        = email_id,
            email_text      = email_text
        )

        # Color the score: green=perfect, yellow=okay, red=bad
        color = "green" if result.total_score == 3 else \
                "yellow" if result.total_score == 2 else "red"

        console.print(
            f"    Score : [{color}]{result.total_score}/3[/{color}]"
            f"  |  parse: {result.parse_status}"
        )

        # Show what the model actually said for chain of thought
        # This is interesting — with CoT you can SEE the model thinking
        if prompt_name == "PROMPT_E_cot":
            console.print(f"    [dim]Model thinking: {result.raw_output[:120]}[/dim]")

        results.append(result)

    return results


def print_comparison_table(v1_summaries, v2_summaries):
    """
    Prints a side by side table showing v1 best vs v2 new prompts.
    This is how you show improvement — always compare to your baseline.
    """

    table = Table(
        title        = "v1 Best vs v2 Few-Shot vs v2 Chain-of-Thought",
        header_style = "bold white on dark_blue",
        show_header  = True
    )

    table.add_column("Prompt",       style="cyan",  width=26)
    table.add_column("Version",      style="white", width=10, justify="center")
    table.add_column("Valid JSON",   style="white", width=12, justify="center")
    table.add_column("All Fields",  style="white", width=12, justify="center")
    table.add_column("Urgency OK",  style="white", width=12, justify="center")
    table.add_column("Overall %",   style="bold",  width=12, justify="center")

    # v1 baseline row
    for s in v1_summaries:
        pct   = s["overall_pct"]
        color = "green" if pct >= 80 else "yellow" if pct >= 60 else "red"
        table.add_row(
            s["prompt_name"],
            "[yellow]v1[/yellow]",
            s["valid_json"],
            s["all_fields"],
            s["urgency_valid"],
            f"[{color}]{pct}%[/{color}]"
        )

    # divider
    table.add_section()

    # v2 new prompt rows
    for s in v2_summaries:
        pct   = s["overall_pct"]
        color = "green" if pct >= 80 else "yellow" if pct >= 60 else "red"
        table.add_row(
            s["prompt_name"],
            "[green]v2[/green]",
            s["valid_json"],
            s["all_fields"],
            s["urgency_valid"],
            f"[{color}]{pct}%[/{color}]"
        )

    console.print()
    console.print(table)


def save_results(all_results, all_summaries):
    """Saves v2 results so we can compare across all versions later."""

    os.makedirs("outputs", exist_ok=True)

    output = {
        "version":   "v2_few_shot",
        "timestamp": datetime.now().isoformat(),
        "model":     MODEL_NAME,
        "summaries": all_summaries,
        "results":   [vars(r) for r in all_results]
    }

    with open("outputs/v2_results.json", "w") as f:
        json.dump(output, f, indent=2)

    console.print("\n[green]Results saved to outputs/v2_results.json[/green]")


def main():

    console.print("\n[bold]P1 — Prompt Engineering Lab[/bold]")
    console.print("[bold]v2 — Few-Shot + Chain of Thought[/bold]")
    console.print(f"Model  : [cyan]{MODEL_NAME}[/cyan]")
    console.print(f"Goal   : Beat v1 best score using examples and reasoning")

    # Load v1 results so we can compare in the table
    v1_summaries = []
    try:
        with open("outputs/v1_results.json") as f:
            v1_data = json.load(f)
            # Get only the best v1 prompt for comparison
            v1_summaries = [
                s for s in v1_data["summaries"]
                if s["prompt_name"] == "PROMPT_B_with_role"
            ]
    except FileNotFoundError:
        console.print("[yellow]v1 results not found — run v1 first for comparison[/yellow]")

    # Run all v2 prompts
    all_results   = []
    all_summaries = []

    for prompt_name, prompt_template in PROMPTS_TO_TEST.items():
        results = run_prompt_against_all_emails(prompt_name, prompt_template)
        summary = summarize_prompt_results(prompt_name, results)
        all_results.extend(results)
        all_summaries.append(summary)

    # Print comparison: v1 best vs v2 prompts
    print_comparison_table(v1_summaries, all_summaries)

    # What to study after running
    console.print("\n[bold yellow]Study These Questions:[/bold yellow]")
    console.print("  1. Did few-shot beat the best v1 prompt?")
    console.print("  2. Did chain-of-thought help or hurt compared to few-shot alone?")
    console.print("  3. Look at PROMPT_E output — can you see the model thinking?")
    console.print("  4. Which email is still the hardest? Why?")

    save_results(all_results, all_summaries)


if __name__ == "__main__":
    main()