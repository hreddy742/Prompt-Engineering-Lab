# src/evaluator.py
#
# Scoring logic for prompt evaluation
#
# Real engineering practice: evaluation code is separate from running code.
# You can import and reuse this in any future project.

import json
from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────
# PYDANTIC OUTPUT SCHEMA
#
# This is the blueprint of what our LLM output MUST look like.
#
# BaseModel  = Pydantic's base class. Inherit from it to get validation.
# Field()    = lets us add a description to each field. The description
#              is sent to the model so it knows what to put in each field.
# Literal    = means this field can ONLY be one of these exact values.
#              urgency: Literal["high","medium","low"] means the model
#              physically cannot return "HIGH" or "urgent" — only those 3.
#
# When the model returns output, Pydantic checks every field:
#   - Is it present? If not → ValidationError
#   - Is it the right type? If not → ValidationError  
#   - Is it one of the allowed values? If not → ValidationError
# ─────────────────────────────────────────────────────────────────────────

class EmailExtraction(BaseModel):
    """Schema for extracting structured data from customer emails."""
    
    customer_issue: str = Field(
        description="What is the customer's problem — short phrase"
    )
    product: str | None = Field(
        description="Product name if mentioned in the email, else null"
    )
    order_number: str | None = Field(
        description="Order number if mentioned in the email, else null"
    )
    urgency: Literal["high", "medium", "low"] = Field(
        description="How urgent is this: high=angry/ASAP/threatening, medium=wants resolution, low=just asking"
    )
    action_needed: str = Field(
        description="What action does the customer want us to take"
    )

# ─────────────────────────────────────────────────────────────────────────
# WHAT IS A DATACLASS?
#
# A dataclass is a clean way to define a data structure in Python.
# Instead of passing around loose dictionaries like {"format_ok": 1, ...}
# we define a proper typed structure.
#
# Real engineers prefer dataclasses over plain dicts because:
# - You get autocomplete in VS Code (type result. and see all fields)
# - Errors are caught earlier (wrong field name = immediate error)
# - Code is self-documenting (the class definition tells you what fields exist)
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Stores the evaluation result for one model response."""
    
    email_id:       int
    prompt_name:    str
    raw_output:     str          # exactly what the model returned
    parsed:         dict | None  # the parsed JSON, or None if parsing failed
    parse_status:   str          # "clean_json", "extracted_json", "parse_failed"
    format_ok:      int          # 1 if valid JSON, 0 if not
    has_all_fields: int          # 1 if all 5 required fields present
    urgency_valid:  int          # 1 if urgency is exactly high/medium/low
    total_score:    int          # sum of above three (max 3)


def parse_llm_output(raw_text: str) -> tuple[dict | None, str]:
    """
    Extracts valid JSON from whatever the model returned.
    
    Why do we still need this even with LangChain?
    LangChain has output parsers, yes. But for learning purposes we are
    doing manual parsing in v1 so you SEE the failures clearly.
    In v3 we will switch to LangChain's JsonOutputParser which handles
    this automatically. First understand the problem, then use the tool.
    
    Returns:
        (parsed_dict, status_string)
        status is one of: "clean_json", "extracted_json", "parse_failed"
    """
    
    # Attempt 1 — maybe the model returned clean JSON with no extra text
    try:
        parsed = json.loads(raw_text.strip())
        return parsed, "clean_json"
    except json.JSONDecodeError:
        pass
    
    # Attempt 2 — model added text around the JSON, find the {...} part
    start = raw_text.find("{")
    end   = raw_text.rfind("}") + 1
    
    if start != -1 and end > start:
        try:
            parsed = json.loads(raw_text[start:end])
            return parsed, "extracted_json"
        except json.JSONDecodeError:
            pass
    
    # Both failed — model did not produce parseable JSON
    return None, "parse_failed"


def score_response(
    email_id: int,
    prompt_name: str,
    raw_output: str
) -> EvalResult:
    """
    Takes one model response and scores it on 3 criteria.
    Returns an EvalResult dataclass with all scores.
    
    The 3 criteria:
    1. format_ok      — did we get valid JSON at all?
    2. has_all_fields — does it have all 5 required fields?
    3. urgency_valid  — is urgency exactly high/medium/low?
    
    Why these 3 specifically?
    Because they test different things:
    - format_ok tests if the model follows output format instructions
    - has_all_fields tests if the model reads and follows field specifications
    - urgency_valid tests if the model follows value constraints
    Each failure points to a different fix in the prompt.
    """
    
    REQUIRED_FIELDS = [
        "customer_issue",
        "product",
        "order_number",
        "urgency",
        "action_needed"
    ]
    
    # Parse the raw output
    parsed, status = parse_llm_output(raw_output)
    
    # If we could not parse JSON, all scores are 0
    if parsed is None:
        return EvalResult(
            email_id       = email_id,
            prompt_name    = prompt_name,
            raw_output     = raw_output,
            parsed         = None,
            parse_status   = status,
            format_ok      = 0,
            has_all_fields = 0,
            urgency_valid  = 0,
            total_score    = 0
        )
    
    # Score 1: format — we got valid JSON so this is 1
    format_ok = 1
    
    # Score 2: all required fields present?
    has_all_fields = 1 if all(f in parsed for f in REQUIRED_FIELDS) else 0
    
    # Score 3: urgency is exactly one of our three valid values?
    urgency = str(parsed.get("urgency", "")).lower().strip()
    urgency_valid = 1 if urgency in ["high", "medium", "low"] else 0
    
    return EvalResult(
        email_id       = email_id,
        prompt_name    = prompt_name,
        raw_output     = raw_output,
        parsed         = parsed,
        parse_status   = status,
        format_ok      = format_ok,
        has_all_fields = has_all_fields,
        urgency_valid  = urgency_valid,
        total_score    = format_ok + has_all_fields + urgency_valid
    )


def summarize_prompt_results(
    prompt_name: str,
    results: list[EvalResult]
) -> dict:
    """
    Calculates aggregate scores for one prompt across all test emails.
    Returns a summary dictionary.
    """
    
    n = len(results)
    
    total_format  = sum(r.format_ok      for r in results)
    total_fields  = sum(r.has_all_fields for r in results)
    total_urgency = sum(r.urgency_valid  for r in results)
    total_score   = sum(r.total_score    for r in results)
    
    return {
        "prompt_name":    prompt_name,
        "n_emails":       n,
        "valid_json":     f"{total_format}/{n}",
        "all_fields":     f"{total_fields}/{n}",
        "urgency_valid":  f"{total_urgency}/{n}",
        "overall_pct":    round(total_score / (n * 3) * 100, 1),
        "parse_failures": [r.email_id for r in results 
                           if r.parse_status == "parse_failed"]
    }