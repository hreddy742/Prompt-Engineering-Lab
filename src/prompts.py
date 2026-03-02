from langchain_core.prompts import ChatPromptTemplate

PROMPT_A = ChatPromptTemplate.from_messages([
    ("human","""Extract the information from this customer email and Return JSON object with these fields:
    - customer_issue: what is the problem (short phrase)
    - product: product name if mentioned, else null
    - order_number: order number if mentioned, else null
    - urgency: high, medium, or low
    - action_needed: what action the customer wants

Email: {email}

Return only the JSON. No explanation."""
    )
])

PROMPT_B = ChatPromptTemplate.from_messages([
    ('system', """You are a customer service data extraction assistant.
Your job is to read emails and extract the key information in a structured JSON.
You are precise, consistent, and always follow the output format exactly."""),

    ('human', """Extract information from the customer email below.

Return a JSON object with EXACTLY these keys:
- customer_issue  : what is the problem (short phrase)
- product         : product name if mentioned, else null
- order_number    : order number if mentioned, else null
- urgency         : must be exactly one of: high, medium, low
- action_needed   : what action the customer wants

Rules:
- Return ONLY the JSON object. No explanation, no markdown, no code fences.
- If a field is not present in the email, use null
- urgency must be lowercase: high, medium, or low — nothing else

Customer Email:
{email}""")
])


PROMPT_C = ChatPromptTemplate.from_messages([
    (
        "human",
        "Extract data from this email as JSON: {email}"
    )
])

# ─────────────────────────────────────────────────────────────────────────
# V2 PROMPTS — Few-shot + Chain of Thought
# ─────────────────────────────────────────────────────────────────────────

# PROMPT D — Few-shot
# We show the model 2 examples of perfect input → output pairs
# Now the model has seen what we want — no guessing needed
PROMPT_D = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a customer service data extraction assistant.
Your job is to extract key information from customer emails and return it as JSON.
You always return exactly the fields specified, nothing more, nothing less."""
    ),
    (
        "human",
        """Extract information from the customer email.

Return a JSON object with EXACTLY these keys:
  customer_issue  : what is the problem
  product         : product name if mentioned, else null
  order_number    : order number if mentioned, else null
  urgency         : must be exactly: high, medium, or low
  action_needed   : what the customer wants

Return ONLY the JSON. No explanation. No markdown.

Here are two examples of perfect extractions:

EXAMPLE 1:
Email: "My laptop screen cracked after 2 days, order #55123. I need a refund immediately."
Output: {{"customer_issue": "cracked screen", "product": "laptop", "order_number": "55123", "urgency": "high", "action_needed": "refund"}}

EXAMPLE 2:
Email: "Hi just wondering if you have the red jacket in size medium? No order yet."
Output: {{"customer_issue": "product availability inquiry", "product": "red jacket", "order_number": null, "urgency": "low", "action_needed": "check stock"}}

Now extract from this email:
Email: {email}
Output:"""
    )
])


# PROMPT E — Few-shot + Chain of Thought
# Same examples as above BUT we ask the model to think before answering
# Chain of thought helps with ambiguous cases
PROMPT_E = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a customer service data extraction assistant.
Your job is to extract key information from customer emails and return it as JSON.
You think carefully before extracting to make sure you get every field right."""
    ),
    (
        "human",
        """Extract information from the customer email.

Return a JSON object with EXACTLY these keys:
  customer_issue  : what is the problem
  product         : product name if mentioned, else null
  order_number    : order number if mentioned, else null
  urgency         : must be exactly: high, medium, or low
  action_needed   : what the customer wants

Here are two examples of perfect extractions:

EXAMPLE 1:
Email: "My laptop screen cracked after 2 days, order #55123. I need a refund immediately."
Output: {{"customer_issue": "cracked screen", "product": "laptop", "order_number": "55123", "urgency": "high", "action_needed": "refund"}}

EXAMPLE 2:
Email: "Hi just wondering if you have the red jacket in size medium? No order yet."
Output: {{"customer_issue": "product availability inquiry", "product": "red jacket", "order_number": null, "urgency": "low", "action_needed": "check stock"}}

Now for the email below, follow these steps:
STEP 1 - Read the email carefully
STEP 2 - Identify: is there a product? an order number? how urgent is this?
STEP 3 - Decide urgency: high = angry/ASAP/threatening, medium = wants resolution, low = just asking
STEP 4 - Write the JSON output

Email: {email}
Think through steps 1-3 first, then on the LAST LINE write only the JSON."""
    )
])

ALL_PROMPTS = {
    "PROMPT_A_minimal":      PROMPT_A,
    "PROMPT_B_with_role":    PROMPT_B,
    "PROMPT_C_ultra_bare":   PROMPT_C,
    "PROMPT_D_few_shot":     PROMPT_D,
    "PROMPT_E_cot":          PROMPT_E,
}