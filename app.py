# app.py
#
# P1 Prompt Engineering Lab — v4: Streamlit Playground
#
# What this does:
#   - Web UI to test any email against any prompt version
#   - Shows extracted data in a clean layout
#   - Shows response time so you can feel the difference
#   - Lets you compare v2 (manual parsing) vs v3 (structured output)
#
# Run it with: streamlit run app.py

import time
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from src.evaluator import EmailExtraction
from src.prompts import PROMPT_B, PROMPT_D, PROMPT_E

# ── page config ────────────────────────────────────────────────────────────
# This must be the FIRST streamlit command in your script
# Sets the browser tab title and page layout
st.set_page_config(
    page_title = "Prompt Engineering Playground",
    page_icon  = "🔧",
    layout     = "wide"   # uses full browser width
)

# ── initialise the model ───────────────────────────────────────────────────
# @st.cache_resource is important here
# Without it: every time you click a button, Streamlit reruns the whole
# script and creates a new ChatOllama object — slow and wasteful
# With it: ChatOllama is created ONCE and reused across all interactions
# Think of it as: create the model connection once, keep it alive

@st.cache_resource
def load_model():
    return ChatOllama(
        model       = "llama3.1:8b",
        temperature = 0
    )

llm = load_model()

# ── prompt for structured output ───────────────────────────────────────────
PROMPT_V3 = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a customer service data extraction assistant. "
        "Extract key information from customer emails accurately."
    ),
    (
        "human",
        "Extract the structured information from this customer email:\n\n{email}"
    )
])

# ── example emails for the dropdown ───────────────────────────────────────
# We give users example emails so they can try the app immediately
# without having to type their own

EXAMPLE_EMAILS = {
    "Select an example...": "",
    "Angry customer — missing package": 
        "WORST EXPERIENCE EVER. Package never arrived, order #99012, "
        "been waiting 3 weeks. I want my money back immediately or I "
        "am disputing with my bank.",
    "Calm customer — refund check": 
        "Hello, just checking if my refund for order 77203 has been "
        "processed? I returned the blue jacket 2 weeks ago and still "
        "nothing in my account. Not urgent just want to know.",
    "Product defect — urgent": 
        "Hi, I bought the UltraBoost X shoes last week, order #84521, "
        "and the left shoe sole is coming apart already. "
        "I need a replacement ASAP.",
    "Just browsing — no order": 
        "Hey can you help me find a gift for my mom? "
        "She likes gardening. Budget around $50. No order yet just browsing.",
    "Technical issue — medium urgency": 
        "My password reset email is not coming through. "
        "Email is jane@example.com. Not super urgent but would like to fix today."
}


# ── helper: run structured extraction ─────────────────────────────────────

def run_structured(email_text: str) -> tuple:
    """
    Runs the email through Pydantic structured output chain.
    Returns (EmailExtraction object or None, time taken, error message)
    """

    structured_llm = llm.with_structured_output(EmailExtraction)
    chain          = PROMPT_V3 | structured_llm

    start = time.time()

    try:
        result   = chain.invoke({"email": email_text})
        duration = round(time.time() - start, 2)
        return result, duration, None

    except Exception as e:
        duration = round(time.time() - start, 2)
        return None, duration, str(e)


# ── helper: display the extraction result ─────────────────────────────────

def show_result(result: EmailExtraction, duration: float):
    """
    Displays the extracted data in a clean layout.
    Uses Streamlit's metric and colored text components.
    """

    # Urgency gets a colored badge
    urgency_colors = {
        "high":   "🔴 high",
        "medium": "🟡 medium",
        "low":    "🟢 low"
    }
    urgency_display = urgency_colors.get(result.urgency, result.urgency)

    # st.metric shows a big labeled value — good for key info
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Urgency", urgency_display)

    with col2:
        st.metric("Order Number", result.order_number or "not mentioned")

    with col3:
        st.metric("Product", result.product or "not mentioned")

    st.divider()

    # Remaining fields as labeled text
    st.markdown(f"**Customer Issue:** {result.customer_issue}")
    st.markdown(f"**Action Needed:** {result.action_needed}")

    st.divider()

    # Response time at the bottom
    st.caption(f"⏱ Response time: {duration} seconds  |  Model: llama3.1:8b")


# ── main UI ────────────────────────────────────────────────────────────────
# Everything below is the actual page layout
# Streamlit reads your script top to bottom and renders each element

def main():

    # Page title
    st.title("🔧 Prompt Engineering Playground")
    st.markdown(
        "Test how different prompts extract structured data from customer emails. "
        "Built with LangChain + Ollama — runs 100% locally."
    )

    st.divider()

    # Two column layout — left for input, right for output
    # [1, 1] means equal width. [1, 2] would make right column twice as wide.
    left_col, right_col = st.columns([1, 1])

    # ── LEFT COLUMN — inputs ───────────────────────────────────────────────
    with left_col:

        st.subheader("📧 Input")

        # Example email dropdown
        # When user picks an example, it fills the text area automatically
        selected_example = st.selectbox(
            "Load an example email:",
            options = list(EXAMPLE_EMAILS.keys())
        )

        # Text area for the email
        # value= pre-fills it with the selected example
        email_input = st.text_area(
            label       = "Or type your own email:",
            value       = EXAMPLE_EMAILS[selected_example],
            height      = 180,
            placeholder = "Paste any customer email here..."
        )

        st.divider()

        # Prompt version selector
        st.markdown("**Choose extraction method:**")

        method = st.radio(
            label    = "Method",
            options  = [
                "v3 — Pydantic Structured Output",
                "v2 — Few-Shot (manual parsing)",
            ],
            index    = 0,   # default to v3
            label_visibility = "collapsed"
        )

        st.divider()

        # The run button
        # st.button returns True when clicked, False otherwise
        run_clicked = st.button(
            label = "▶ Run Extraction",
            type  = "primary",   # makes it blue and prominent
            use_container_width = True
        )

    # ── RIGHT COLUMN — output ──────────────────────────────────────────────
    with right_col:

        st.subheader("📊 Extracted Result")

        # Only run when button is clicked AND there is an email
        if run_clicked:

            if not email_input.strip():
                st.warning("Please enter an email first.")

            else:
                # Show a spinner while the model is thinking
                # This gives the user feedback that something is happening
                with st.spinner("Extracting... (this takes 1-3 seconds)"):

                    if "v3" in method:
                        # Structured output path
                        result, duration, error = run_structured(email_input)

                        if error:
                            st.error(f"Error: {error}")
                        else:
                            st.success("Extraction complete")
                            show_result(result, duration)

                    else:
                        # v2 manual parsing path
                        # We use PROMPT_E (best v2 prompt — few-shot + CoT)
                        chain      = PROMPT_E | llm
                        start      = time.time()
                        response   = chain.invoke({"email": email_input})
                        duration   = round(time.time() - start, 2)
                        raw_output = response.content

                        st.info("v2 raw output (manual parsing):")
                        st.code(raw_output, language="json")
                        st.caption(f"⏱ {duration} seconds")

        else:
            # Default state before button is clicked
            st.info(
                "👈 Paste an email on the left and click Run Extraction"
            )

    # ── BOTTOM SECTION — what you learned ─────────────────────────────────
    st.divider()

    with st.expander("📚 What Each Version Does — Click to Read"):
        st.markdown("""
**v1 — Zero Shot**
Just instructions. No examples. Model guesses the format.
Result: 73% accuracy. Inconsistent JSON. Parsing fails sometimes.

**v2 — Few Shot + Chain of Thought**  
Instructions + 2 examples + step by step reasoning.
Result: 100% accuracy. But still manual JSON parsing — can still fail in edge cases.

**v3 — Pydantic Structured Output**  
Schema enforced. Model cannot return wrong values or missing fields.
Result: 100% accuracy with type safety. This is how production systems work.

**Key lesson:**  
Do not hope for correct output. Enforce it.
        """)


if __name__ == "__main__":
    main()