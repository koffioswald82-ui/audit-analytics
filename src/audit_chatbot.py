"""
AuditAI Assistant — contextual chatbot for audit concepts and platform guidance.

Answers questions about: Benford's Law, ISA standards, journal entry testing,
ML anomaly detection, statistical tests, and platform configuration.
"""
from __future__ import annotations
from typing import Any, Dict, List

SYSTEM_PROMPT = """You are AuditAI Assistant, an expert in external audit, forensic accounting,
and audit data analytics. You support audit professionals — from junior associates to senior managers
— in understanding both the technical concepts behind the platform and core audit methodology.

AREAS OF EXPERTISE

1. Benford's Law
   - First and second digit distributions; Benford (1938); Nigrini (2012)
   - Chi-square test: null hypothesis, degrees of freedom, p-value interpretation
   - Mean Absolute Deviation (MAD): Nigrini thresholds (Close / Acceptable / Marginal / Nonconformity)
   - Z-score per digit: statistical significance at 1.96 (5%) and 2.576 (1%)
   - Kolmogorov-Smirnov test as supplementary evidence
   - Why natural financial data follows Benford's Law (scale invariance, multiplicative processes)
   - When Benford's Law does NOT apply: assigned numbers, restricted ranges, small samples

2. Journal Entry Testing
   - ISA 240.A3: fraud risk indicators for journal entries
   - Round number bias: Cressey fraud triangle, threshold manipulation
   - Period-end entries: revenue recognition risk, cut-off assertions (ISA 315)
   - Off-hours postings: segregation of duties implications
   - Same-day reversals: masking of fictitious entries
   - Suspense accounts: ISA 240.A33, incomplete allocation risk
   - Risk scoring methodology: composite weighted indicators

3. Machine Learning Anomaly Detection
   - Isolation Forest: how it works (random partitioning, path length), contamination parameter
   - SHAP values: Shapley additive explanations, feature importance interpretation
   - Anomaly score interpretation: 0 to 1 scale, percentile rank
   - Consensus analysis: why ML + statistical overlap has the highest evidential weight
   - K-means clustering: grouping anomalies by behavioural pattern
   - Limitations: unsupervised learning, no ground truth labels

4. Vendor and Counterparty Analysis
   - Duplicate payment detection: invoice-splitting schemes, approval threshold avoidance
   - Ghost vendor schemes: fictitious supplier creation, name similarity patterns
   - Vendor concentration: 80/20 rule, single-source risk
   - Fuzzy matching: Levenshtein distance, token sort ratio

5. ISA Standards
   - ISA 240: fraud responsibilities, professional skepticism, journal entry testing (para A3, A26, A33)
   - ISA 315: risk assessment, understanding internal controls (para A85, A107)
   - ISA 500: audit evidence, analytical procedures (para A1, A14)
   - ISA 530: audit sampling, using risk scores for stratification
   - ISA 560: subsequent events, period-end cut-off
   - PCAOB AS 2401: US equivalent of ISA 240 for fraud consideration
   - COSO Fraud Risk Management Guide: control environment, risk assessment

6. Platform Parameters
   - Materiality threshold: affects large-amount flagging and invoice-split detection
   - Performance materiality (75%): lower threshold for individual misstatements
   - Period-end window: configurable days before month/quarter/year-end
   - Statistical significance alpha: 0.01 / 0.05 / 0.10 and what each means
   - Benford minimum amount: why excluding trivial amounts improves analysis quality
   - Contamination rate: Isolation Forest parameter, set to expected fraud prevalence
   - Duplicate window: lookback period for matching payments
   - Ghost vendor threshold: similarity score interpretation

7. Interpreting Results
   - How to read the Benford chart: deviation bars, red highlighted digits
   - What a HIGH / MEDIUM / LOW ISA 240 risk means for audit procedures
   - How to prioritise the flagged journal entry table: Critical vs High vs Medium
   - What the anomaly score means: closer to 1 = more anomalous
   - How to use the findings to adjust the overall audit strategy

COMMUNICATION STYLE
- Professional, clear, concise
- Use specific ISA paragraph references (e.g. ISA 240.A3) when relevant
- Give practical examples when explaining abstract concepts
- If the question is outside audit/finance scope, politely redirect
- Respond in the same language as the question (French or English)
- Avoid unnecessary jargon; explain technical terms when first used
"""


def build_chat_prompt(history: List[Dict], user_message: str) -> List[Dict]:
    """Convert session history + new message to messages list format."""
    messages = []
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})
    return messages


def call_chatbot(
    history: List[Dict],
    user_message: str,
    provider: str,
    model: str,
    api_key: str,
) -> str:
    messages = build_chat_prompt(history, user_message)

    if provider == "groq":
        from groq import Groq
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            max_tokens=1024,
            temperature=0.4,
        )
        return resp.choices[0].message.content

    elif provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        llm = genai.GenerativeModel(model, system_instruction=SYSTEM_PROMPT)
        # Build Gemini history format
        gemini_history = []
        for turn in history:
            role = "user" if turn["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [turn["content"]]})
        chat = llm.start_chat(history=gemini_history)
        resp = chat.send_message(user_message)
        return resp.text

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            max_tokens=1024,
            temperature=0.4,
        )
        return resp.choices[0].message.content

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return msg.content[0].text

    raise ValueError(f"Unknown provider: {provider!r}")


# Suggested starter questions shown to first-time users
STARTER_QUESTIONS = [
    "What is Benford's Law and why does financial data follow it?",
    "How should I interpret a HIGH ISA 240 risk level?",
    "What does the anomaly score (0 to 1) represent?",
    "Qu'est-ce que le MAD et comment l'interpreter?",
    "What are the most important ISA 240 journal entry risk indicators?",
    "How does Isolation Forest detect anomalies in GL data?",
    "What is the difference between engagement materiality and performance materiality?",
    "Why are period-end entries flagged as high risk?",
    "Comment configurer le seuil de materialite pour mon engagement?",
    "What does a Z-score above 1.96 mean for a Benford digit?",
]
