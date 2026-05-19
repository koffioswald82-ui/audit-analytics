"""
AI-powered audit memo generator.

Supported providers:
  - Groq (primary — llama-3.3-70b-versatile, llama-3.1-8b-instant)
  - Google Gemini (gemini-1.5-pro, gemini-2.0-flash)
  - OpenAI (gpt-4o, gpt-4-turbo)
  - Anthropic Claude (claude-opus-4-7, claude-sonnet-4-6)

Usage:
    reporter = AIReporter()
    memo = reporter.generate(findings, provider="groq", model="llama-3.3-70b-versatile", api_key="...")
"""
from __future__ import annotations
from typing import Any, Dict, Optional

# ── Provider catalogue ─────────────────────────────────────────────────────
PROVIDERS: Dict[str, Dict] = {
    "groq": {
        "label": "Groq (Recommended — fastest)",
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
            "mixtral-8x7b-32768",
        ],
        "key_hint": "gsk_...",
        "docs": "https://console.groq.com/keys",
    },
    "gemini": {
        "label": "Google Gemini",
        "models": [
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ],
        "key_hint": "AIza...",
        "docs": "https://aistudio.google.com/app/apikey",
    },
    "openai": {
        "label": "OpenAI",
        "models": [
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ],
        "key_hint": "sk-...",
        "docs": "https://platform.openai.com/api-keys",
    },
    "anthropic": {
        "label": "Anthropic Claude",
        "models": [
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ],
        "key_hint": "sk-ant-...",
        "docs": "https://console.anthropic.com/settings/api-keys",
    },
}


def _build_prompt(findings: Dict[str, Any]) -> str:
    """Build a comprehensive audit memo prompt from findings dict."""
    cfg        = findings.get("cfg", {})
    benford    = findings.get("benford")
    je         = findings.get("je_stats", {})
    ml         = findings.get("ml_stats", {})
    vendor     = findings.get("vendor_stats", {})
    top_f      = findings.get("top_findings", [])

    materiality  = cfg.get("materiality", 10_000)
    client       = cfg.get("client_name", "Client")
    period       = cfg.get("audit_period", "FY 2024")
    partner      = cfg.get("partner", "Engagement Partner")
    total        = je.get("total_entries", 0)
    flagged      = je.get("flagged_entries", 0)
    flagged_pct  = flagged / total * 100 if total else 0

    b_risk    = benford.isa_240_risk_level if benford else "N/A"
    b_chi2    = f"{benford.chi_square:.3f}"    if benford else "N/A"
    b_p       = f"{benford.chi_square_p:.4f}"  if benford else "N/A"
    b_mad     = f"{benford.mad:.5f}"           if benford else "N/A"
    b_conf    = benford.mad_conformity        if benford else "N/A"
    b_flags   = ", ".join(str(d) for d in (benford.red_flag_digits if benford else [])) or "None"
    b_comment = benford.isa_240_comment       if benford else "N/A"
    b_n       = benford.n                     if benford else 0

    top_ind = "\n".join(
        f"  • {k}: {v}" for k, v in je.get("top_indicators", {}).items()
    ) or "  • No high-frequency indicators"

    top_findings_text = "\n".join(
        f"  [{i+1}] {f['risk_level'].upper()} — {f['title']}: {f['description']} "
        f"(ISA ref: {f['isa_ref']})"
        for i, f in enumerate(top_f)
    ) or "  No significant findings identified."

    return f"""You are a Senior Audit Manager at a Big 4 accounting firm specialising in data analytics and forensic auditing. Write a comprehensive, professional AUDIT MEMORANDUM (Working Paper DA-001) based on the data analytics results below.

=== ENGAGEMENT DETAILS ===
Client:                  {client}
Audit Period:            {period}
Engagement Partner:      {partner}
Engagement Materiality:  ${materiality:,.0f}
Performance Materiality: ${materiality * 0.75:,.0f} (75%)
Working Paper Ref:       DA-001 — Data Analytics

=== DATA ANALYTICS RESULTS ===

1. BENFORD'S LAW ANALYSIS ({findings.get('digit_type','first')} digit, n={b_n:,})
   ISA 240 Risk Level:     {b_risk}
   Chi-Square:             {b_chi2}  (p-value: {b_p})
   Mean Absolute Deviation:{b_mad}  →  {b_conf}
   Red Flag Digits:        {b_flags}
   Comment:                {b_comment}

2. JOURNAL ENTRY RISK SCORING
   Total Entries:          {total:,}
   Flagged (score >= 40):  {flagged:,}  ({flagged_pct:.1f}%)
   Critical (score >= 70): {je.get('critical_je', 0):,}
   Top Risk Indicators:
{top_ind}

3. ML ANOMALY DETECTION (Isolation Forest)
   Anomalies Detected:     {ml.get('anomaly_count', 0):,}  ({ml.get('anomaly_count', 0) / total * 100:.2f}% of population)
   Consensus (ML + Stat):  {ml.get('consensus', 0):,} entries flagged by both methods
   Top Risk Drivers (SHAP):{ml.get('top_drivers', 'N/A')}

4. VENDOR / COUNTERPARTY ANALYSIS
   Duplicate Payments:     {vendor.get('duplicate_count', 0):,} entries
   Ghost Vendor Pairs:     {vendor.get('ghost_vendor_pairs', 0):,} similar name pairs
   Top Vendor Spend:       {vendor.get('top_vendor', 'N/A')} = {vendor.get('top_vendor_pct', 0):.1f}% of total

=== KEY FINDINGS SUMMARY ===
{top_findings_text}

=== REQUESTED OUTPUT FORMAT ===

Write a formal audit memorandum with the following sections:

**AUDIT MEMORANDUM — DA-001**
(Header: client, period, partner, date, materiality, classification)

**1. PURPOSE AND SCOPE**
State objective, methodology, standards applied (ISA 240, ISA 315, ISA 500, PCAOB AS 2401).

**2. EXECUTIVE RISK ASSESSMENT**
Overall risk level (Critical/High/Medium/Low) with concise justification. Include a risk matrix table.

**3. BENFORD'S LAW FINDINGS**
Technical results, statistical significance interpretation, specific red-flag digits, implications for fraud risk (ISA 240.A4, ISA 240.A26).

**4. JOURNAL ENTRY TESTING**
Key risk indicators activated, volume and amount analysis, top findings ranked by risk, ISA 240.A3 alignment.

**5. ML ANOMALY DETECTION**
Methodology summary, anomaly rate assessment, consensus analysis, PCAOB AS 2401.66 reference.

**6. VENDOR AND COUNTERPARTY RISK**
Duplicate payment analysis, ghost vendor risk, concentration risk, recommended follow-up.

**7. RANKED FINDINGS AND RECOMMENDED PROCEDURES**
For each finding: risk level, description, specific recommended audit procedure, ISA reference.

**8. IMPACT ON AUDIT STRATEGY**
How these findings affect the overall audit approach, sample sizes, substantive testing scope.

**9. LIMITATIONS**
Data quality assumptions, model limitations, ISA 500.6 caveat.

**10. CONCLUSION**
Summary of overall risk level and next steps.

Requirements:
- Use professional Big 4 language throughout
- Reference specific ISA paragraph numbers (e.g. ISA 240.A3, ISA 315.A107)
- Be precise about numbers from the data analytics results above
- Provide specific, actionable audit procedure recommendations
- Maintain an objective, professional tone
- Length: comprehensive but concise (approximately 800-1200 words)
"""


# ── Provider call functions ────────────────────────────────────────────────
def _call_groq(prompt: str, model: str, api_key: str) -> str:
    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert audit manager at a Big 4 accounting firm. "
                    "You write precise, professional audit documentation in accordance "
                    "with International Standards on Auditing (ISA) and PCAOB standards."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        temperature=0.3,
    )
    return response.choices[0].message.content


def _call_gemini(prompt: str, model: str, api_key: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel(
        model,
        system_instruction=(
            "You are an expert audit manager at a Big 4 accounting firm. "
            "Write precise, professional audit documentation aligned with ISA standards."
        ),
    )
    response = llm.generate_content(
        prompt,
        generation_config={"temperature": 0.3, "max_output_tokens": 4096},
    )
    return response.text


def _call_openai(prompt: str, model: str, api_key: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert audit manager at a Big 4 accounting firm. "
                    "Write precise, professional audit documentation aligned with ISA standards."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        temperature=0.3,
    )
    return response.choices[0].message.content


def _call_anthropic(prompt: str, model: str, api_key: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=(
            "You are an expert audit manager at a Big 4 accounting firm. "
            "Write precise, professional audit documentation aligned with ISA standards."
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ── Public API ─────────────────────────────────────────────────────────────
class AIReporter:

    _DISPATCHERS = {
        "groq":      _call_groq,
        "gemini":    _call_gemini,
        "openai":    _call_openai,
        "anthropic": _call_anthropic,
    }

    def generate(
        self,
        findings: Dict[str, Any],
        provider: str,
        model: str,
        api_key: str,
    ) -> str:
        if not api_key or not api_key.strip():
            raise ValueError("API key is required.")
        if provider not in self._DISPATCHERS:
            raise ValueError(f"Unknown provider: {provider!r}. Choose from: {list(self._DISPATCHERS)}")

        prompt   = _build_prompt(findings)
        fn       = self._DISPATCHERS[provider]
        return fn(prompt, model, api_key.strip())
