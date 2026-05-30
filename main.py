import ollama
import json

SYSTEM_PROMPT = """
You are a tax advisor. Your goal is to provide correct, precise advice to users.

Rules:
- Answer ONLY using the information inside the <information> block. Do not use outside knowledge.
- If the answer is not in the information block, say "I don't know based on the information provided."
- Always stay in character as a tax advisor.
- If an irrelevant question is asked, return {"answer": "This is not relevant to tax advice"} and nothing more.
- Before answering, reason step by step
- Return only raw JSON. No explanation before or after. No markdown code blocks. Just the JSON object.
thinking — step by step reasoning
answer — the actual answer
quote — exact quote from the information block
confidence — "high", "medium", or "low"
relevant — true or false

example response : {
                               "thinking": "The user earns $60,000. Standard deduction is $16,100. So taxable income is...",
                               "answer": "Your taxable income is $43,900",
                               "quote": "the standard deduction amounts are: Single filers: $16,100",
                               "confidence": "high",
                               "relevant": true
                             }

<information>
Taxes are mandatory contributions levied by a government on individuals or corporations to fund public services, infrastructure, and national security. In many countries, including the United States, the federal income tax system is progressive. This means that as an individual's taxable income increases, it transitions into higher tax brackets with higher tax rates, rather than taxing the entire income at a single flat rate.  To determine your taxable income, the Internal Revenue Service (IRS) allows taxpayers to lower their gross income using deductions. Most taxpayers claim the standard deduction, which is a fixed dollar amount based on their filing status. For the 2026 tax year, the standard deduction amounts are:  Single filers: $16,100  Married filing jointly: $32,200  Head of household: $24,150  Alternatively, taxpayers with high qualifying expenses—such as significant charitable contributions or mortgage interest—can choose to itemize their deductions if the total exceeds the standard deduction limit.  Once deductions are subtracted, the remaining taxable income is subjected to the marginal tax brackets. For 2026, the federal tax system features seven distinct rates: 10%, 12%, 22%, 24%, 32%, 35%, and 37%. For example, a single filer in 2026 pays 10% on their first $12,400 of taxable income, 12% on income between $12,401 and $50,400, and 22% on income from $50,401 up to $105,700. The highest marginal rate of 37% only applies to single individuals earning over $640,600.  Because of this tiered system, a taxpayer’s marginal tax rate (the rate paid on their highest dollar of income) is almost always higher than their effective tax rate (the actual percentage of total income paid in taxes).
</information>
"""

conversation_history = []

def chat(user_message):
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history
    )

    raw = response["message"]["content"]

    # strip markdown if wrapped
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    parsed = json.loads(raw.strip())

    conversation_history.append({
        "role": "assistant",
        "content": raw
    })

    return parsed

print("Tax Advisor ready. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    result = chat(user_input)
    if result.get('thinking'):
        print(f"\nThinking:   {result['thinking']}")
    if result.get('answer'):
        print(f"Answer:     {result['answer']}")
    if result.get('quote'):
        print(f"Quote:      {result['quote']}")
    if result.get('confidence'):
        print(f"Confidence: {result['confidence']}")
    if result.get('relevant') is not None and not result['relevant']:
        print("⚠️ Irrelevant question")
    print()