"""Production wiring: real embeddings plus an LLM judge.

Setup (see README, "Setup and API keys"):

    pip install -e ".[openai]"
    cp .env.example .env        # then put your key in .env
    python examples/with_openai.py

Every setting is read from the environment (or .env):

    OPENAI_API_KEY          required
    OPENAI_BASE_URL         optional; point at any OpenAI-compatible server
    ENGRAM_LLM_MODEL        optional; default gpt-4o-mini
    ENGRAM_EMBEDDING_MODEL  optional; default text-embedding-3-small
"""

import json
import os
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()  # reads .env from the current directory, if present
except ImportError:
    pass  # python-dotenv is optional; plain environment variables also work

from openai import OpenAI

from engramdb import Brain, LLMJudge, Observation, Polarity

if not os.getenv("OPENAI_API_KEY"):
    sys.exit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")

client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)  # api key read from OPENAI_API_KEY
LLM_MODEL = os.getenv("ENGRAM_LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("ENGRAM_EMBEDDING_MODEL", "text-embedding-3-small")


class OpenAIEmbedder:
    def embed(self, texts):
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=list(texts))
        return [item.embedding for item in response.data]


def complete_json(system: str, user: str) -> dict:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return json.loads(response.choices[0].message.content)


brain = Brain(embedder=OpenAIEmbedder(), judge=LLMJudge(complete_json))

for text in ["Isn't a morning person", "Hates meetings before 10am", "Loves sunrise hikes on weekends"]:
    polarity = Polarity.LIKE if "Loves" in text else Polarity.DISLIKE
    result = brain.observe(Observation("u1", text, polarity=polarity))
    print(result.action, text)

print(brain.prompt_block("u1", message="schedule a call with the team"))
