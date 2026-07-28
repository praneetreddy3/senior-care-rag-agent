"""Agentic RAG graph:  retrieve -> grade -> generate -> verify.

Implemented with LangGraph when available; otherwise the same nodes run as a
plain sequential pipeline (identical logic, no external dependency). Every node
degrades gracefully when no LLM key is present so the demo always returns an
answer with visible sources.
"""
from __future__ import annotations

import re
from typing import List, Optional, TypedDict

from . import llm
from .config import settings
from .ingest import Chunk
from .pipeline import Retriever, get_retriever

SYSTEM = (
    "You are a warm, patient, and reassuring assistant helping older adults and "
    "their caregivers understand senior-care programs. Speak simply and kindly, as "
    "you would to a worried family member. When it feels natural, open with a brief, "
    "genuine acknowledgement, then give the answer clearly.\n\n"
    "Rules you must always follow:\n"
    "- Answer ONLY using the provided context. If the context does not contain the "
    "answer, gently say you don't have that information.\n"
    "- Never give individualized medical, legal, or benefits advice — kindly suggest "
    "speaking with a qualified professional or their care team.\n"
    "- Cite sources inline using [n] that map to the numbered context passages."
)

# Warm small-talk handling (greetings/thanks) so the bot feels human instead of
# replying "I don't have that information" to a friendly hello.
_SMALLTALK = re.compile(
    r"^(hi|hii|hey|hello|yo|hiya|good\s(morning|afternoon|evening)|how\sare\syou"
    r"|how'?s\sit\sgoing|what'?s\sup|thanks|thank\syou|thankyou|bye|goodbye|"
    r"good\snight|see\syou|take\scare)[\s!.,'a-z]*$",
    re.IGNORECASE,
)
SMALLTALK_SYSTEM = (
    "You are a warm, friendly senior-care assistant. Reply to the user's greeting or "
    "small talk in one or two short, kind sentences, then gently invite them to ask "
    "about senior-care topics (PACE, Medicare/Medicaid, medication safety, fall "
    "prevention, caregiver support). Be genuine; do not state facts you're unsure of."
)


# Never treat something with a real topic/question in it as small talk.
_TOPICAL = re.compile(
    r"(pace|medicare|medicaid|eligib|enroll|fall|medication|caregiver|nursing|"
    r"benefit|cost|cover|\?)",
    re.IGNORECASE,
)


def smalltalk_reply(query: str) -> Optional[str]:
    """Return a warm reply if the message is just a greeting/thanks, else None."""
    q = query.strip()
    if not _SMALLTALK.match(q) or _TOPICAL.search(q) or len(q.split()) > 6:
        return None
    if llm.settings.llm_enabled:
        reply = llm.generate(prompt=query, system=SMALLTALK_SYSTEM, temperature=0.6)
        if reply:
            return reply
    return (
        "Hello! I'm doing well, thank you for asking. 😊 I'm here to help with "
        "questions about senior-care programs — things like PACE eligibility, "
        "Medicare and Medicaid, medication safety, fall prevention, or caregiver "
        "support. What would you like to know?"
    )


class AgentState(TypedDict, total=False):
    query: str
    docs: List[dict]          # [{text, title, source, score}]
    relevant: List[dict]
    answer: str
    citations: List[dict]
    grounded: bool
    notes: List[str]
    attempts: int             # self-correction retry counter
    retry: bool               # set by verify when grounding fails
    graded_found: int         # relevant passages found before fallback


# Max self-correction retries: if verify says the answer isn't grounded, the agent
# widens retrieval and tries again, up to this many times.
MAX_RETRIES = 1


# ---- helpers ---------------------------------------------------------------
def _format_context(docs: List[dict]) -> str:
    return "\n\n".join(
        f"[{i + 1}] (source: {d['source']} — {d['title']})\n{d['text']}"
        for i, d in enumerate(docs)
    )


def _keyword_overlap(query: str, text: str) -> float:
    q = {w for w in query.lower().split() if len(w) > 3}
    if not q:
        return 0.0
    t = set(text.lower().split())
    return len(q & t) / len(q)


def _citations(docs: List[dict]) -> List[dict]:
    return [
        {
            "n": i + 1,
            "source": d["source"],
            "title": d["title"],
            "score": d["score"],
            "text": d["text"],          # the actual chunk, for the "show passage" UI
        }
        for i, d in enumerate(docs)
    ]


def condense_question(query: str, history: List[dict]) -> str:
    """Turn a follow-up into a standalone question using chat history.

    This is what makes multi-turn work: "what about someone under 55?" becomes
    "Is someone under 55 eligible for PACE?" so retrieval has enough context.
    Falls back to the raw query if there's no history or no LLM.
    """
    if not history or not llm.settings.llm_enabled:
        return query
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    rewritten = llm.generate(
        prompt=(
            f"Conversation so far:\n{convo}\n\n"
            f"Follow-up question: {query}\n\n"
            "Rewrite the follow-up as a standalone question that includes any "
            "context needed to answer it on its own. Return ONLY the question."
        ),
        temperature=0.0,
    )
    return (rewritten or query).strip()


# ---- nodes -----------------------------------------------------------------
def make_nodes(retriever: Retriever):
    def retrieve(state: AgentState) -> AgentState:
        # On a self-correction retry, widen the search to pull in more context.
        retrying = state.get("retry", False)
        k = settings.top_k * 2 if retrying else settings.top_k
        results = retriever.retrieve(state["query"], top_k=k)
        docs = [
            {
                "text": c.text,
                "title": c.title,
                "source": c.source,
                "score": round(score, 4),
            }
            for c, score in results
        ]
        note = f"retrieved {len(docs)} passages" + (
            " (widened after weak grounding)" if retrying else ""
        )
        return {"docs": docs, "notes": state.get("notes", []) + [note], "retry": False}

    def grade(state: AgentState) -> AgentState:
        """Keep only context passages that are actually relevant to the query."""
        docs = state["docs"]
        relevant: List[dict] = []
        if settings.llm_enabled:
            for d in docs:
                verdict = llm.generate(
                    prompt=(
                        f"Question: {state['query']}\n\nPassage:\n{d['text']}\n\n"
                        "Is this passage relevant and useful for answering the "
                        "question? Answer strictly 'yes' or 'no'."
                    ),
                    temperature=0.0,
                )
                if verdict and verdict.strip().lower().startswith("y"):
                    relevant.append(d)
        else:
            relevant = [d for d in docs if _keyword_overlap(state["query"], d["text"]) > 0]
        found = len(relevant)  # how many were judged relevant BEFORE any fallback
        # Fall back to top passages if grading filtered everything out.
        if not relevant and docs:
            relevant = docs[:2]
        note = f"graded: {found}/{len(docs)} passages relevant"
        return {
            "relevant": relevant,
            "graded_found": found,
            "notes": state.get("notes", []) + [note],
        }

    def generate(state: AgentState) -> AgentState:
        docs = state["relevant"]
        citations = _citations(docs)
        context = _format_context(docs)
        answer = llm.generate(
            prompt=f"Context:\n{context}\n\nQuestion: {state['query']}\n\nAnswer:",
            system=SYSTEM,
        )
        if answer is None:
            # Extractive fallback: return the most relevant passage(s) verbatim.
            snippet = docs[0]["text"] if docs else "No relevant information found."
            answer = (
                "LLM generation is disabled (no API key set), so here is the most "
                f"relevant source passage:\n\n\"{snippet}\" [1]"
            )
        return {"answer": answer, "citations": citations}

    def verify(state: AgentState) -> AgentState:
        """Self-check that the answer is grounded in the retrieved context."""
        if not settings.llm_enabled:
            grounded = bool(state.get("relevant"))
            note = "verification skipped (no LLM); grounded flag from retrieval"
            return {"grounded": grounded, "notes": state.get("notes", []) + [note]}
        context = _format_context(state["relevant"])
        verdict = llm.generate(
            prompt=(
                f"Context:\n{context}\n\nAnswer:\n{state['answer']}\n\n"
                "Ignoring any greeting, empathy, or conversational wording, is every "
                "FACTUAL claim in the answer supported by the context? "
                "Reply strictly 'yes' or 'no'."
            ),
            temperature=0.0,
        )
        grounded = bool(verdict and verdict.strip().lower().startswith("y"))
        attempts = state.get("attempts", 0)
        notes = state.get("notes", [])
        if grounded:
            return {"grounded": True, "notes": notes + ["verified: grounded"]}
        if attempts < MAX_RETRIES:
            # Self-correct: widen retrieval and try again.
            return {
                "grounded": False,
                "attempts": attempts + 1,
                "retry": True,
                "notes": notes + ["verified: NOT grounded → self-correcting (wider retrieval)"],
            }
        return {"grounded": False, "notes": notes + ["verified: NOT fully grounded (after retry)"]}

    return retrieve, grade, generate, verify


# ---- graph assembly --------------------------------------------------------
class Agent:
    def __init__(self, retriever: Optional[Retriever] = None):
        self.retriever = retriever or get_retriever()
        self.retrieve, self.grade, self.generate, self.verify = make_nodes(
            self.retriever
        )
        self._graph = self._build_graph()

    def _build_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph

            g = StateGraph(AgentState)
            g.add_node("retrieve", self.retrieve)
            g.add_node("grade", self.grade)
            g.add_node("generate", self.generate)
            g.add_node("verify", self.verify)
            g.add_edge(START, "retrieve")
            g.add_edge("retrieve", "grade")
            g.add_edge("grade", "generate")
            g.add_edge("generate", "verify")

            # Self-correction: if verify set retry, loop back to retrieve; else end.
            def route_after_verify(state: AgentState):
                return "retrieve" if state.get("retry") else END

            g.add_conditional_edges(
                "verify", route_after_verify, {"retrieve": "retrieve", END: END}
            )
            return g.compile()
        except Exception as exc:  # noqa: BLE001 - fall back to sequential
            print(f"[agent] LangGraph unavailable ({exc}); using sequential runner.")
            return None

    def run(self, query: str) -> AgentState:
        chat = smalltalk_reply(query)
        if chat is not None:
            return {
                "query": query, "answer": chat, "citations": [], "relevant": [],
                "grounded": True, "notes": ["small talk — no retrieval needed"],
            }
        if self._graph is not None:
            return self._graph.invoke({"query": query})
        # Sequential fallback with identical logic, including the self-correction loop.
        state: AgentState = {"query": query}
        for _ in range(MAX_RETRIES + 1):
            for node in (self.retrieve, self.grade, self.generate, self.verify):
                state.update(node(state))
            if not state.get("retry"):
                break
        return state

    def stream(self, query: str, history: Optional[List[dict]] = None):
        """Generator for a chat turn. Yields events in order:
        one {"type":"sources"}, then many {"type":"token"}, then one {"type":"done"}.

        Reuses the same retrieve/grade/verify nodes as run(); only the generate
        step is streamed token-by-token. Multi-turn is handled by condensing the
        follow-up against history before retrieval.
        """
        history = history or []
        chat = smalltalk_reply(query)
        if chat is not None:
            yield {"type": "sources", "citations": [], "condensed": query}
            yield {"type": "token", "text": chat}
            yield {"type": "done", "answer": chat, "grounded": True,
                   "citations": [], "trace": ["small talk — no retrieval needed"]}
            return
        condensed = condense_question(query, history)
        state: AgentState = {"query": condensed}
        state.update(self.retrieve(state))
        state.update(self.grade(state))
        # Corrective retrieval: if nothing graded as relevant, widen the search once
        # before answering (a streaming-friendly form of the self-correction loop).
        if state.get("graded_found", 0) == 0:
            state["retry"] = True
            state.update(self.retrieve(state))
            state.update(self.grade(state))
        docs = state["relevant"]
        citations = _citations(docs)
        yield {"type": "sources", "citations": citations, "condensed": condensed}

        context = _format_context(docs)
        answer_parts: List[str] = []
        if llm.settings.llm_enabled:
            prompt = f"Context:\n{context}\n\nQuestion: {condensed}\n\nAnswer:"
            for tok in llm.generate_stream(prompt, system=SYSTEM):
                answer_parts.append(tok)
                yield {"type": "token", "text": tok}
            answer = "".join(answer_parts)
        else:
            snippet = docs[0]["text"] if docs else "No relevant information found."
            answer = (
                "LLM generation is disabled (no API key set), so here is the most "
                f'relevant source passage:\n\n"{snippet}" [1]'
            )
            yield {"type": "token", "text": answer}

        state["answer"] = answer
        state.update(self.verify(state))
        yield {
            "type": "done",
            "answer": answer,
            "grounded": bool(state.get("grounded", False)),
            "citations": citations,
            "trace": state.get("notes", []),
        }


def answer_question(query: str, agent: Optional[Agent] = None) -> AgentState:
    agent = agent or Agent()
    return agent.run(query)
