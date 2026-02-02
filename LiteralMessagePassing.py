# ---------- LiteralMessagePassing.py (base class + helpers) ----------
from __future__ import annotations
from abc import ABC
import os
import re
import json
import asyncio
import regex
import networkx as nx
from typing import Optional, Tuple, Dict, Any, List

from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_ollama import ChatOllama
from langchain.schema import messages_to_dict

from CodeSolvers import SolveMatchingGreedyMaximal, SolveMatchingEdmonds, SolveColoringBacktrackPython, \
    SolveColoringDsaPython, SolvePolyColoringTentativeCommitPython, SolveNPColoringTentativeCommitPython, \
    SolveMatchingPriorityStatus, SolveNaiveMatching, SolveCentralMISMatching, matching_assignment, \
    SolveMatchingBacktrack, SolveVertexCoverPriority, SolveVertexCoverBacktrack
from globals_ import *
from optimal_solutions_scripts import max_matching_size
from optimal_solver_graph_coloring import _solve_max_3color_edges

# If you keep a global `names` list elsewhere, import it.
# For parsing sanity here, we fall back to allowing any string keys.
try:
    from utils import names
except Exception:
    names = None  # fallback: skip the "name in names" check


def _normalize_model_key(name: str) -> str:
    n = name.lower()
    aliases = {
        "meta-llama/llama-3.1-8b-instruct:free": "llama3.1",
        "google/gemini-2.5-flash": "gemini-2.5-flash",
        "google/gemini-2.0-flash": "gemini-2.0-flash",
        "llama3.1:8b": "llama3.1",
        "llama3": "llama3",
    }
    return aliases.get(n, n)


def parse_messages_from_text(response: str) -> Optional[Dict[str, str]]:
    """
    Try to find the first valid JSON object in a string and return it if it looks like
    neighbor->message mapping.
    """
    if not response:
        return None
    response = response.replace("\\n", "\n")
    pattern = r'(\{.*\})'
    matches = regex.findall(pattern, response, regex.DOTALL, overlapped=True)
    if not matches:
        return None

    for json_candidate in matches:
        try:
            out = json.loads(json_candidate)
            if not isinstance(out, dict):
                continue
            # All values must be strings; keys are neighbor names.
            if not all(isinstance(v, str) for v in out.values()):
                continue
            # If we have a global names list, enforce it; otherwise accept any keys.
            if names is not None and not all(k in names for k in out.keys()):
                continue
            return out
        except json.JSONDecodeError:
            continue
    return None


# ---- Rate limiter presets + a safe default ----
RATE_LIMITER_KWARGS = {
    "gpt-4o-mini": dict(requests_per_second=10, check_every_n_seconds=0.1, max_bucket_size=200),
    "gpt-4.1-mini": dict(requests_per_second=5,  check_every_n_seconds=0.1, max_bucket_size=20),
    "gpt-4o":       dict(requests_per_second=1,  check_every_n_seconds=0.1, max_bucket_size=20),
    "o1":           dict(requests_per_second=1,  check_every_n_seconds=0.1, max_bucket_size=20),
    "o1-mini":      dict(requests_per_second=1,  check_every_n_seconds=0.1, max_bucket_size=20),
    "o3-mini":      dict(requests_per_second=10, check_every_n_seconds=0.1, max_bucket_size=20),
    "o4-mini":      dict(requests_per_second=5,  check_every_n_seconds=0.2, max_bucket_size=20),
    "llama3":       dict(requests_per_second=20, check_every_n_seconds=0.1, max_bucket_size=20),
    "llama3.1":     dict(requests_per_second=20, check_every_n_seconds=0.1, max_bucket_size=20),

    # OpenRouter free model (works via OpenAI-compatible API)
    "meta-llama/llama-3.1-8b-instruct:free": dict(
        requests_per_second=0.3, check_every_n_seconds=0.25, max_bucket_size=1
    ),
    # Gemini presets (if you use them)
    "gemini-2.0-flash":               dict(requests_per_second=100, check_every_n_seconds=0.1, max_bucket_size=20),
    "gemini-2.0-flash-lite":          dict(requests_per_second=100, check_every_n_seconds=0.1, max_bucket_size=20),
    "gemini-2.5-flash-preview-04-17": dict(requests_per_second=10,  check_every_n_seconds=0.1, max_bucket_size=20),
    "gemini-2.5-flash-preview-04-17-thinking": dict(requests_per_second=10, check_every_n_seconds=0.1, max_bucket_size=20),
    "gemini-2.5-pro-exp-03-25":       dict(requests_per_second=2,   check_every_n_seconds=0.1, max_bucket_size=10),
    "gemini-2.5-pro-preview-03-25":   dict(requests_per_second=2,   check_every_n_seconds=0.1, max_bucket_size=1),
    "gemini-2.5-pro-preview-05-06":   dict(requests_per_second=2,   check_every_n_seconds=0.1, max_bucket_size=1),
    "gemini-1.5-pro":                 dict(requests_per_second=100, check_every_n_seconds=0.1, max_bucket_size=20),
    # Claude presets (if you use them)
    "claude-3-5-haiku-20241022":      dict(requests_per_second=10,  check_every_n_seconds=0.1, max_bucket_size=20),
    "claude-3-opus-20240229":         dict(requests_per_second=10,  check_every_n_seconds=0.1, max_bucket_size=20),
    "claude-3-7-sonnet-20250219":     dict(requests_per_second=10,  check_every_n_seconds=0.1, max_bucket_size=20),
    "claude-3-7-sonnet-20250219-thinking": dict(requests_per_second=10, check_every_n_seconds=0.1, max_bucket_size=20),
    "gemini-2.5-flash": dict(
        requests_per_second=3,   # conservative; OpenRouter will queue anyway
        check_every_n_seconds=0.1,
        max_bucket_size=6,
    ),

    "deepseek/deepseek-r1": dict( requests_per_second=3,   # conservative; OpenRouter will queue anyway
        check_every_n_seconds=0.1,
        max_bucket_size=6,)

}

SAFE_DEFAULT_RL = dict(requests_per_second=5, check_every_n_seconds=0.1, max_bucket_size=20)


class LiteralMessagePassing(ABC):
    """
    Base class: async, non-hanging, rate-limited, provider-agnostic.

    OpenRouter support:
      - Set env in PowerShell before running:
          $env:OPENAI_API_KEY = "sk-or-v1_..."
          $env:OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
      - Use model_provider="openai" and model_name="meta-llama/llama-3.1-8b-instruct:free"
    """

    """
    Base class: async, non-hanging, rate-limited, provider-agnostic.

    OpenRouter (OpenAI-compatible) options:

      Option A (recommended): OPENROUTER_* env vars
        $env:OPENROUTER_API_KEY = "sk-or-..."
        $env:OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
        Use model_provider="openai" and model_name like "google/gemini-2.5-flash"
        or "meta-llama/llama-3.1-8b-instruct:free"

      Option B: OPENAI_* env vars pointing to OpenRouter
        $env:OPENAI_API_KEY = "sk-or-..."
        $env:OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
        model_provider="openai"
    """

    def __init__(
            self,
            graph: nx.Graph,
            problem_level: ProblemLevel,
            model_name: str = "gpt-4o",
            model_provider: str = "openai",
            chain_of_thought: bool = False,
            is_must_send: bool = False,
    ):
        self.problem_level = problem_level
        self.graph = graph
        self.model_name = model_name
        self.chain_of_thought = chain_of_thought

        # ---- rate limiter (never crash on unknown model) ----
        model_key = _normalize_model_key(model_name)
        rate_kwargs = RATE_LIMITER_KWARGS.get(model_key, SAFE_DEFAULT_RL)
        rate_limiter = InMemoryRateLimiter(**rate_kwargs)

        # ---- build model client ----
        chat_kwargs = {}

        # Anthropic thinking variants
        if model_name.startswith("claude-3-7-sonnet") and model_name.endswith("thinking"):
            model_name = "-".join(model_name.split("-")[:-1])
            chat_kwargs.update({
                "max_tokens": 5000,
                "thinking": {"type": "enabled", "budget_tokens": 2000}
            })

        # Gemini thinking flag
        if model_name == "gemini-2.5-flash-preview-04-17-thinking":
            model_name = "gemini-2.5-flash-preview-04-17"
            chat_kwargs.update({"thinking_budget": 24000, "include_thoughts": True})

        # Local Ollama
        if model_provider == "ollama":
            self.model = ChatOllama(model=model_name, base_url="http://localhost:11434")

        else:
            # Route OpenRouter keys to the OpenAI-compatible provider automatically
            use_openrouter = bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_BASE_URL"))
            provider_effective = model_provider

            if use_openrouter and model_provider in ("openrouter", "openai"):
                provider_effective = "openai"  # OpenRouter is OpenAI-compatible
                # Prefer OPENROUTER_* if present, else OPENAI_*
                api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
                base_url = os.getenv("OPENROUTER_BASE_URL") or os.getenv(
                    "OPENAI_BASE_URL") or "https://openrouter.ai/api/v1"
                if api_key:
                    chat_kwargs["api_key"] = api_key
                chat_kwargs["base_url"] = base_url

            self.model = init_chat_model(
                model_name,
                model_provider=provider_effective,
                rate_limiter=rate_limiter,
                timeout=45,  # SDK-level timeout
                **chat_kwargs,
            ).with_retry(stop_after_attempt=3)

        # ---- timeout + concurrency guard for the actual LLM call ----
        self.request_timeout_s = int(os.getenv("LMP_REQUEST_TIMEOUT", "60"))
        self.sema = asyncio.Semaphore(int(os.getenv("LMP_MAX_CONCURRENCY", "1")))

        # ---- LangGraph single-node wrapper ----
        self.workflow = StateGraph(state_schema=MessagesState)

        async def call_model(state: MessagesState):
            async with self.sema:
                try:
                    response = await asyncio.wait_for(
                        self.model.ainvoke(state["messages"]),
                        timeout=self.request_timeout_s,
                    )
                    # IMPORTANT: return a list so LangGraph appends correctly
                    return {"messages": [response]}
                except asyncio.TimeoutError:
                    return {"messages": [AIMessage(content="[TIMEOUT] No response")]}
                except Exception as e:
                    return {
                        "messages": [AIMessage(content=f"[ERROR] {type(e).__name__}: {e}")]}  # parseable fallback

        self.workflow.add_edge(START, "model")
        self.workflow.add_node("model", call_model)

        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)

        # ---- per-node bookkeeping ----
        self.messages: Dict[int, Dict[str, str]] = {v: {} for v in graph.nodes()}
        self.chat_history = {v: [] for v in graph.nodes()}
        self.num_fallbacks = [0 for _ in graph.nodes()]
        self.num_failed_json_parsings_after_retry = [0 for _ in graph.nodes()]
        self.num_failed_answer_parsings_after_retry = [0 for _ in graph.nodes()]

        # ---- prompts ----
        boot1 = """You are an agent that is connected with other agents (your neighbors), who you communicate with. Your neighbors can in turn communicate with their neighbors and so forth. {short_task_description}
The rules are as follows:
1. There are {num_agents} agents in total. Everybody has a unique name. Your name is {own_name}.
2. You can only communicate with your immediate neighbors ({neighbors}). You cannot see or directly communicate with anyone else, unless information is relayed by intermediate agents.
3. You can exchange text-based messages with your neighbors in rounds. In each round, you will first receive the last messages sent by your neighbors and then be asked to generate your response messages which your neighbors receive in the next round. This process repeats for {num_rounds} rounds of message passing. Importantly, the process is synchronous: Every agent decides on which messages to send at the same time and sees the messages from other agents only in the next round.
4. Everybody (including you) decides what to share or request from neighbors. In every round, think step-by-step about the next set of messages you want to send. Output a JSON string that contains your response messages.
5. The messages you send to your neighbors are formatted as JSON. For example, if your neighbors are Alan and Bob, your output should look as follows:
{{
  "Alan": "Message that will be sent to Alan.",
  "Bob": "Message that will be sent to Bob."
}}"""
        if is_must_send:
            boot2 = """It is ***mandatory*** to send a message to every neighbor in every round."""
        else:
            boot2 = """It is not mandatory to send a message to every neighbor in every round. If you do not want to send a message to a particular neighbor, you may omit their name from the JSON."""
        boot3 = """6. After {num_rounds} message passes, you have to solve the following task: {long_problem_description}"""
        self.bootstrap_template = PromptTemplate.from_template(boot1 + " " + boot2 + " " + boot3)

        self.cot_prompt = "Elaborate your chain of thought step-by-step first, then output the messages for your neighbors."
        self.cot_prompt_final_prediction = "\n\nElaborate your chain of thought step-by-step first, then answer the following: "
        if not chain_of_thought:
            self.cot_prompt = ""
            self.cot_prompt_final_prediction = ""

        self.bootstrap_ask_for_first_messages = PromptTemplate.from_template(
            "What are the first messages you want to send to your neighbors? {cot_prompt} Output your messages in JSON format as specified earlier."
        )
        self.format_instructions = PromptTemplate.from_template(
            "{question} Format your answer as follows: '### Final Answer ###', followed by your final answer. Don't use any text for your final answer except one of these valid options: {valid_answers}"
        )

        # Derived classes must set:
        # - self.rounds
        # - self.short_task_description
        # - self.long_problem_description
        # - self.question_for_prediction

    # ---------- small helpers some agents rely on ----------
    def get_symbolic_intro(self) -> str:
        return self._intro_mode_block() + self._intro_generic_rules_suffix()

    def _intro_generic_rules_suffix(self) -> str:
        return (
            "**Protocol Rules:**\n"
            "- If any neighbor deviates—e.g., does not share their state even though you expect them —**remind them of the expected behavior** and instruct them to follow the protocol.\n"
            "- At the last round you must come up with a decision.\n"

        )

    # Derived classes should implement this if they use get_symbolic_intro()
    def _intro_mode_block(self) -> str:
        return ""

    # ---------- fallbacks ----------
    async def fallback_json_request(self, node_id: int) -> Dict[str, str]:
        self.num_fallbacks[node_id] += 1
        user_message = HumanMessage(content="Your messages could not be parsed into JSON. Please check your response and try again.")
        config = {"configurable": {"thread_id": str(node_id)}}
        response = await self.app.ainvoke({'messages': [user_message]}, config=config)
        self.chat_history[node_id] = messages_to_dict(response['messages'])
        last_message = self.chat_history[node_id][-1]['data']['content']
        messages_sent = parse_messages_from_text(last_message)
        if messages_sent is None:
            self.num_failed_json_parsings_after_retry[node_id] += 1
            return {}
        return messages_sent

    async def fallback_answer_request(self, node_id: int) -> Optional[str]:
        self.num_fallbacks[node_id] += 1
        user_message = HumanMessage(content="Your answer could not be parsed. Please check your answer and try again.")
        config = {"configurable": {"thread_id": str(node_id)}}
        response = await self.app.ainvoke({'messages': [user_message]}, config=config)
        self.chat_history[node_id] = messages_to_dict(response['messages'])
        last_message = self.chat_history[node_id][-1]['data']['content']
        messages_sent = self.parse_answer(node_id, last_message)
        if messages_sent is None:
            self.num_failed_answer_parsings_after_retry[node_id] += 1
            return None
        return messages_sent

    # ---------- parsing ----------
    def parse_answer(self, node_id: int, message: str) -> Optional[str]:
        valid_answers = self.get_valid_answers(node_id)
        if not valid_answers:
            return None
        pattern = r"### Final Answer ###\s*(" + "|".join(re.escape(ans) for ans in valid_answers) + r")"
        parsed_answer = re.search(pattern, message)
        if parsed_answer:
            return parsed_answer.group(1)
        return None

    async def parse_response_to_dict(
        self, message_dict, node_id: int, last_round: bool = False
    ) -> Tuple[Optional[Dict[str, str]], Optional[asyncio.Task]]:
        last_message = message_dict[-1]['data']['content'] if message_dict else ""
        if not last_round:
            messages_sent = parse_messages_from_text(last_message)
        else:
            messages_sent = self.parse_answer(node_id, last_message)

        if messages_sent is None:
            if not last_round:
                return None, self.fallback_json_request(node_id)
            else:
                return None, self.fallback_answer_request(node_id)
        return messages_sent, None

    async def update_messages(self, results):
        fallback_tasks = []
        fallback_nodes = []
        for node, result in zip(self.graph.nodes(), results):
            message, fallback = result
            if message is None:
                fallback_nodes.append(node)
                fallback_tasks.append(fallback)
            else:
                self.messages[node] = message

        if fallback_tasks:
            recovered = await asyncio.gather(*fallback_tasks)
            for node, message in zip(fallback_nodes, recovered):
                self.messages[node] = message

    # ---------- bootstrap (sequential to avoid free-tier bursts) ----------
    async def bootstrap(self):
        results = []
        for v in self.graph.nodes():
            r = await self.bootstrap_node(v)
            results.append(r)
        await self.update_messages(results)

    async def bootstrap_node(self, node_id: int):
        """Bootstraps node with task-specific instructions."""
        params = self.get_bootstrap_parameters(node_id)
        system_message = SystemMessage(content=self.bootstrap_template.format(**params))
        user_message = HumanMessage(content=self.bootstrap_ask_for_first_messages.format(cot_prompt=self.cot_prompt))
        config = {"configurable": {"thread_id": str(node_id)}}
        response = await self.app.ainvoke({"messages": [system_message, user_message]}, config=config)
        self.chat_history[node_id] = messages_to_dict(response['messages'])
        return await self.parse_response_to_dict(self.chat_history[node_id], node_id)

    def get_bootstrap_parameters(self, node_id: int) -> Dict[str, str]:
        neighbors = ", ".join(str(self.graph.nodes[i]["name"]) for i in self.graph.neighbors(node_id))
        return {
            "short_task_description": self.short_task_description,
            "long_problem_description": self.long_problem_description,
            "num_agents": self.graph.order(),
            "own_name": self.graph.nodes[node_id]["name"],
            "neighbors": neighbors,
            "num_rounds": self.rounds,
        }

    # ---------- message passing ----------
    async def message_passing(self, node_id: int, rounds_left: int, messages: Dict[str, str]):
        last_round = rounds_left == 0
        if last_round:
            messages_str = "Message passing has finished, here are the last messages you got from your neighbors:\n\n"
        else:
            messages_str = "These are the messages from your neighbors:\n\n"
        for name, msg in messages.items():
            messages_str += f"Message from {name}:\n\n{msg}\n\n"

        neighbour_names = [str(self.graph.nodes[i]["name"]) for i in self.graph.neighbors(node_id)]
        silent_neighbors = [name for name in neighbour_names if name not in messages]
        if silent_neighbors:
            messages_str += f"The following neighbors did not send you a messages in this round: {', '.join(silent_neighbors)}\n\n"

        if not last_round:
            neighbors = ", ".join(neighbour_names)
            messages_str += f"{self.cot_prompt} Output your messages in JSON format as specified earlier. You have {rounds_left} rounds of communication left before you need to decide. Your neighbors are: {neighbors} "
            if rounds_left == 1:
                messages_str += "These are the last messages that your neighbors will receive from you."
        else:
            messages_str += self.cot_prompt_final_prediction + self.format_instructions.format(
                question=self.question_for_prediction, valid_answers=", ".join(self.get_valid_answers(node_id))
            )

        user_message = HumanMessage(content=messages_str)
        config = {"configurable": {"thread_id": str(node_id)}}
        response = await self.app.ainvoke({'messages': [user_message]}, config=config)
        self.chat_history[node_id] = messages_to_dict(response['messages'])
        return await self.parse_response_to_dict(self.chat_history[node_id], node_id, last_round)

    async def pass_messages(self):
        """Executes synchronous message passing rounds."""
        for r in range(1, self.rounds + 1):
            rounds_left = self.rounds - r
            all_messages_sent = {v: self.messages[v] for v in self.graph.nodes()}

            tasks = []
            for v in self.graph.nodes():
                name = self.graph.nodes[v]["name"]
                messages_received = {}
                for neigh in self.graph.neighbors(v):
                    if name in all_messages_sent[neigh]:
                        neigh_name = self.graph.nodes[neigh]["name"]
                        messages_received[neigh_name] = all_messages_sent[neigh][name]
                tasks.append(self.message_passing(node_id=v, rounds_left=rounds_left, messages=messages_received))

            results = await asyncio.gather(*tasks)
            await self.update_messages(results)

        return self.get_final_answers(self.messages, self.graph.order())

    # ---------- results ----------
    def get_final_answers(self, last_messages: Dict[int, Dict[str, str]], num_nodes: int):
        answers = [None] * num_nodes
        for node in last_messages.keys():
            answers[node] = last_messages[node]
        return answers

    def get_transcripts(self):
        """Returns the full transcript of all messages exchanged."""
        return {self.graph.nodes[v]['name']: self.chat_history[v] for v in self.graph.nodes()}

    # ---------- abstract hooks ----------
    def get_score(self, answers):
        raise NotImplementedError

    def get_SSE_easy(self, answers):
        raise NotImplementedError

    def get_SSE_hard(self, answers):
        raise NotImplementedError

    def get_norm_SSE_easy(self, answers):
        return 1 / (1 + self.get_SSE_easy(answers))

    def get_norm_SSE_hard(self, answers):
        return 1 / (1 + self.get_SSE_hard(answers))
        # ---------- abstract hooks ----------
    def get_score_opt(self, answers):
        raise NotImplementedError

    def get_required_calculations(self, answers):
        raise NotImplementedError

    def get_valid_answers(self, node_id: int):
        raise NotImplementedError


####----------- Consensus Majority-Update Consensus  -----------####



"""
"**Algorithm Name:** Synchronous Majority-Update Consensus (Fault-Free Variant)\n"
            "**Reference:** Pease, Shostak, and Lamport (1980), 'Reaching agreement in the presence of faults'. "
            "Adapted here for the fault-free, synchronous setting.\n\n"
"""



####----------- LeaderElection FloodMax  -----------####


class LeaderElection(LiteralMessagePassing):
    def __init__(self, graph, rounds, model_name="gpt-4o", model_provider="openai", chain_of_thought=True,
                 is_must_send=False):
        self.rounds = rounds
        self.short_task_description = "Your task is to collaboratively solve the problem of electing a single leader."
        self.long_problem_description = (
            "You will be requested to state whether or not you are the leader. The response should either be 'Yes' or 'No'. "
            "The final result should be such that exactly one agent responds with 'Yes' and all others say 'No' as there should be exactly one leader."
        )
        self.question_for_prediction = "Are you the leader?"
        super().__init__(graph, model_name, model_provider, chain_of_thought, is_must_send=is_must_send)

    def get_score(self, answers: list[str]) -> float:
        """
        Continuous score in [0, 1] for Leader Election.

        - Perfect (1.0) when exactly one agent says "Yes" AND all answers are valid ("Yes"/"No").
        - 0 when zero valid answers, or everyone invalid.
        - Penalizes extra/missing leaders linearly with n.
        - Invalid/None answers reduce the score via a validity factor.
        """
        n = len(answers)
        if n == 0:
            return 0.0

        valid_values = {"Yes", "No"}
        valid_count = sum(1 for a in answers if a in valid_values)
        yes_count = sum(1 for a in answers if a == "Yes")

        # Validity factor: fraction of agents who answered with a valid token.
        validity = valid_count / n

        # Triangular "distance from exactly one leader" scaled by n:
        # - 0 at yes=0 (no leader) and yes=n (everyone claims leader)
        # - 1 at yes=1 (exactly one leader)
        if yes_count <= 1:
            leader_shape = float(yes_count)  # 0 -> 0.0, 1 -> 1.0
        else:
            leader_shape = max(0.0, 1.0 - (yes_count - 1) / max(1, n - 1))

        # Final score combines correctness of tokens and closeness to exactly one leader
        return validity * leader_shape

    def get_valid_answers(self, node_id):
        return ["Yes", "No"]



####----------- Matching Greedy Maximal Matching  -----------####

#
# class MatchingPseudocodeOnlyAgent(MatchingFullySymbolicAgent):
#     """
#     Symbolic framing only (more abstract): no pseudocode, no helper list, no concrete message names.
#     Agents receive high-level protocol expectations stated in natural language.
#     """
#     def __init__(self, graph, rounds, model_name="gpt-4o-mini",
#                  model_provider="openai", chain_of_thought=True):
#         # Call the fully symbolic init, then overwrite the description with an abstract version
#         super().__init__(graph, rounds, model_name, model_provider, chain_of_thought,is_must_send = True)
#
#         self.long_problem_description = (
#                 self.get_basic_long_prob_description() + "\n\n"
#                 + self.get_symbolic_intro() + "\n\n"
#                 + "Here is the shared pseudocode that defines your behavior:\n"
#                 + self.get_symbolic_pseudo_code()
#
#         )
# class MatchingTransparentHelperFunctionsAgent(MatchingFullySymbolicAgent):
#     def __init__(self, graph, rounds, model_name="gpt-4o-mini",
#                  model_provider="openai", chain_of_thought=True):
#
#         super().__init__(graph, rounds, model_name, model_provider, chain_of_thought,is_must_send=False)
#         self.long_problem_description = (
#             self.get_basic_long_prob_description() + "\n\n"
#             + self.get_symbolic_intro() + "\n\n"
#             + "Here is a list of helper function definitions that I want you to use:\n"
#             + self.get_symbolic_functions()
#         )
#
#     # Only override the MODE block; keep problem flow in _intro_problem_unique_rules()
#     def _intro_mode_block(self) -> str:
#         return (
#             "**Mode:** Transparent Helper Functions.\n"
#             "You and all your neighbors are provided with the **same set of clearly explained helper functions**.\n\n"
#             "**Protocol Expectations (mode-specific):**\n"
#             "- You must always be either **unmatched** or **matched with exactly one** neighbor.\n"
#             "- There is **no step-by-step pseudocode**; construct your behavior **only** from the provided helper functions.\n"
#             "- The functions are **not black boxes** — you are expected to understand their internal logic and reason **explicitly** with it.\n"
#             "- Use the functions to **announce availability**, **initiate/accept/decline pairing**, and **publish matched status**.\n"
#             "- Do **not** invent operations beyond the defined functions; stay within their documented capabilities.\n\n"
#         )
# class MatchingBlackBoxHelperFunctionsAgent(MatchingFullySymbolicAgent):
#     """
#     Black-Box Helper Functions mode for Matching.
#     - Inherits the generic intro composer from MatchingFullySymbolicAgent.
#     - Only overrides `_intro_mode_block()` to describe the mode.
#     - Provides an I/O-only function list (no internal semantics).
#     """
#     def __init__(self, graph, rounds, model_name="gpt-4o-mini",
#                  model_provider="openai", chain_of_thought=True):
#         super().__init__(graph, rounds, model_name, model_provider, chain_of_thought,is_must_send=False)
#         self.long_problem_description = (
#             self.get_basic_long_prob_description() + "\n\n"
#             + self.get_symbolic_intro() + "\n\n"
#             + "Here is a list of helper function **I/O contracts** that you must use (treat them as black boxes):\n"
#             + self.get_symbolic_functions()
#         )
#
#     # -------- mode block override (do NOT override get_symbolic_intro) --------
#     def _intro_mode_block(self) -> str:
#         return (
#             "**Mode:** Black-Box Helper Functions.\n"
#             "- Construct your behavior **only** from the provided helper functions.\n"
#             "- Treat functions as **black boxes**: rely strictly on documented inputs/outputs; do **not** assume internal behavior.\n"
#             "- Do not invent new operations beyond the allowed function calls.\n\n"
#         )
#
#     def get_symbolic_functions(self):
#         # I/O contracts only (no internal semantics). These enable a greedy, conflict-safe local matching.
#         return """
#       - get_self_name()
#         Input:  None
#         Output: self_name (str)
#
#       - list_available_neighbors()
#         Input:  None
#         Output: neighbors (list[str])
#
#       - rank_neighbors_greedy(candidates: list[str])
#         Input:  candidates (list[str])
#         Output: ordered (list[str])
#
#       - attempt_match(neighbor: str)
#         Input:  neighbor (str)
#         Output: success (bool), partner (str | None)
#
#       - get_match_status()
#         Input:  None
#         Output: status (dict) with keys:
#                 {"matched": bool, "partner": str | None}
#
#       - announce_matched(partner: str)
#         Input:  partner (str)
#         Output: None
#
#       - receive_status_updates()
#         Input:  None
#         Output: updates (list[dict])
#
#
#       - mark_neighbor_unavailable(neighbor: str)
#         Input:  neighbor (str)
#         Output: None
#
#       - output_final_partner()
#         Input:  None
#         Output: partner_or_None (str)
#       """
# class MatchingHeuristicHintGuidedAgent(MatchingFullySymbolicAgent):
#     """
#     Heuristic, hint-guided mode for Matching:
#     - No strict pseudocode or fixed function script.
#     - Agents act using shared principles and local observations.
#     """
#     def __init__(self, graph, rounds, model_name="gpt-4o-mini",
#                  model_provider="openai", chain_of_thought=True):
#
#         super().__init__(graph, rounds, model_name, model_provider, chain_of_thought,is_must_send=False)
#         self.long_problem_description = (
#             self.get_basic_long_prob_description() + "\n\n"
#             + self.get_generic_h()
#             + self.get_intro()
#         )
#
#     def get_generic_h(self) -> str:
#         return (
#             "**Mode:** Heuristic, Hint-Guided (no strict protocol).\n"
#             "You coordinate using local information and shared principles—not a fixed algorithm.\n\n"
#             "**Heuristic Expectations:**\n"
#         )
#
#     def get_intro(self) -> str:
#         return (
#             "- Seek to establish **one mutual pairing** and, once paired, regard yourself as **unavailable**.\n"
#             "- When considering neighbors, rely on a **shared priority** (any consistent ordering suffices); "
#             "- prefer higher-priority opportunities when you are free.\n"
#             "- Follow a **greedy attitude**: pursue at most one viable opportunity at a time rather than exploring many simultaneously.\n"
#             "- If no opportunity is currently viable, **wait and reassess** as local information evolves, adapting decisions to the most credible available option.\n"
#         )





from abc import ABC, abstractmethod


class AgentToOverride(ABC):

    def _intro_generic_header(self) -> str:
        return (
            "This task is solved using a **shared symbolic protocol**. "
            "All others receives the same information as you**.\n\n"
            "You must follow the rules and prescribed flow exactly. "
        )

    @abstractmethod
    def _intro_mode_block(self): pass

    def _intro_generic_rules_suffix(self) -> str:
        return (
            "**Protocol Rules:**\n"
            "- At all times, you must be ready to share your current state with neighbors.\n"
            "- You are expected to send your state/decision at the end of each round, and your neighbors are expected to do the same.\n"
            "- If any neighbor deviates—e.g., does not share their state—**remind them of the expected behavior** and instruct them to follow the protocol.\n"
        )


class CompletePseudoCode(AgentToOverride, ABC):
    def __init__(self):
        self.long_problem_description = (
                self.get_basic_long_prob_description() + "\n\n"
                + self.get_symbolic_intro() + "\n\n"
                + "Here is the shared pseudocode that defines your behavior:\n"
                + self.get_symbolic_pseudo_code() + "\n\n"

        )

    def _intro_mode_block(self) -> str:
        """Default = Fully Symbolic mode description (short, no problem details)."""
        return (
            "- You may not deviate from the prescribed strategy.\n"
            "- All agents follow the exact same protocol with no creative changes.\n\n"
            "- You will be provided with the main pseudocode and the pseudocode the functions used which describes how to use them"
        )

    @abstractmethod
    def get_symbolic_pseudo_code(self): pass


class MainPseudocodeOnly(AgentToOverride, ABC):
    def __init__(self):
        self.long_problem_description = (
                self.get_basic_long_prob_description() + "\n\n"
                + self.get_symbolic_intro() + "\n\n"
                + "Here is the shared pseudocode that defines your behavior:\n"
                + self.get_symbolic_pseudo_code()
        )

    def _intro_mode_block(self) -> str:
        """Default = Fully Symbolic mode description (short, no problem details)."""
        return (
            "- You may not deviate from the prescribed strategy.\n"
            "- You will be provided with the main pseudocode only and expected to implement the functions used as you see fit."
        )


class FunctionPseudoCodeOnly(AgentToOverride):
    def __init__(self):
        self.long_problem_description = (
                self.get_basic_long_prob_description() + "\n\n"
                + self.get_symbolic_intro() + "\n\n"
                + "Here is a list of helper function definitions that I want you to use:\n"
                + self.get_symbolic_functions()
        )

    def _intro_mode_block(self) -> str:
        return (
            "You and all your neighbors are provided with the **same set of clearly explained functions**.\n\n"
            "- You must always have a group assigned.\n"
            "- You do not receive a step-by-step algorithm. Your behavior must be constructed **using the provided functions**.\n"
            "- The functions are **not black boxes** — each one is carefully explained, and you are expected to understand how it works internally.\n"
        )


class FunctionDescriptionOnly(AgentToOverride, ABC):

    def __init__(self):
        self.long_problem_description = (
                self.get_basic_long_prob_description() + "\n\n"
                + self.get_symbolic_intro() + "\n\n"
                + "Here is a list of helper function definitions that I want you to use:\n"
                + self.get_symbolic_functions()
        )

    def _intro_mode_block(self) -> str:
        return (
            "You and all your neighbors are provided with the **same set of functions** but their internal workings are treated as black boxes.\n\n"
            "- You must always have a group assigned.\n"
            "- You do not receive a step-by-step algorithm. Your behavior must be constructed **using the provided functions**.\n"
            "- You are expected to implement the functions used as you see fit."
        )

    @abstractmethod
    def get_symbolic_functions(self): pass


class HeuristicHintGuided(AgentToOverride):
    def __init__(self):
        self.long_problem_description = (
                self.get_basic_long_prob_description() + "\n\n" + self._intro_mode_block()
                + self.get_intro()
        )

    def _intro_mode_block(self):
        return (
            "**Mode:** Heuristic, Hint-Guided (no strict protocol).\n"
            "You coordinate using local information and shared principles—not a fixed algorithm.\n\n"

            "You and your neighbors coordinate based solely on limited, local information and shared behavioral principles.\n"
            "All agents, including your neighbors, receive the same high-level strategic guidance, but there is no fixed pseudocode or symbolic protocol to follow exactly. You have the freedom to decide how to act, reason, and communicate based on your understanding of these guidelines.\n"
            "**Heuristic Expectations:**\n"
        )


# ************************************************ Coloring ************************************************

# ************************************************ Matching ************************************************


class Matching(LiteralMessagePassing):
    def __init__(self, graph, rounds,problem_level, model_name="gpt-4o-mini", model_provider="openai", chain_of_thought=True,
                 is_must_send=False):

        self.rounds = rounds
        self.short_task_description = "Your task is to build as many groups as possible of two agents each which can communicate with each other."
        self.long_problem_description = self.get_basic_long_prob_description()
        self.question_for_prediction = "Please answer with the name of the neighbor you build a group with or 'None' if all your neighbors are already assigned to other groups."
        super().__init__(graph, problem_level,model_name,  model_provider, chain_of_thought,  is_must_send=is_must_send)

    def get_required_calculations(self, answers: list[str]) -> dict[str, int | Any]:
        """
        Count 'valid' edges w.r.t. agents' answers.
        Valid = (i) mutual, vertex-disjoint chosen edge; OR (ii) unchosen but covered
        by at least one endpoint matched elsewhere.
        """
        G = self.graph
        nodes = list(G.nodes())

        # --- map display name -> node id (uses node['name'] if present) ---
        def disp_name(u):
            return G.nodes[u].get("name", str(u))

        name_to_node = {disp_name(u): u for u in nodes}

        # If names may differ in case/spacing, build a relaxed map too
        relaxed = {disp_name(u).strip().lower(): u for u in nodes}

        # build once above
        node_set = set(nodes)  # membership in actual labels

        def to_node_id(token):
            # Unmatched sentinel
            if token is None:
                return None

            # Already an int? accept only if it's an actual node label
            if isinstance(token, int):
                return token if token in node_set else None

            # Normalize strings / other types
            s = str(token).strip()
            if not s or s.lower() in {"none", "null", "nil"}:
                return None

            # Exact display-name match
            if s in name_to_node:
                return name_to_node[s]

            # Numeric string → try int label (handles "14", " 14 ", "-1")
            if s.lstrip("-").isdigit():
                k = int(s)
                if k in node_set:
                    return k

            # Relaxed (lowercased) display-name match
            return relaxed.get(s.lower(), None)

        # --- normalize answers -> node ids (or None) in the same node order ---
        chosen = {u: to_node_id(ans) for u, ans in zip(nodes, answers)}
        # --- build mutual, vertex-disjoint matching (IDs only) ---
        matched = {}
        matching_edges = set()  # store as frozenset({u,v}) for order-invariance
        for u in nodes:
            v = chosen[u]
            if v is None:
                continue
            # must be mutual, and an actual edge in G
            if chosen.get(v) == u and G.has_edge(u, v):
                if u not in matched and v not in matched:
                    matched[u] = v
                    matched[v] = u
                    matching_edges.add(frozenset((u, v)))

        # --- count valid edges ---
        valid = 0
        for u, v in G.edges():
            e = frozenset((u, v))
            if e in matching_edges:
                # case (i): mutual chosen edge in the matching
                valid += 1
            elif (u in matched) or (v in matched):
                # case (ii): covered by a matched endpoint
                valid += 1
            # else: invalid (free–free edge or inconsistent proposal)

        return {"valid_edges": valid, "invalid_edges": self.graph.number_of_edges() - valid,
                "matches_size": len(matching_edges), "opt_max_matches": max_matching_size(self.graph)}

    def get_score_opt(self, answers):
        matches_found = self.get_required_calculations(answers)["matches_size"]
        matches_max = self.get_required_calculations(answers)["opt_max_matches"]
        return  (matches_found/matches_max)


    def get_score(self, answers: list[str]) -> float:
        """
        Returns a score between 0 and 1 representing the fraction of successful matches.
        A successful match = two agents naming each other as partners.
        """

        valid_edges = self.get_required_calculations(answers)["valid_edges"]
        # valid_edges = [answers[u] != answers[v] for (u, v) in self.graph.edges].count(True)
        valid_ratio = valid_edges / self.graph.number_of_edges()
        return valid_ratio

        # graph = self.graph
        # node_names = [graph.nodes[node]['name'] for node in graph.nodes]
        #
        # name_to_idx = {name: i for i, name in enumerate(node_names)}
        # matched_pairs = set()
        #
        # for node in graph.nodes:
        #     my_name = node_names[node]
        #     partner_name = answers[node]
        #
        #     # Skip if no partner chosen
        #     if partner_name == 'None':
        #         continue
        #
        #     # Partner must be a neighbor
        #     if partner_name not in [node_names[n] for n in graph.neighbors(node)]:
        #         continue
        #
        #     # Partner must choose me back
        #     partner_idx = name_to_idx[partner_name]
        #     if answers[partner_idx] == my_name:
        #         # Store pair in canonical order to avoid double counting
        #         matched_pairs.add(tuple(sorted([my_name, partner_name])))
        #
        # # Max possible matches = total nodes / 2
        # max_possible_matches = graph.order() / 2
        # score = len(matched_pairs) / max_possible_matches
        return score

    def get_valid_answers(self, node_id):
        return [self.graph.nodes[neigh]["name"] for neigh in self.graph.neighbors(node_id)] + ["None"]

    def get_basic_long_prob_description(self):
        return (
            "You will be requested to name one of your neighbors that you build a group with or 'None' if all your neighbors are already assigned to other groups and cannot be in a group with you." \
            "In the end, every agent should only be in at most one group and agents in the same group have to name each other as the second group member consistently."
        )


    def get_SSE_easy(self, answers):
        return self.get_required_calculations(answers)["invalid_edges"]**2



    def get_SSE_hard(self, answers):
        dict__ = self.get_required_calculations(answers)
        matches_found = dict__["matches_size"]
        max_matches = dict__["opt_max_matches"]
        invalid_edges = dict__["invalid_edges"]
        return (max_matches-matches_found)**2#+invalid_edges**2




###########---------- Matching incomplete  GreedyMaximal ----------###########

class MatchingTentativeCommitCode(Matching):
    def __init__(self, graph: nx.Graph, rounds: int,problem_level, **kwargs):
        super().__init__(graph=graph, rounds=rounds,problem_level = problem_level, **kwargs)
        self._order = list(sorted(self.graph.nodes()))
        self.model_name = "python"

    async def bootstrap(self):  # no-op
        return

    async def pass_messages(self):
        solver = SolveMatchingPriorityStatus(self.graph, rounds=self.rounds)
        answers = solver.answers  # list of partner names (or 'None')
        return answers


class MatchingTentativeCommitCompletePseudoCode(Matching, CompletePseudoCode):
    def __init__(self, graph, rounds,problem_level, model_name="gpt-4o-mini", model_provider="openai", chain_of_thought=True):
        Matching.__init__(self, graph, rounds,problem_level,  model_name, model_provider, chain_of_thought, is_must_send=True)
        CompletePseudoCode.__init__(self)

    def get_symbolic_pseudo_code(self):
        return """
STEP 0 — Initialize your state:
You begin with no partner, a list of neighbors you could match with, and an empty record of neighbor messages.
How: init_variables
- Set committed = "none".
- Set tentative = "none".
- Set P = list of your neighbors (options; this will shrink as neighbors commit).
- Set neighbors_latest = {} (last Committed and Tentative messages received).

STEP 1 — Make your first tentative choice:
If you are free and have options, pick one neighbor at random and propose to them.
How: warm_start
- If P is not empty, set tentative = random choice from P.

STEP 2 — Share your status:
In each round, tell your neighbors whether you are committed and who you are proposing to.
How: share_status
- Send to all neighbors: Committed = committed (or "none"); Tentative = tentative (or "none").

STEP 3 — Remove unavailable neighbors:
If a neighbor commits to someone else, drop them from your options. If you were pointing to them, cancel that proposal.
How: update_choices_from_neighbors
- Read new messages into neighbors_latest.
- Remove from P any neighbor whose Committed ≠ "none".
- If your current tentative is no longer in P, set tentative = "none".

STEP 4 — Make or accept proposals:
If you are free and have no tentative, choose a neighbor and propose. If a higher-priority neighbor proposes to you and you are free, accept them as your tentative.
How: propose_or_accept
- If committed = "none" and tentative = "none" and P not empty: pick one random neighbor from P and set tentative = that neighbor.
- Share your status and update your knowledge again.
- If you are still free: find offers = neighbors where Tentative = you and Committed = "none".
- If any offer comes from a higher-priority neighbor (earlier name), set tentative = that neighbor.

STEP 5 — Resolve conflicts and commit:
You commit only if you and your tentative neighbor chose each other and no earlier-name agent is also trying for you or your neighbor. Otherwise, you back off.
How: resolve_conflicts_by_priority
- If tentative = "none", stop.
- If your tentative neighbor did not also choose you, stop.
- Look at everyone proposing to you or your tentative neighbor.
- The agent with the earliest name wins.
- If the winner is you or your tentative neighbor, commit to each other.
- Otherwise, cancel your tentative.

STEP 6 — Continue until stable:
Repeat the proposing and conflict resolution until no progress can be made.
How: continue_until_stable
- Repeat Steps 2 → 5 each round.
- If you are committed, you are finished.
- If you are free and all your neighbors are committed, you are finished.
- Otherwise, continue until stability or until the maximum number of rounds ends.

STEP 7 — Final round behavior:
If this is the last round and you are still free, try once more by picking the best available neighbor, but respect higher-priority claims.
How: final_choice
- If committed = "none": let candidates = neighbors in P not committed.
- Remove candidates already targeted by a higher-priority neighbor.
- If candidates remain, set tentative = earliest-name candidate. Otherwise, set tentative = "none".
"""



class MatchingTentativeCommitMainPseudocodeOnly(Matching, MainPseudocodeOnly):
    def __init__(self, graph, rounds,problem_level, model_name="gpt-4o-mini", model_provider="openai", chain_of_thought=True):
        Matching.__init__(self, graph, rounds,problem_level,  model_name, model_provider, chain_of_thought, is_must_send=True)
        MainPseudocodeOnly.__init__(self)

    def get_symbolic_pseudo_code(self):
        return """
        STEP 0 — Initialize your state: ou begin with no partner, a list of neighbors you could match with, and an empty record of neighbor messages.
    
        STEP 1 — Make your first tentative choice: If you are free and have options, pick one neighbor at random and propose to them.
      
        STEP 2 — Share your status: In each round, tell your neighbors whether you are committed and who you are proposing to.
      
        STEP 3 — Remove unavailable neighbors: If a neighbor commits to someone else, drop them from your options. If you were pointing to them, cancel that proposal.
     
        STEP 4 — Make or accept proposals: If you are free and have no tentative, choose a neighbor and propose. If a higher-priority neighbor proposes to you and you are free, accept them as your tentative.
    
        STEP 5 — Resolve conflicts and commit: You commit only if you and your tentative neighbor chose each other and no earlier-name agent is also trying for you or your neighbor. Otherwise, you back off.
   
        STEP 6 — Continue until stable: Repeat the proposing and conflict resolution until no progress can be made.

        STEP 7 — Final round behavior: If this is the last round and you are still free, try once more by picking the best available neighbor, but respect higher-priority claims.

        """


class MatchingTentativeCommitFunctionPseudoCodeOnly(Matching, FunctionPseudoCodeOnly):
    def __init__(self, graph, rounds,problem_level, model_name="gpt-4o-mini", model_provider="openai", chain_of_thought=True):
        Matching.__init__(self, graph, rounds,problem_level,  model_name, model_provider, chain_of_thought, is_must_send=True)
        FunctionPseudoCodeOnly.__init__(self)

    def get_symbolic_functions(self):
        return """
1) init_variables
   - Set committed = "none".
   - Set tentative = "none".
   - Set P = list of your neighbors (options; this will shrink as neighbors commit).
   - Set neighbors_latest = {} (last Committed and Tentative messages received).

2) warm_start
   - If P is not empty, set tentative = random choice from P.

3) share_status
   - Send to all neighbors:
     • Committed = committed (or "none")
     • Tentative = tentative (or "none")

4) update_choices_from_neighbors
   - Read new messages into neighbors_latest.
   - Remove from P any neighbor whose Committed ≠ "none".
   - If your current tentative is no longer in P, set tentative = "none".

5) propose_or_accept
   - If committed = "none" and tentative = "none" and P not empty: pick one random neighbor from P and set tentative = that neighbor.
   - Share your status and update your knowledge again.
   - If you are still free:
       • Find offers = neighbors where Tentative = you and Committed = "none".
       • If any offer comes from a higher-priority neighbor (earlier name), set tentative = that neighbor.

6) resolve_conflicts_by_priority
   - If tentative = "none", stop.
   - If your tentative neighbor did not also choose you, stop.
   - Look at everyone proposing to you or your tentative neighbor.
   - The agent with the earliest name wins.
   - If the winner is you or your tentative neighbor, commit to each other.
   - Otherwise, cancel your tentative.

7) continue_until_stable
   - Repeat Steps 2 → 5 each round.
   - If you are committed, you are finished.
   - If you are free and all your neighbors are committed, you are finished.
   - Otherwise, continue until stability or until the maximum number of rounds ends.

8) final_choice
   - If committed = "none":
       • Let candidates = neighbors in P not committed.
       • Remove candidates already targeted by a higher-priority neighbor.
       • If candidates remain, set tentative = earliest-name candidate.
       • Otherwise, set tentative = "none".
"""


class MatchingTentativeCommitFunctionDescriptionOnly(Matching, FunctionDescriptionOnly):
    def __init__(self, graph, rounds,problem_level, model_name="gpt-4o-mini", model_provider="openai", chain_of_thought=True):
        Matching.__init__(self, graph, rounds,problem_level,  model_name, model_provider, chain_of_thought, is_must_send=True)
        FunctionDescriptionOnly.__init__(self)

    # Only override the MODE block; keep Coloring flow in _intro_problem_unique_rules()

    def get_symbolic_functions(self):
        return """
1) init_variables:
You set up your local state, starting free with no partner and a list of neighbors to consider.

2) warm_start:
You make an initial tentative choice by picking one neighbor at random.

3) share_status:
You tell your neighbors who you are paired with, if anyone, and who you are currently trying.

4) update_choices_from_neighbors:
You update your view based on neighbors’ messages, removing those who are already paired and clearing invalid choices.

5) propose_or_accept:
If you are free, you propose to a neighbor. If a higher-priority neighbor proposes to you (by neighbors alphabetical name), you may accept them.

6) resolve_conflicts_by_priority:
You only commit if you and your neighbor chose each other and no earlier agent conflicts(by neighbors alphabetical name). Otherwise, you step back.

7) continue_until_stable:
You keep repeating the round process until you are paired or cannot move anymore.

8) final_choice:
If it is the last round and you are still free, you make one best effort to choose a valid neighbor before stopping.
"""


class MatchingTentativeCommitHeuristicHintGuidedAgentV2(Matching, HeuristicHintGuided):
    def __init__(self, graph, rounds,problem_level, model_name="gpt-4o-mini", model_provider="openai", chain_of_thought=True):
        Matching.__init__(self, graph, rounds,problem_level,  model_name, model_provider, chain_of_thought, is_must_send=True)
        HeuristicHintGuided.__init__(self)

    def get_intro(self):
        return """
Each round, you should propose to one available neighbor and share your status with all others, accepting the proposal of a higher-priority neighbor if you are still free. You only commit when you and your tentative neighbor select each other and no earlier-name agent is also trying for you or your neighbor; if a conflict arises, you step back and try again. Continue this process until you are paired or have no free neighbors, and in the final round make your best possible choice while respecting higher-priority claims.
"""

#########################





class MatchingNaiveCode(Matching):
    def __init__(self, graph: nx.Graph, rounds: int,problem_level, **kwargs):
        super().__init__(graph=graph, rounds=rounds,problem_level = problem_level, **kwargs)
        self._order = list(sorted(self.graph.nodes()))
        self.model_name = "python"
        if run_until_converge:
            self.rounds = 1000

    async def bootstrap(self):  # no-op
        return

    async def pass_messages(self):
        solver = SolveNaiveMatching(self.graph, rounds=self.rounds)
        answers = solver.run()  # make sure to run the framework
        return answers  # list of partner names or 'None'


class MatchingNaiveCompletePseudoCode(Matching, CompletePseudoCode):
    def __init__(self, graph, rounds, problem_level, model_name="gpt-4o-mini", model_provider="openai",
                 chain_of_thought=True):
        Matching.__init__(self, graph, rounds, problem_level, model_name, model_provider, chain_of_thought,
                          is_must_send=True)
        CompletePseudoCode.__init__(self)

    def get_symbolic_pseudo_code(self):
        return """
STEP 0 — Initialize your state
You start free, with a list of neighbors and a blank view of what they’re doing. Keep track of who you’re tentatively trying and who (if anyone) you’re committed to. Earlier names mean higher priority.
How: init_variables
- Set up your memory so you know you are free, who your neighbors are, and that you haven’t seen any messages yet.

STEP 1 — First tentative try (downward only)
If you’re free, consider only neighbors you outrank (your name comes earlier). If any exist, pick one and start a try.
How: warm_start
- Choose one lower-priority neighbor to try and SEND a PROPOSE message to them.

STEP 2 — Share what you’re doing (every round)
Let all neighbors know whether you’re already committed and, if not, who you are currently trying; read theirs too.
How: share_status
- BROADCAST a STATUS message (Committed partner or “none”, Tentative target or “none”) and UPDATE your inbox view.

STEP 3 — Remove unavailable options
If a neighbor is already committed elsewhere, stop considering them. If you were trying them, drop that try and let them know.
How: update_choices_from_neighbors
- Refresh your view from incoming STATUS; DROP committed neighbors from your options.
- If you were trying someone who became unavailable, SEND a WITHDRAW to them and clear your tentative.

STEP 4 — Propose downward, yield upward immediately
If you’re free and not trying anyone, propose to one of your lower-priority neighbors. If a higher-priority neighbor proposes to you and you’re free, accept them right away and treat that as your current commitment (until they release you).
How: propose_or_accept_nice
- If idle and free, SEND PROPOSE to one lower-priority neighbor.
- If a superior has PROPOSED to you and you are free, SEND ACCEPT to that superior and mark yourself committed to them; if you were trying someone, SEND WITHDRAW to that neighbor.

STEP 5 — Settle gently (soft commitments)
If the lower-priority neighbor you tried also tries you back—and no superior has asked for you in the meantime—commit to each other. If a superior asks for you at any point, switch to them and release others. If your superior drops you later, you become free again.
How: resolve_conflicts_soft_priority
- If mutual with your tentative (they PROPOSE you), SEND ACCEPT to them and BROADCAST MATCHED to all neighbors.
- If a superior PROPOSES you at any time, SEND ACCEPT to the superior, SEND WITHDRAW/REJECT to anyone else, and BROADCAST MATCHED.
- If you were committed to a superior but they stop trying you (or announce otherwise), mark yourself free and BROADCAST STATUS to reflect that.

STEP 6 — Continue until calm (or time’s up)
Keep sharing, pruning, proposing downward, and yielding upward. Stop when nothing changes or when the round limit is reached.
How: continue_until_stable
- Repeat: share_status → update_choices_from_neighbors → propose_or_accept_nice → resolve_conflicts_soft_priority.
- Detect no-change, or stop at the round budget.

STEP 7 — Last nudge
If you’re still free, first accept any current superior offer; if none exists, make one last downward proposal and then stop.
How: final_choice
- If a superior is proposing now, SEND ACCEPT and BROADCAST MATCHED.
- Otherwise, if any lower-priority option remains, SEND a final PROPOSE; after one more resolve step, BROADCAST your final STATUS and stop.

"""


class MatchingNaiveMainPseudocodeOnly(Matching, MainPseudocodeOnly):
    def __init__(self, graph, rounds, problem_level, model_name="gpt-4o-mini", model_provider="openai",
                 chain_of_thought=True):
        Matching.__init__(self, graph, rounds, problem_level, model_name, model_provider, chain_of_thought,
                          is_must_send=True)
        MainPseudocodeOnly.__init__(self)

    def get_symbolic_pseudo_code(self):
        return """
STEP 0 — Initialize your state
You start free, with a list of neighbors and a blank view of what they’re doing. Keep track of who you’re tentatively trying and who (if anyone) you’re committed to. Earlier names mean higher priority.

STEP 1 — First tentative try (downward only)
If you’re free, consider only neighbors you outrank (your name comes earlier). If any exist, pick one and start a try.

STEP 2 — Share what you’re doing (every round)
Let all neighbors know whether you’re already committed and, if not, who you are currently trying; read theirs too.

STEP 3 — Remove unavailable options
If a neighbor is already committed elsewhere, stop considering them. If you were trying them, drop that try and let them know.

STEP 4 — Propose downward, yield upward immediately
If you’re free and not trying anyone, propose to one of your lower-priority neighbors. If a higher-priority neighbor proposes to you and you’re free, accept them right away and treat that as your current commitment (until they release you).

STEP 5 — Settle gently (soft commitments)
If the lower-priority neighbor you tried also tries you back—and no superior has asked for you in the meantime—commit to each other. If a superior asks for you at any point, switch to them and release others. If your superior drops you later, you become free again.

STEP 6 — Continue until calm (or time’s up)
Keep sharing, pruning, proposing downward, and yielding upward. Stop when nothing changes or when the round limit is reached.

STEP 7 — Last nudge
If you’re still free, first accept any current superior offer; if none exists, make one last downward proposal and then stop.


        """


class MatchingNaiveFunctionPseudoCodeOnly(Matching, FunctionPseudoCodeOnly):
    def __init__(self, graph, rounds, problem_level, model_name="gpt-4o-mini", model_provider="openai",
                 chain_of_thought=True):
        Matching.__init__(self, graph, rounds, problem_level, model_name, model_provider, chain_of_thought,
                          is_must_send=True)
        FunctionPseudoCodeOnly.__init__(self)

    def get_symbolic_functions(self):
        return """
1) init_variables
- Set up your memory so you know you are free, who your neighbors are, and that you haven’t seen any messages yet.

2) warm_start
- Choose one lower-priority neighbor to try and SEND a PROPOSE message to them.

3) share_status
- BROADCAST a STATUS message (Committed partner or “none”, Tentative target or “none”) and UPDATE your inbox view.

4) update_choices_from_neighbors
- Refresh your view from incoming STATUS; DROP committed neighbors from your options.
- If you were trying someone who became unavailable, SEND a WITHDRAW to them and clear your tentative.

5) propose_or_accept_nice
- If idle and free, SEND PROPOSE to one lower-priority neighbor.
- If a superior has PROPOSED to you and you are free, SEND ACCEPT to that superior and mark yourself committed to them; if you were trying someone, SEND WITHDRAW to that neighbor.

6) resolve_conflicts_soft_priority
- If mutual with your tentative (they PROPOSE you), SEND ACCEPT to them and BROADCAST MATCHED to all neighbors.
- If a superior PROPOSES you at any time, SEND ACCEPT to the superior, SEND WITHDRAW/REJECT to anyone else, and BROADCAST MATCHED.
- If you were committed to a superior but they stop trying you (or announce otherwise), mark yourself free and BROADCAST STATUS to reflect that.

7) continue_until_stable
- Repeat: share_status → update_choices_from_neighbors → propose_or_accept_nice → resolve_conflicts_soft_priority.
- Detect no-change, or stop at the round budget.

8) final_choice
- If a superior is proposing now, SEND ACCEPT and BROADCAST MATCHED.
- Otherwise, if any lower-priority option remains, SEND a final PROPOSE; after one more resolve step, BROADCAST your final STATUS and stop.

"""


class MatchingNaiveFunctionDescriptionOnly(Matching, FunctionDescriptionOnly):
    def __init__(self, graph, rounds, problem_level, model_name="gpt-4o-mini", model_provider="openai",
                 chain_of_thought=True):
        Matching.__init__(self, graph, rounds, problem_level, model_name, model_provider, chain_of_thought,
                          is_must_send=True)
        FunctionDescriptionOnly.__init__(self)

    # Only override the MODE block; keep Coloring flow in _intro_problem_unique_rules()

    def get_symbolic_functions(self):
        return """
1) init_variables
   - Initialize your local state: start free, record your neighbors, and clear any prior observations.

2) warm_start
   - Make an initial tentative choice among neighbors you outrank, if any are available.

3) share_status
   - Let neighbors know your current situation and take note of theirs to stay in sync.

4) update_choices_from_neighbors
   - Refresh your view based on neighbors’ latest states and drop options that are no longer available.

5) propose_or_accept_nice
   - If idle, reach out to a lower-priority neighbor; if a higher-priority neighbor reaches out to you, prefer them.

6) resolve_conflicts_soft_priority
   - Settle your tentative choice: commit when it’s mutually agreeable and no higher-priority claim preempts it; otherwise, yield gracefully.

7) continue_until_stable
   - Repeat the share–update–choose–settle cycle until nothing changes or you hit the round limit.

8) final_choice
   - In the last round, prefer any current higher-priority interest; if none, make one careful final attempt and conclude.

"""


class MatchingNaiveHeuristicHintGuidedAgentV2(Matching, HeuristicHintGuided):
    def __init__(self, graph, rounds, problem_level, model_name="gpt-4o-mini", model_provider="openai",
                 chain_of_thought=True):
        Matching.__init__(self, graph, rounds, problem_level, model_name, model_provider, chain_of_thought,
                          is_must_send=True)
        HeuristicHintGuided.__init__(self)

    def get_intro(self):
        return """
As an agent, you should always try to pair with neighbors you outrank, but be ready to yield if a higher-priority neighbor shows interest in you. Once you and a neighbor agree on each other and no superior neighbor interferes, you both commit and step out of the process. Keep repeating this cycle of proposing, accepting, and updating until no more progress is possible or the rounds run out."""


################# distributed version of Edmonds’ Blossom Algorithm


class MatchingEdmondsCode(Matching):
    def __init__(self, graph: nx.Graph, rounds: int, **kwargs):
        super().__init__(graph=graph, rounds=rounds, **kwargs)
        self._order = list(sorted(self.graph.nodes()))
        self.model_name = "python"

    async def bootstrap(self):
        return

    async def pass_messages(self):
        # Choose how many augmentations you allow per round (default 1)
        solver = SolveMatchingEdmonds(self.graph, rounds=self.rounds, max_augment_per_round=1)
        return solver.answers


class MatchingEdmondsCompletePseudoCode(Matching, CompletePseudoCode):
    def __init__(self, graph, rounds, model_name="gpt-4o-mini", model_provider="openai", chain_of_thought=True):
        Matching.__init__(self, graph, rounds, model_name, model_provider, chain_of_thought, is_must_send=True)
        CompletePseudoCode.__init__(self)

    def get_symbolic_pseudo_code(self):
        return r"""
1) Set up your local state so you can participate in a distributed, optimal matching search.
   How: Use get_self_name and init_state.
   - Learn your own name.
   - Start with partner = 'None' and label = 'FREE'.
   - Set parent = None and base = your own name.
   - Set phase = 0.

2) Begin a new global search phase from every free agent at the same time.
   How: Use begin_phase and send_message.
   - Increase phase by 1.
   - If you have no partner:
       • Mark yourself as EVEN.
       • Set parent = None and base = your own name.
       • Notify all neighbors that you are exploring by sending an EXPLORE message with {phase, from, base, parity='EVEN'}.

3) Grow the alternating search by reacting to EXPLORE messages and detect when an augmenting path or an odd cycle appears.
   How: Use receive_messages and handle_explore_edge.
   - Continuously read messages for the current phase.
   - When you receive EXPLORE(phase, from=u, base=b_u, parity='EVEN') and the edge (u, you) is currently UNMATCHED and you are unlabeled (FREE):
       • Mark yourself ODD, set parent = u, and base = b_u.
       • If you already have a partner w:
           – Make w EVEN with parent = you and base = b_u, and forward EXPLORE to w.
         Otherwise (you are free):
           – Report an AUGMENT event because a free node has been reached through alternating edges.
   - When you receive EXPLORE(phase, from=u, base=b_u, parity='ODD') and the edge (u, you) is currently MATCHED and you are unlabeled (FREE):
       • Mark yourself EVEN, set parent = u, and base = b_u.
       • Continue exploration by informing neighbors.
   - If you are already labeled and an EXPLORE comes from a node u that is ALSO labeled EVEN:
       • If you and u belong to different search roots, report an AUGMENT event (two EVEN frontiers connect).
       • If you and u share the same root but have different bases, report a BLOSSOM event (an odd cycle has been detected).

4) Resolve an odd cycle so it does not block the discovery of augmenting paths.
   How: Use find_lca_base, contract_blossom, and relabel_after_contraction.
   - Identify the lowest common ancestor base by walking parent pointers from each endpoint until they meet.
   - Collect all vertices on the paths from each endpoint to that base; these vertices form the blossom along with the closing edge.
   - Contract the entire blossom into a single unit represented by the base.
   - Relabel members so they behave as that one unit.
   - Resume exploration from the base as an EVEN frontier and continue sending EXPLORE messages.

5) Increase the size of the matching whenever an augmenting path is found.
   How: Use build_augmenting_path, flip_along_path, set_matched_partner, and announce_matched.
   - Construct the full path by walking parent pointers from each endpoint back to their free roots and adding the connecting edge.
   - Flip the status of every edge on this path: unmatched edges become matched, and matched edges become unmatched.
   - If you are on this path:
       • If you now have a partner, record it.
       • If you do not, keep partner = 'None'.
   - Inform neighbors (and your partner, if any) that you are matched so they treat you as unavailable going forward.

6) Decide whether another search phase is needed or the matching is already maximum.
   How: Use detect_quiescence.
   - If at least one AUGMENT event occurred anywhere in the current phase, start a new phase and repeat the exploration steps.
   - If no AUGMENT events occurred during the entire phase, there are no remaining augmenting paths; the current matching is maximum.


"""


class MatchingEdmondsMainPseudocodeOnly(Matching, MainPseudocodeOnly):
    def __init__(self, graph, rounds, model_name="gpt-4o-mini", model_provider="openai", chain_of_thought=True):
        Matching.__init__(self, graph, rounds, model_name, model_provider, chain_of_thought, is_must_send=True)
        MainPseudocodeOnly.__init__(self)

    def get_symbolic_pseudo_code(self):
        return r"""
    1) Set up your local state so you can participate in a distributed, optimal matching search.
       How: Use get_self_name and init_state.


    2) Begin a new global search phase from every free agent at the same time.
       How: Use begin_phase and send_message.

    3) Grow the alternating search by reacting to EXPLORE messages and detect when an augmenting path or an odd cycle appears.
       How: Use receive_messages and handle_explore_edge.

    4) Resolve an odd cycle so it does not block the discovery of augmenting paths.
       How: Use find_lca_base, contract_blossom, and relabel_after_contraction.


    5) Increase the size of the matching whenever an augmenting path is found.
       How: Use build_augmenting_path, flip_along_path, set_matched_partner, and announce_matched.

    6) Decide whether another search phase is needed or the matching is already maximum.
       How: Use detect_quiescence.


    """


class MatchingEdmondsFunctionPseudoCodeOnly(Matching, FunctionPseudoCodeOnly):
    def __init__(self, graph, rounds, model_name="gpt-4o-mini", model_provider="openai", chain_of_thought=True):
        Matching.__init__(self, graph, rounds, model_name, model_provider, chain_of_thought, is_must_send=True)
        FunctionPseudoCodeOnly.__init__(self)

    def get_symbolic_functions(self):
        return (r"""
get_self_name
- Return your own identifier as a string. This is the unique name you use in all communications.

init_state
- Initialize your local variables at the beginning.
- Set partner = 'None', label = 'FREE', parent = None, base = your own name, phase = 0.

receive_messages
- Collect all incoming messages that neighbors have sent since your last call.
- Each message may include exploration requests, accept/reject replies, matched announcements, or event signals.

send_message
- Send a message to a specific neighbor or broadcast to all neighbors.
- Used to share EXPLORE messages, ACCEPT/REJECT, MATCHED, BLOSSOM, and AUGMENT events.

set_matched_partner
- Update your state to record the name of the agent you are currently matched with.
- Mark yourself as matched.

output_final_partner
- At the end of the algorithm, return your partner’s name if matched.
- If you remain unmatched, return 'None'.

mark_neighbor_unavailable
- Record that a neighbor is no longer available for proposals because they are already matched.

choose_one_proposer_greedy
- When multiple neighbors propose to you, select exactly one deterministically (e.g., the highest or lowest by name).
- Ensure that the choice is consistent across all runs.

list_unmatched_neighbors
- Return the list of all your neighbors who are still unmatched and available for pairing.

pick_candidate_neighbor
- From a list of candidate neighbors, choose exactly one deterministically (e.g., lexicographic maximum).
- Used when you want to propose to a neighbor.

find_lca_base
- Given two nodes inside the same search tree, follow parent pointers upward until you find their lowest common ancestor.
- This ancestor becomes the base of a contracted blossom.

contract_blossom
- Take all nodes that form an odd cycle (a blossom) and treat them as a single super-node.
- Update their base to the lowest common ancestor.

relabel_after_contraction
- After contracting a blossom, update the labels of its members so they behave consistently as one unit in further searches.

build_augmenting_path
- Construct the sequence of vertices and edges that forms an augmenting path between two free roots.

flip_along_path
- Reverse the status of all edges along an augmenting path: matched edges become unmatched, and unmatched edges become matched.

announce_matched
- Notify your neighbors and your new partner that you are now matched so they mark you as unavailable.

detect_quiescence
- Check whether any AUGMENT events happened in the current phase.
- If none occurred, declare that the maximum matching has been reached.""")


class MatchingEdmondsFunctionDescriptionOnly(Matching, FunctionDescriptionOnly):
    def __init__(self, graph, rounds, model_name="gpt-4o-mini", model_provider="openai", chain_of_thought=True):
        Matching.__init__(self, graph, rounds, model_name, model_provider, chain_of_thought, is_must_send=True)
        FunctionDescriptionOnly.__init__(self)

    # Only override the MODE block; keep Coloring flow in _intro_problem_unique_rules()

    def get_symbolic_functions(self):
        return r"""
get_self_name
- Return your own unique name as a string.

init_state
- Initialize state: partner='None', label='FREE', parent=None, base=self, phase=0.

receive_messages
- Collect all messages sent by neighbors since the last round.

send_message
- Send a message to a neighbor or broadcast to all neighbors.

set_matched_partner
- Record the name of the agent you are matched with.

output_final_partner
- Return your partner’s name if matched, otherwise 'None'.

mark_neighbor_unavailable
- Mark a neighbor as unavailable once it is matched.

choose_one_proposer_greedy
- Pick one proposer deterministically (e.g., by highest or lowest name).

list_unmatched_neighbors
- List neighbors who are currently free and available.

pick_candidate_neighbor
- Select one candidate neighbor deterministically from a list.

find_lca_base
- Find the lowest common ancestor base between two nodes.

contract_blossom
- Collapse all nodes in an odd cycle into a single unit with one base.

relabel_after_contraction
- Update labels of blossom members so they behave as one node.

build_augmenting_path
- Trace and return the sequence of nodes forming an augmenting path.

flip_along_path
- Switch matched edges to unmatched and unmatched edges to matched along a path.

announce_matched
- Notify neighbors and your partner that you are matched.

detect_quiescence
- Check if no augmenting paths were found this phase (termination condition).
"""


class MatchingEdmondsHintGuidedAgent(Matching, HeuristicHintGuided):
    def __init__(self, graph, rounds, model_name="gpt-4o-mini", model_provider="openai", chain_of_thought=True):
        Matching.__init__(self, graph, rounds, model_name, model_provider, chain_of_thought, is_must_send=True)
        HeuristicHintGuided.__init__(self)

    def get_intro(self):
        return r"""
        Begin free and try to grow connections in alternating steps of free and matched edges.  
        If your path meets another or reaches a free neighbor, take the opportunity to form a match.  
        When cycles appear, simplify them as a single unit so progress can continue.  
        Stay consistent, avoid unnecessary messages, and clearly confirm when you are matched.
        """

################################



class MatchingBackTrackCode(Matching):
    def __init__(self, graph: nx.Graph, rounds: int, **kwargs):
        super().__init__(graph=graph, rounds=rounds, **kwargs)
        self._order = list(sorted(self.graph.nodes()))
        self.model_name = "python"
        if run_until_converge:
            self.rounds =1000

    async def bootstrap(self):
        return

    async def pass_messages(self):
        # Oracle: compute exact maximum matching once
        solver = SolveMatchingBacktrack(self.graph, rounds=self.rounds)
        answers = solver.run()  # List[Optional[str]] aligned with solver.order
        # Ensure alignment with this agent's ordering if you rely on it elsewhere
        if solver.order != self._order:
            # remap to this class's order
            index_in_solver = {u: i for i, u in enumerate(solver.order)}
            remapped: List[Optional[str]] = []
            for u in self._order:
                remapped.append(answers[index_in_solver[u]])
            return remapped
        return answers




class MatchingBackTrackCompletePseudoCodeV2(Matching, CompletePseudoCode):
    def __init__(
        self,
        graph: nx.Graph,
        rounds: int,
        problem_level,
        model_name: str = "gpt-4o-mini",
        model_provider: str = "openai",
        chain_of_thought: bool = True,
        is_must_send: bool = True,
    ):
        Matching.__init__(
            self,
            graph=graph,
            rounds=rounds,
            problem_level=problem_level,
            model_name=model_name,
            model_provider=model_provider,
            chain_of_thought=chain_of_thought,
            is_must_send=is_must_send,
        )
        CompletePseudoCode.__init__(self)
        self.problem_level = problem_level

    def get_symbolic_pseudo_code(self):
        main_function = """
STEP 0 — Create a clear notion of “who you must listen to” (Higher) and “who follows your lead” (Lower) alphabetically.
Choose exactly one neighbor above you as the escalation target so that dead-ends have a single, predictable escape path.
How: establish_order_and_parent
• Put all agent names in alphabetical order; that order is the one everyone uses.
• “Higher” are neighbors whose names come before yours; “Lower” are those after you.
• Your parent is the closest Higher neighbor (the one immediately before you in the alphabet). If you have no Higher, you have no parent.
• Important: you only ever start proposals to Lower neighbors (never to Higher).

STEP 1 — Initialize your local search state so you can explore candidates and remember what failed.
Set up a compact “ledger” of what you know (neighbors’ states), what you are trying (your tentative choice), and what already failed 
(so you don’t repeat it). Keep a monotonically increasing phase_id for deduping and ignoring stale messages.
How: init_state
• Keep exactly one of these states at any time: free; tentative(partner); committed(partner).
• Keep pending_proposal (who you most recently proposed to) or None.
• Keep a simple list of neighbors in your preferred order (alphabetical is fine).
• Keep a small record for each neighbor: what they last announced (free/tentative/committed), with whom, and the version number they attached.
• Keep a version number (phase_id). Increase it only when your own state or pending choice changes. When you increase it, clear your “already tried” notes for this version.

STEP 2 — Choose your first candidate consistently with what Higher neighbors already decided.
Act only after you understand Higher’s constraints; then attempt the best neighbor that’s currently attainable. 
Your first move should neither collide with a known commitment nor repeat a previously failed attempt under the same Higher context.
How: choose_first_candidate
• If you have no Higher, you can act now. Otherwise, wait until you have heard a STATUS from each Higher (or a reasonable timeout).
• A neighbor is “feasible” if: (a) you are allowed to propose to them (they are Lower); (b) they are not known to be committed to someone else; 
  and (c) if they are tentatively paired with someone else, your name comes earlier than that someone else’s name.
• Also, don’t retry the same person again within the same version number.
• Scan your list in order; pick the first feasible neighbor. Set them as your pending proposal, send PROPOSE to them, bump your version, and send a STATUS that you are tentative with them.
• If no one looks feasible, go to backtrack (STEP 6).

STEP 3 — Decide what to do when someone proposes to you.
Treat concurrent proposals like bids: accept exactly one, deterministically, and decline the rest. 
Prefer the highest-priority proposer to keep the global search consistent and to avoid oscillations.
How: on_propose_from_neighbor(nei)
• Collect all proposals you received this round and pick the one from the alphabetically earliest proposer.
• If you are free, become tentative with that proposer and reject the others. Bump your version and send a STATUS.
• If you are already tentative with someone and this new proposer comes earlier in the alphabet than your current tentative partner, switch to the new proposer and reject everyone else. Bump your version and send a STATUS.
• If you are committed, reject all proposals; commitments are not broken just because someone asks.
• Limit yourself to at most one such upgrade per round so you don’t flip-flop.

STEP 4 — React to responses to your own proposal so you either lock it in or keep searching.
When accepted, convert tentative to committed and broadcast stability; when rejected, record that failure and move on. You either strengthen a match or prune a branch of the search space.
How: on_accept_from(candidate) / on_reject_from(candidate)
• If accepted: mark yourself committed with that partner, clear your pending proposal, send COMMIT to them, bump your version, and send a STATUS that you are committed.
• If rejected: note that this person was tried under your current version and move on to the next option (STEP 5).

STEP 5 — Try the next option; if none exist, escalate upward.
Keep iterating candidates that are currently feasible; if you exhaust them, escalate so that pressure propagates toward Higher priority 
agents who can free capacity or change their choices.
How: propose_next_or_backtrack
• Look again through your Lower neighbors in order using the same “feasible” idea as before.
• If you find one: set as pending, send PROPOSE, bump your version, and send a STATUS.
• If you don’t find any: go to backtrack (STEP 6).

STEP 6 — Ask the closest Higher neighbor to reconsider when you are blocked.
When you cannot proceed, request relief from the designated Higher neighbor. 
Pause accepting new tentatives while waiting, to avoid thrashing against the same constraint.
How: backtrack_if_needed
• If you have no parent: tell your Lower neighbors you could not find a partner (optional FAIL), send one final STATUS saying you are free, and stop proposing.
• If you have a parent: send a BACKTRACK request to that parent, send a STATUS saying you are requesting help, and pause starting new proposals until something changes upstream (you’ll see it in a new STATUS from a Higher).

STEP 7 — When a Lower neighbor backtracks to you, try to free capacity or pass the backtrack upward.
First attempt a local reconfiguration (switch tentative or release a commit only if it’s safe); 
if you cannot help without violating priority discipline, push the backtrack upward unchanged.
How: on_backtrack_from_child(child)
• If you are tentative with someone, try moving your tentative to a different feasible Lower neighbor to relieve the pressure. If you change, send a STATUS.
• If you are committed: only release your current partner if you can immediately propose to a safe alternative in the same step (undo then propose right away). Otherwise, forward the backtrack to your parent. If you have no parent, report failure to the child.

STEP 8 — Keep everyone synced with minimal chatter so feasibility checks are cheap.
Share only meaningful changes, tagged with a phase_id, so neighbors can quickly update their view and ignore stale data. 
This keeps the network quiet while maintaining enough information for correct feasibility tests.
How: share_status
• Send STATUS only when something actually changes (free ↔ tentative ↔ committed, or requesting_help).
• Include who you’re paired with (if anyone) and your current version number. Neighbors ignore messages that carry an older version than they already know.
• Don’t repeat the same STATUS over and over.

STEP 9 — Finish cleanly (and force a decision) when nothing moves or when you reach the final round.
End with no dangling offers: upgrade safe pairs to committed, pick exactly one winner in ties, and otherwise stay free. 
In the final round, perform a one-shot **forced-commit** so at least some commitments are made.
How: finalize_now
• First, close any unanswered pending proposal by noting it as tried and clearing it.
• Next, if you and someone else both currently point to each other tentatively, upgrade that pair to a commitment and announce it.
• If you are still free, you may make one clean, deterministic commitment to a Lower neighbor who is free or only tentative with you, provided there is no one earlier than you also trying for that same neighbor.
• If more than one commitment hits you at once, accept the earliest sender and politely reject the others.
• Send one final STATUS with your finished state and stop.
"""
        return main_function

class MatchingBackTrackFunctionPseudoCodeOnlyV2(Matching, FunctionPseudoCodeOnly):
    def __init__(
        self,
        graph: nx.Graph,
        rounds: int,
        problem_level,
        model_name: str = "gpt-4o-mini",
        model_provider: str = "openai",
        chain_of_thought: bool = True,
        is_must_send: bool = True,
    ):
        Matching.__init__(
            self,
            graph=graph,
            rounds=rounds,
            problem_level=problem_level,
            model_name=model_name,
            model_provider=model_provider,
            chain_of_thought=chain_of_thought,
            is_must_send=is_must_send,
        )
        FunctionPseudoCodeOnly.__init__(self)
        self.problem_level = problem_level

    def get_symbolic_functions(self):
        main_function = """
1) establish_order_and_parent
• Use alphabetical order as the shared priority. Neighbors before you are “Higher”; after you are “Lower”.
• Your parent is the closest Higher neighbor (the one immediately before you). If none exist, you have no parent.
• You only initiate proposals toward Lower neighbors.

2) init_state
• Hold exactly one state at a time: free; tentative(partner); committed(partner).
• Reset any pending proposal. Keep neighbors in a simple ordered list.
• Track what each neighbor last announced and a local version number that you increase only when your own state or pending choice changes.
• Keep a small “already tried in this version” note so you don’t repeat the same attempt immediately.

3) choose_first_candidate
• If you have Higher neighbors, wait until you’ve heard their latest STATUS (or a sensible timeout).
• Pick the first Lower neighbor who looks compatible: not committed to someone else; and if they are tentative with someone, your name comes earlier than that someone’s.
• Propose to exactly one such neighbor, mark them as your current attempt, bump your version, and send a STATUS about your tentative choice.
• If no one fits, prepare to backtrack.

4) on_propose_from_neighbor(nei)
• If multiple proposals arrive, pick the one from the alphabetically earliest sender.
• If you’re free, tentatively accept that one and reject the rest, then announce.
• If you’re already tentative and the new proposer is earlier than your current tentative partner, switch once this round to the better one and reject others, then announce.
• If you’re committed, reject all proposals; commitments aren’t broken by proposals.

5) on_accept_from(candidate)
• Turn your tentative into a commitment, clear your pending proposal, and announce. You’re now taken.

6) on_reject_from(candidate)
• Record that this try didn’t work with your current view. Move on to the next option or escalate if none remain.

7) propose_next_or_backtrack
• Try the next compatible Lower neighbor in order, one at a time, announcing when you change.
• If you can’t find any, send a BACKTRACK to your parent (if you have one) and pause new attempts until something upstream changes. If you have no parent, stop proposing further and announce that you remain free.

8) backtrack_if_needed
• Ask your parent (closest Higher) to reconsider; this frees room for you and others downstream.
• While waiting for a change from Higher, don’t create new tentative offers.

9) on_backtrack_from_child(child)
• If you’re tentative, try shifting your tentative to another compatible Lower neighbor to relieve pressure.
• If you’re committed, only release if you can immediately switch to a safe alternative in the same step; otherwise pass the backtrack to your parent (or report failure if you have none).

10) share_status
• Announce only when your state meaningfully changes (free/tentative/committed/requesting_help) and include who you’re paired with if anyone.
• Neighbors ignore older versions; don’t repeat the same STATUS unnecessarily.

11) finalize_now
• Close any dangling attempts, then upgrade mutual tentatives to commitments.
• If still free, you may make one clean, deterministic commitment to a Lower neighbor who has no better known contender than you.
• Break simultaneous ties by the same alphabetical rule, announce once, and stop.
"""
        return main_function
class MatchingBackTrackMainPseudocodeOnly(Matching, MainPseudocodeOnly):
    def __init__(self, graph, rounds, problem_level, model_name="gpt-4o-mini", model_provider="openai",
                 chain_of_thought=True):
        Matching.__init__(self, graph, rounds, problem_level, model_name, model_provider, chain_of_thought,
                          is_must_send=True)
        MainPseudocodeOnly.__init__(self)


    def get_symbolic_pseudo_code(self):
        main_function = """
STEP 0 — Create a clear notion of “who you must listen to” (Higher) and “who follows your lead” (Lower) alphabetically.
Choose exactly one neighbor above you as the escalation target so that dead-ends have a single, predictable escape path.


STEP 1 — Initialize your local search state so you can explore candidates and remember what failed.
Set up a compact “ledger” of what you know (neighbors’ states), what you are trying (your tentative choice), and what already failed 
(so you don’t repeat it). Keep a monotonically increasing phase_id for deduping and ignoring stale messages.


STEP 2 — Choose your first candidate consistently with what Higher neighbors already decided.
Act only after you understand Higher’s constraints; then attempt the best neighbor that’s currently attainable. 
Your first move should neither collide with a known commitment nor repeat a previously failed attempt under the same Higher context.


STEP 3 — Decide what to do when someone proposes to you.
Treat concurrent proposals like bids: accept exactly one, deterministically, and decline the rest. 
Prefer the highest-priority proposer to keep the global search consistent and to avoid oscillations.


STEP 4 — React to responses to your own proposal so you either lock it in or keep searching.
When accepted, convert tentative to committed and broadcast stability; when rejected, record that failure and move on. You either strengthen a match or prune a branch of the search space.

STEP 5 — Try the next option; if none exist, escalate upward.
Keep iterating candidates that are currently feasible; if you exhaust them, escalate so that pressure propagates toward Higher priority 
agents who can free capacity or change their choices.

STEP 6 — Ask the closest Higher neighbor to reconsider when you are blocked.
When you cannot proceed, request relief from the designated Higher neighbor. 
Pause accepting new tentatives while waiting, to avoid thrashing against the same constraint.


STEP 7 — When a Lower neighbor backtracks to you, try to free capacity or pass the backtrack upward.
First attempt a local reconfiguration (switch tentative or release a commit only if it’s safe); 
if you cannot help without violating priority discipline, push the backtrack upward unchanged.


STEP 8 — Keep everyone synced with minimal chatter so feasibility checks are cheap.
Share only meaningful changes, tagged with a phase_id, so neighbors can quickly update their view and ignore stale data. 
This keeps the network quiet while maintaining enough information for correct feasibility tests.


STEP 9 — Finish cleanly (and force a decision) when nothing moves or when you reach the final round.
End with no dangling offers: upgrade safe pairs to committed, pick exactly one winner in ties, and otherwise stay free. 
In the final round, perform a one-shot **forced-commit** so at least some commitments are made.




"""
        return main_function





class MatchingBackTracFunctionDescriptionOnly(Matching, FunctionDescriptionOnly):
    def __init__(
            self,
            graph: nx.Graph,
            rounds: int,
            problem_level,
            model_name: str = "gpt-4o-mini",
            model_provider: str = "openai",
            chain_of_thought: bool = True,
            is_must_send: bool = True,
    ):
        Matching.__init__(
            self,
            graph=graph,
            rounds=rounds,
            problem_level=problem_level,
            model_name=model_name,
            model_provider=model_provider,
            chain_of_thought=chain_of_thought,
            is_must_send=is_must_send,
        )
        FunctionDescriptionOnly.__init__(self)
        self.problem_level = problem_level

    def get_symbolic_functions(self):
        main_function = """
1) establish_priority_and_parent
• Agree on a simple global ordering so everyone knows who they defer to and who defers to them.
• Identify exactly one neighbor “above” you to contact if you get stuck (your escalation path).
• This creates a one-direction flow for resolving conflicts and avoids circular backtracking.

2) init_state
• Start fresh: clear any old choices, remember that you’re free, and list neighbors you might match with.
• Create a lightweight notebook of what you hear from neighbors and what you’ve already tried.
• Keep a small counter/version so others can ignore your outdated messages.

3) choose_first_candidate
• Before acting, make sure you’ve heard from higher-priority neighbors so you don’t clash with them.
• Pick a reasonable partner from your list who currently looks available/compatible.
• Announce your tentative choice and reach out to that neighbor to see if they agree.

4) on_propose_from_neighbor(nei)
• When someone asks to pair with you, decide in a consistent and fair way using the shared tie-break rule.
• Accept at most one proposal; decline the others so everyone has a clear signal.
• Let neighbors know your temporary decision so they can adjust.

5) on_accept_from(candidate)
• When your proposal is accepted, turn the tentative choice into a committed one.
• Share that commitment with neighbors so they can mark you as unavailable.
• Clear any pending offers you no longer need.

6) on_reject_from(candidate)
• If your proposal is declined, note that this path didn’t work under the current situation.
• Try a different plausible option next; don’t repeat a failed attempt under the same conditions.

7) propose_next_or_backtrack
• Move down your list and attempt another compatible partner.
• If no options remain right now, escalate politely to your designated higher neighbor for help.
• Keep messages minimal but clear so others can quickly recompute their options.

8) backtrack_if_needed
• Ask your escalation contact to reconsider their choice so the system can free space for you.
• While waiting, avoid creating new conflicts; pause making fresh tentative offers.
• Resume searching as soon as upstream changes arrive.

9) on_backtrack_from_child(child)
• If someone who depends on you is stuck, first see if you can shuffle your own tentative/commitment to help.
• Only release a commitment when it’s safe and consistent with the shared priority rule.
• If you can’t help locally, forward the backtrack upward along the established path.

10) share_status
• Whenever your state meaningfully changes (free/tentative/committed/requesting_help), send a concise update.
• Include just enough info for neighbors to refresh their view and ignore stale messages.
• Avoid spamming duplicates—send on change, not on a timer.

11) finalize_now
• When it’s time to stop (quiescence or final round), close any dangling offers and settle on a stable outcome.
• Prefer mutual, uncontested pairs for final commitments; otherwise end free rather than ambiguous.
• Announce your final state once to all neighbors and stop.
"""
        return main_function


class MatchingBackTrackHeuristicHintGuidedAgentV2(Matching, HeuristicHintGuided):
    def __init__(self, graph, rounds,problem_level, model_name="gpt-4o-mini", model_provider="openai", chain_of_thought=True):
        Matching.__init__(self, graph, rounds,problem_level,  model_name, model_provider, chain_of_thought, is_must_send=True)
        HeuristicHintGuided.__init__(self)

    def get_intro(self):
        return (
            "Follow the shared priority: hear from Higher neighbors first, refresh your view, then tentatively propose to the alphabetically earliest feasible neighbor; if proposals arrive to you, accept at most one—the alphabetically earliest proposer—and reject the rest. "
            "If your choice conflicts or is rejected, record that attempt under the current of agents higher context, try the next alphabetically feasible neighbor, or escalate once to your designated Higher with a BACKTRACK; pause new offers while waiting for upstream changes. "
            "Broadcast only on real state changes (free ↔ tentative ↔ committed)"
        )


class MatchingBackTrackFunctionPseudoCodeOnly(Matching, FunctionPseudoCodeOnly):
    def __init__(
            self,
            graph: nx.Graph,
            rounds: int,
            problem_level,
            model_name: str = "gpt-4o-mini",
            model_provider: str = "openai",
            chain_of_thought: bool = True,
            is_must_send: bool = True,
    ):
        Matching.__init__(
            self,
            graph=graph,
            rounds=rounds,
            problem_level=problem_level,
            model_name=model_name,
            model_provider=model_provider,
            chain_of_thought=chain_of_thought,
            is_must_send=is_must_send,
        )
        FunctionPseudoCodeOnly.__init__(self)
        self.problem_level = problem_level

    def get_symbolic_functions(self):
        main_function = """
1) establish_priority_and_parent
• Sort all agent names alphabetically.
• Higher = neighbors whose names come BEFORE yours; Lower = neighbors whose names come AFTER yours.
• If Higher is non-empty, set escalation_contact = the alphabetically last in Higher; otherwise None.

2) init_state
• matched_with = None                       # your committed partner (None if free).
• pending_proposal = None                   # who you most recently proposed to.
• candidate_list = neighbors in preferred order (default: alphabetical).
• context = {nei: {status: "unknown", partner: None, phase_id: -1} for nei in neighbors}
• tried = set()                             # remembers (Higher_signature, candidate) already failed.
• Higher_signature = []                     # snapshot of (h, status, partner) for all Higher used to key 'tried'.
• phase_id = 0                              # increases whenever your state meaningfully changes.
• On entry: zero everything above and compute Higher_signature from current context for all Higher.

3) choose_first_candidate
• Wait rule: if Higher is empty, act immediately; else wait until you received STATUS from every Higher or a timeout.
• Recompute Higher_signature = sorted list of (h, context[h].status, context[h].partner).
• Define FEASIBLE(cand):
  – cand is not (context[cand].status == "committed" and context[cand].partner != me).
  – If context[cand].status == "tentative", only consider cand if outranks(me, context[cand].partner) is True.
  – (Higher_signature, cand) not in tried.
• Scan candidate_list; pick the first FEASIBLE candidate.
• If found:
  – pending_proposal = cand
  – SEND PROPOSE(me) to cand
  – phase_id += 1; SEND STATUS(mode="tentative", partner=cand, phase_id) to all Lower
• If none found:
  – SEND BACKTRACK to escalation_contact (or FAIL if None)

4) on_propose_from_neighbor(nei)
• Collect all current proposers (including nei) this round.
• Pick the earliest by alphabetical name as winner.
• If matched_with is None (free):
  – ACCEPT the winner, set matched_with = winner (tentative), REJECT all others.
  – phase_id += 1; SEND STATUS(mode="tentative", partner=winner, phase_id) to Lower
• If already tentative/committed:
  – If winner outranks current tentative partner (alphabetical tie-break):
    * REJECT current tentative, ACCEPT winner, set matched_with = winner
  – Else REJECT winner

5) on_accept_from(candidate)
• on_accept_from(candidate):
  – matched_with = candidate
  – pending_proposal = None
  – SEND COMMIT(me↔candidate) to candidate
  – phase_id += 1; SEND STATUS(mode="committed", partner=candidate, phase_id) to all Lower

6) on_reject_from(candidate):
  – tried.add((Higher_signature, candidate))
  – Call propose_next_or_backtrack()

7) propose_next_or_backtrack
• Recompute Higher_signature (Higher may have changed).
• For cand in candidate_list:
  – If FEASIBLE(cand): pending_proposal = cand; SEND PROPOSE(me) to cand; phase_id += 1; 
    SEND STATUS("tentative", cand, phase_id) to Lower; RETURN
• If no FEASIBLE remains: backtrack_if_needed()

8) backtrack_if_needed
• If escalation_contact is None:
  – SEND FAIL to all Lower; stop taking new proposals; remain free.
• Else:
  – SEND BACKTRACK to escalation_contact
  – SEND STATUS(mode="requesting_help", partner=None, phase_id) to all Lower
  – Temporarily pause accepting new tentative matches until a Higher status change arrives.

9) on_backtrack_from_child(child)
• If you are only TENTATIVELY matched (not COMMITTED):
  – Try propose_next_or_backtrack() (may switch your tentative partner)
• If you are COMMITTED:
  – Decide if releasing partner could unblock the child without harming lexicographic priorities:
    * If yes: SEND UNDO_COMMIT to your partner; set matched_with=None; then propose_next_or_backtrack()
    * If not: forward BACKTRACK to escalation_contact (or FAIL if None)
• Purpose: locally explore alternatives before pushing pressure further up.

9) share_status
• On any change, SEND STATUS to all Lower with:
  – mode ∈ {"free","tentative","committed","requesting_help"}
  – partner (name or None)
  – phase_id (monotone counter) and your name for reproducible tie-breaks
• Receivers update context[you] accordingly so FEASIBLE() stays accurate and ignore STATUS with lower phase_id.

10) finalize_now

• If this is the final round (one-shot forced-commit; no new PROPOSE loops):
  1) Close open offers:
     – If pending_proposal ≠ None and unanswered: mark tried under Higher_signature; clear pending_proposal.
  2) Forced-commit rule (deterministic, single step):
     – If you and p are **mutually tentative** (both point to each other):
         * If outranks(me, p) is True → SEND COMMIT(me↔p). Else wait; accept COMMIT from p.
     – Else if you are free and see a **clean candidate q**:
         * Clean = q is free OR q is tentative only with you OR q is committed to you.
         * If outranks(me, any other contender of q) is True → SEND COMMIT(me↔q).
     – If you receive multiple COMMITs, **accept the earliest sender** alphabetically; REJECT others.
"""
        return main_function


class MatchingBackTrackCompletePseudoCode(Matching, CompletePseudoCode):
    def __init__(
        self,
        graph: nx.Graph,
        rounds: int,
        problem_level,
        model_name: str = "gpt-4o-mini",
        model_provider: str = "openai",
        chain_of_thought: bool = True,
        is_must_send: bool = True,
    ):
        Matching.__init__(
            self,
            graph=graph,
            rounds=rounds,
            problem_level=problem_level,
            model_name=model_name,
            model_provider=model_provider,
            chain_of_thought=chain_of_thought,
            is_must_send=is_must_send,
        )
        CompletePseudoCode.__init__(self)
        self.problem_level = problem_level

    def get_symbolic_pseudo_code(self):
        main_function = """
STEP 0 — Create a clear notion of “who you must listen to” (Higher) and “who follows your lead” (Lower) alphabetically.
Choose exactly one neighbor above you as the escalation target so that dead-ends have a single, predictable escape path.
How: establish_priority_and_parent
• Sort all agent names alphabetically.
• Higher = neighbors whose names come BEFORE yours; Lower = neighbors whose names come AFTER yours.
• If Higher is non-empty, set escalation_contact = the alphabetically last in Higher; otherwise None.

STEP 1 — Initialize your local search state so you can explore candidates and remember what failed.
Set up a compact “ledger” of what you know (neighbors’ states), what you are trying (your tentative choice), and what already failed 
(so you don’t repeat it). Keep a monotonically increasing phase_id for deduping and ignoring stale messages.
How: init_state
• matched_with = None                       # your committed partner (None if free).
• pending_proposal = None                   # who you most recently proposed to.
• candidate_list = neighbors in preferred order (default: alphabetical).
• context = {nei: {status: "unknown", partner: None, phase_id: -1} for nei in neighbors}
• tried = set()                             # remembers (Higher_signature, candidate) already failed.
• Higher_signature = []                     # snapshot of (h, status, partner) for all Higher used to key 'tried'.
• phase_id = 0                              # increases whenever your state meaningfully changes.
• On entry: zero everything above and compute Higher_signature from current context for all Higher.

STEP 2 — Choose your first candidate consistently with what Higher neighbors already decided.
Act only after you understand Higher’s constraints; then attempt the best neighbor that’s currently attainable. 
Your first move should neither collide with a known commitment nor repeat a previously failed attempt under the same Higher context.
How: choose_first_candidate
• Wait rule: if Higher is empty, act immediately; else wait until you received STATUS from every Higher or a timeout.
• Recompute Higher_signature = sorted list of (h, context[h].status, context[h].partner).
• Define FEASIBLE(cand):
  – cand is not (context[cand].status == "committed" and context[cand].partner != me).
  – If context[cand].status == "tentative", only consider cand if outranks(me, context[cand].partner) is True.
  – (Higher_signature, cand) not in tried.
• Scan candidate_list; pick the first FEASIBLE candidate.
• If found:
  – pending_proposal = cand
  – SEND PROPOSE(me) to cand
  – phase_id += 1; SEND STATUS(mode="tentative", partner=cand, phase_id) to all Lower
• If none found:
  – SEND BACKTRACK to escalation_contact (or FAIL if None)

STEP 3 — Decide what to do when someone proposes to you.
Treat concurrent proposals like bids: accept exactly one, deterministically, and decline the rest. 
Prefer the highest-priority proposer to keep the global search consistent and to avoid oscillations.
How: on_propose_from_neighbor(nei)
• Collect all current proposers (including nei) this round.
• Pick the earliest by alphabetical name as winner.
• If matched_with is None (free):
  – ACCEPT the winner, set matched_with = winner (tentative), REJECT all others.
  – phase_id += 1; SEND STATUS(mode="tentative", partner=winner, phase_id) to Lower
• If already tentative/committed:
  – If winner outranks current tentative partner (alphabetical tie-break):
    * REJECT current tentative, ACCEPT winner, set matched_with = winner
  – Else REJECT winner

STEP 4 — React to responses to your own proposal so you either lock it in or keep searching.
When accepted, convert tentative to committed and broadcast stability; when rejected, record that failure and move on. You either strengthen a match or prune a branch of the search space.
How: on_accept_from(candidate) / on_reject_from(candidate)
• on_accept_from(candidate):
  – matched_with = candidate
  – pending_proposal = None
  – SEND COMMIT(me↔candidate) to candidate
  – phase_id += 1; SEND STATUS(mode="committed", partner=candidate, phase_id) to all Lower
• on_reject_from(candidate):
  – tried.add((Higher_signature, candidate))
  – Call propose_next_or_backtrack()

STEP 5 — Try the next option; if none exist, escalate upward.
Keep iterating candidates that are currently feasible; if you exhaust them, escalate so that pressure propagates toward Higher priority 
agents who can free capacity or change their choices.
How: propose_next_or_backtrack
• Recompute Higher_signature (Higher may have changed).
• For cand in candidate_list:
  – If FEASIBLE(cand): pending_proposal = cand; SEND PROPOSE(me) to cand; phase_id += 1; 
    SEND STATUS("tentative", cand, phase_id) to Lower; RETURN
• If no FEASIBLE remains: backtrack_if_needed()

STEP 6 — Ask the closest Higher neighbor to reconsider when you are blocked.
When you cannot proceed, request relief from the designated Higher neighbor. 
Pause accepting new tentatives while waiting, to avoid thrashing against the same constraint.
How: backtrack_if_needed
• If escalation_contact is None:
  – SEND FAIL to all Lower; stop taking new proposals; remain free.
• Else:
  – SEND BACKTRACK to escalation_contact
  – SEND STATUS(mode="requesting_help", partner=None, phase_id) to all Lower
  – Temporarily pause accepting new tentative matches until a Higher status change arrives.

STEP 7 — When a Lower neighbor backtracks to you, try to free capacity or pass the backtrack upward.
First attempt a local reconfiguration (switch tentative or release a commit only if it’s safe); 
if you cannot help without violating priority discipline, push the backtrack upward unchanged.
How: on_backtrack_from_child(child)
• If you are only TENTATIVELY matched (not COMMITTED):
  – Try propose_next_or_backtrack() (may switch your tentative partner)
• If you are COMMITTED:
  – Decide if releasing partner could unblock the child without harming lexicographic priorities:
    * If yes: SEND UNDO_COMMIT to your partner; set matched_with=None; then propose_next_or_backtrack()
    * If not: forward BACKTRACK to escalation_contact (or FAIL if None)
• Purpose: locally explore alternatives before pushing pressure further up.

STEP 8 — Keep everyone synced with minimal chatter so feasibility checks are cheap.
Share only meaningful changes, tagged with a phase_id, so neighbors can quickly update their view and ignore stale data. 
This keeps the network quiet while maintaining enough information for correct feasibility tests.
How: share_status
• On any change, SEND STATUS to all Lower with:
  – mode ∈ {"free","tentative","committed","requesting_help"}
  – partner (name or None)
  – phase_id (monotone counter) and your name for reproducible tie-breaks
• Receivers update context[you] accordingly so FEASIBLE() stays accurate and ignore STATUS with lower phase_id.

STEP 9 — Finish cleanly (and force a decision) when nothing moves or when you reach the final round.
End with no dangling offers: upgrade safe pairs to committed, pick exactly one winner in ties, and otherwise stay free. 
In the final round, perform a one-shot **forced-commit** so at least some commitments are made.

How: finalize_now

• If this is the final round (one-shot forced-commit; no new PROPOSE loops):
  1) Close open offers:
     – If pending_proposal ≠ None and unanswered: mark tried under Higher_signature; clear pending_proposal.
  2) Forced-commit rule (deterministic, single step):
     – If you and p are **mutually tentative** (both point to each other):
         * If outranks(me, p) is True → SEND COMMIT(me↔p). Else wait; accept COMMIT from p.
     – Else if you are free and see a **clean candidate q**:
         * Clean = q is free OR q is tentative only with you OR q is committed to you.
         * If outranks(me, any other contender of q) is True → SEND COMMIT(me↔q).
     – If you receive multiple COMMITs, **accept the earliest sender** alphabetically; REJECT others.

"""
        return main_function
########
#
#
# ######### Consensus
class Consensus(LiteralMessagePassing):
    """Implementation for consensus without malicious parties."""

    def __init__(self, graph, rounds, model_name="gpt-4o", model_provider="openai", chain_of_thought=True,
                 is_must_send=False):
        self.rounds = rounds
        self.short_task_description = "Your goal is to agree on a single value with the other agents. The possible values that you can decide on are either 0 or 1."
        self.long_problem_description = (
            "After the last round, each agent must decide on a single value."
        )
        self.question_for_prediction = "Please answer with the value you agreed on."

        super().__init__(graph, model_name, model_provider, chain_of_thought, is_must_send=is_must_send)

    def get_basic_long_prob_description(self):
        return "After the last round, each agent must decide on a single value."

    def get_score(self, answers: list[str | None]) -> float:
        """
        Returns a score between 0 and 1 representing the fraction of agents
        that agreed on the majority value (0 or 1).
        - 1.0 means perfect consensus (everyone chose the same valid value).
        - 0.0 means no agreement.
        - Any non-0/1 answers are treated as votes *against* the majority.
        """
        # Count votes
        count_0 = sum(1 for ans in answers if ans == "0")
        count_1 = sum(1 for ans in answers if ans == "1")

        # Determine majority count (if tie, majority = max which will be equal)
        majority_count = max(count_0, count_1)

        # Denominator = all agents (even invalid/None votes count against consensus)
        return majority_count / len(answers) if answers else 0.0
    def get_valid_answers(self, node_id):
        return ["0", "1"]

################# VertexCover
def score_vertex_cover(results, graph):
    def vertex_cover(results):
        covered_edges = sum([1 for u, v in graph.edges if (results[u] == "Yes" or results[v] == "Yes")])
        return covered_edges / graph.number_of_edges()

    coverage = vertex_cover(results)

    minimality = 0
    cover_size = 0
    for u in graph.nodes:
        if results[u] is not None:
            if results[u] == "Yes":
                cover_size += 1
                _results = results.copy()
                _results[u] = "No"
                if vertex_cover(_results) < 1.0:
                    minimality += 1

    return coverage * minimality / cover_size

class VertexCover(LiteralMessagePassing):
    def __init__(self, graph, rounds, problem_level = ProblemLevel.hard, model_name="gpt-4o-mini", model_provider="openai", chain_of_thought=True, is_must_send = False):
        """https://math.stackexchange.com/a/1764484.
        A practical example is that the minimal vertex cover receives resources and is
        important that every channel of communication always has access to this resource,
        meaning there is no need for two-hop communication to obtain some resource. Fundamentally,
        the agents solve a resource allocation problem, which also touches into fairness.
        """
        super().__init__(graph, problem_level,model_name, model_provider, chain_of_thought, is_must_send=is_must_send)
        self.rounds = rounds
        self.short_task_description = """Your task is to select, among all agents, a group of coordinators
such that whenever two agents communicate at least one of them is a coordinator. The group of coordinators
should be selected such that every coordinator has at least one neighbor who is not a coordinator. 
The objective is to make the group of coordinators as small as possible while still satisfying the conditions."""
        # NOTE: Closer to definition
        # reverting any coordinator back to a regular agent results_mgm in at least two agents
        # who can communicate but none of whom is a coordinator.

        self.long_problem_description = (
            "You will be requested to state whether you are a coordinator. The response should either be 'Yes' or 'No'. "
        )
        self.question_for_prediction = "Are you a coordinator?"


    def  get_basic_long_prob_description(self):
        return "You will be requested to state whether you are a coordinator. The response should either be 'Yes' or 'No'. "

    def get_required_calculations(self, answers: list[str]):
        """
        Exact components for minimal-VC and minimum-VC on small graphs (n<=16).
        Returns only JSON-friendly types (no sets).
        """
        import networkx as nx
        from itertools import combinations

        G = self.graph
        V = list(G.nodes())
        n, m = len(V), G.number_of_edges()

        # --- helpers ---
        def disp_name(u):
            return G.nodes[u].get("name", str(u))

        def is_yes(a) -> bool:
            return str(a).strip().lower() in {"yes", "y", "true", "t", "1"}

        def is_cover_for(Gg, Sset) -> bool:
            for u, v in Gg.edges():
                if (u not in Sset) and (v not in Sset):
                    return False
            return True

        def max_matching_lb(Gg) -> int:
            M = nx.algorithms.matching.max_weight_matching(Gg, maxcardinality=True)
            return len(M)

        def tau_exact(Gg) -> int:
            """Exact minimum vertex cover size via enumeration + LB prune (fine for n<=16)."""
            if Gg.number_of_edges() == 0:
                return 0
            nodes = list(Gg.nodes())
            lb = max_matching_lb(Gg)
            for k in range(lb, Gg.number_of_nodes() + 1):
                for idxs in combinations(range(len(nodes)), k):
                    S_try = {nodes[i] for i in idxs}
                    if is_cover_for(Gg, S_try):
                        return k
            return Gg.number_of_nodes()  # fallback

        # --- parse answers -> S (as a set internally), export as lists ---
        S_set = {u for u, a in zip(V, answers) if is_yes(a)}
        S_nodes = [u for u in V if u in S_set]  # ordered by V
        S_names = [disp_name(u) for u in S_nodes]
        chosen_size = len(S_nodes)

        # --- valid/invalid buckets (by your b_S definition) ---
        invalid_inside = valid_inside = invalid_outside = valid_outside = 0
        for v in V:
            neigh = list(G.neighbors(v))
            if v in S_set:
                if all(u in S_set for u in neigh):
                    invalid_inside += 1  # redundant Yes
                else:
                    valid_inside += 1  # necessary Yes
            else:
                if all(u in S_set for u in neigh):
                    valid_outside += 1  # safe No
                else:
                    invalid_outside += 1  # witness of an uncovered edge

        # --- uncovered subgraph H from invalid-outside vertices ---
        W_nodes = [v for v in V if (v not in S_set and any(u not in S_set for u in G.neighbors(v)))]
        H = G.subgraph(W_nodes).copy()
        uncovered_edges = H.number_of_edges()
        covered_edges = m - uncovered_edges
        coverage_fraction = 1.0 if m == 0 else covered_edges / m

        # --- exact optima ---
        tau_G = tau_exact(G)  # minimum VC size of G
        k_patch = tau_exact(H)  # fewest extra Yes needed to fix misses (= tau(H))

        # --- flags ---
        is_cover = (uncovered_edges == 0)  # equivalently, invalid_outside == 0
        is_minimal = is_cover and (invalid_inside == 0)
        is_minimum = is_cover and (chosen_size == tau_G)

        # --- scores ---
        # Minimal-VC score: 1 - (#invalid / |V|)
        invalid_total = invalid_inside + invalid_outside
        if m == 0:
            min_cover_score = 1.0 if chosen_size == 0 else 0.0
        else:
            min_cover_score = 1.0 - (invalid_total / n if n > 0 else 0.0)

        # Minimum-VC score (balanced undershoot/overshoot):
        # (1 - k/τ(G)) * τ(G)/max{τ(G), |S|}
        if tau_G == 0:
            undershoot = 1.0
            overshoot = 1.0 if chosen_size == 0 else 0.0
        else:
            undershoot = max(0.0, 1.0 - (k_patch / tau_G))
            overshoot = tau_G / max(tau_G, chosen_size or tau_G)
        min_opt_score = undershoot * overshoot

        return {
            # graph + selection (lists, not sets)
            "n": n,
            "m": m,
            "S_size": chosen_size,
            "S_nodes": S_nodes,  # JSON-friendly list of node ids
            "S_names": S_names,  # JSON-friendly list of names

            # buckets
            "invalid_inside": invalid_inside,
            "invalid_outside": invalid_outside,
            "valid_inside": valid_inside,
            "valid_outside": valid_outside,

            # coverage
            "covered_edges": covered_edges,
            "uncovered_edges": uncovered_edges,
            "coverage_fraction": coverage_fraction,

            # uncovered subgraph + exact patch size
            "H_size": len(W_nodes),
            "k_patch": k_patch,  # = τ(H)

            # exact optimum for G
            "tau_G": tau_G,

            # properties
            "is_cover": is_cover,
            "is_minimal": is_minimal,
            "is_minimum": is_minimum,

            # scores
            "MinCoverScore": min_cover_score,
            "MinOptimalScore": min_opt_score,
            "MinOptimalScore_factors": {
                "undershoot": undershoot,  # 1 - k/τ(G)
                "overshoot": overshoot  # τ(G)/max{τ(G), |S|}
            },
        }

    def is_vertex_cover(self, results):
        covered_edges = sum([1 for u, v in self.graph.edges if (results[u] == "Yes" or results[v] == "Yes")])
        return covered_edges == self.graph.number_of_edges()

    def get_score(self, results: list[str]) -> float:
        """
        Minimal Vertex Cover score in [0,1]:
        1 - (invalid_total / |V|); equals 1 iff S is a minimal cover.
        """
        calc = self.get_required_calculations(results)
        return float(calc.get("MinCoverScore", 0.0))

    def get_score_opt(self, results: list[str]) -> float:
        """
        Minimum Vertex Cover score in [0,1]:
        (1 - k/τ(G)) * τ(G)/max{τ(G), |S|}; equals 1 iff S is a minimum cover.
        """
        calc = self.get_required_calculations(results)
        return float(calc.get("MinOptimalScore", 0.0))

    def get_SSE_easy(self, answers):
        aaa  =self.get_required_calculations(answers)
        invalid_inside = aaa["invalid_inside"]
        invalid_outside = aaa["invalid_outside"]
        return (invalid_inside+invalid_outside)**2

    def get_SSE_hard(self, answers):
        aaa = self.get_required_calculations(answers)
        opt_min = aaa["tau_G"]
        valid_inside = aaa["valid_inside"]

        invalid_inside = aaa["invalid_inside"]
        invalid_outside = aaa["invalid_outside"]


        return (opt_min-valid_inside)**2+(invalid_inside+invalid_outside)**2

    def get_valid_answers(self, node_id):
        return ["Yes", "No"]
################# VertexCover DGP

############################################################################################



class VertexCoverPriorityCode(VertexCover):
    def __init__(self, graph: nx.Graph, rounds: int, run_until_converge: bool = False, **kwargs):
        super().__init__(graph=graph, rounds=rounds, **kwargs)
        self._order = list(sorted(self.graph.nodes()))
        self.model_name = "python"
        #if run_until_converge:
        self.rounds = 1000  # large budget; solver will still be round-synchronous

    async def bootstrap(self):
        return

    async def pass_messages(self):
        solver = SolveVertexCoverPriority(self.graph, rounds=self.rounds)
        answers = solver.run()  # List[str] ("YES"/"NO") aligned with solver.order

        # Ensure alignment with this agent's ordering if you rely on it elsewhere
        if solver.order != self._order:
            index_in_solver = {u: i for i, u in enumerate(solver.order)}
            remapped: List[Optional[str]] = []
            for u in self._order:
                remapped.append(answers[index_in_solver[u]])
            return remapped
        return answers

class VertexCoverNaiveCompletePseudoCode(VertexCover, CompletePseudoCode):
    def __init__(self, graph, rounds, problem_level, model_name="gpt-4o-mini", model_provider="openai",
                 chain_of_thought=True):
        VertexCover.__init__(self, graph, rounds, problem_level, model_name, model_provider, chain_of_thought,
                             is_must_send=True)
        CompletePseudoCode.__init__(self)

    def get_symbolic_pseudo_code(self):
        return """
GLOBAL — Priority, messages, and promises
- Priority: earlier names = higher priority (lexicographic order).
- State: cover_state is either YES (in the cover) or NO (out of the cover).
- Messages: STATUS(YES/NO), FLIP_NOTICE, DROP_REQ, DROP_ACK, DROP_NACK.
- Promise: if you send DROP_ACK, you must stay YES for the rest of that round and cannot start your own drop.

STEP 0 — Initialize your state
Start off safe: assume you are part of the cover and know who your neighbors are. Clear your memory of past messages.  
How: init_variables
- Set cover_state := YES.
- Record the fixed priority order (names sorted lexicographically).
- inbox := {} (last seen STATUS/messages from neighbors).
- DROPPED_THIS_ROUND := False (flag for cooldown).
- will_hold_yes := False (tracks if you promised to stay YES this round).

STEP 1 — Share what you’re doing (each round)
At the start of a round, tell neighbors if you are YES or NO, and refresh your view of what they are doing.  
How: share_status
- Reset will_hold_yes := False.
- If DROPPED_THIS_ROUND is True, clear it back to False (cooldown handled).
- BROADCAST STATUS(cover_state).
- Update inbox with all neighbors’ current STATUS and any messages.

STEP 2 — Repair uncovered edges
If you and a neighbor are both NO, one of you must flip to YES. The higher-priority endpoint fixes the problem.  
How: repair_uncovered_edges
- If cover_state == NO and you see any neighbor with cover_state == NO:
  • Pick one such neighbor (only one per round).
  • If you outrank that neighbor:
      - SEND FLIP_NOTICE to them.
      - Set cover_state := YES.
      - BROADCAST STATUS(YES).
  • Otherwise, do nothing and let the higher neighbor handle it.

STEP 3 — Detect local redundancy
If all your neighbors are YES, you are not needed in the cover right now and could potentially drop out.  
How: detect_redundant_yes
- Set redundant := True if:
  • cover_state == YES, and
  • every neighbor in inbox is also YES.
- Special case: if you have no neighbors at all, treat yourself as redundant.

STEP 4A — Start a safe drop (your side)
If you are redundant and outrank all YES neighbors, you can ask them for permission to drop. You only leave if everyone agrees.  
How: drop_redundant_yes_originator
- Check:
  • You are redundant.
  • You are higher than every YES neighbor.
  • DROPPED_THIS_ROUND == False.
- Then:
  • SEND DROP_REQ to every YES neighbor.
  • WAIT for replies this round (DROP_ACK or DROP_NACK).
- Decide:
  • If everyone replied with DROP_ACK → set cover_state := NO; set DROPPED_THIS_ROUND := True; BROADCAST STATUS(NO).
  • Otherwise → stay YES.

STEP 4B — Answer a drop request (neighbor side)
When a neighbor asks to drop, only allow it if you can safely hold the cover this round.  
How: handle_drop_req_as_neighbor
- On receiving DROP_REQ from neighbor u:
  1) If you are NO → SEND DROP_NACK to u.
  2) If you already promised someone else this round (will_hold_yes == True) → SEND DROP_NACK.
  3) If a higher-priority neighbor also asked you this round → SEND DROP_NACK.
  4) Otherwise:
      - SEND DROP_ACK to u.
      - Set will_hold_yes := True (promise to remain YES and not drop this round).

STEP 5 — Continue until calm
Keep repeating the cycle of sharing, repairing, and dropping until nothing changes.  
How: continue_until_stable
- Each round do:
  1) share_status
  2) repair_uncovered_edges
  3) detect_redundant_yes → drop_redundant_yes_originator
  4) handle_drop_req_as_neighbor
- Stop if a full round finishes with:
  • No FLIP_NOTICE,
  • No DROP_REQ,
  • No DROP_ACK or DROP_NACK,
  • No changes to cover_state.
- Otherwise keep going, or stop if you hit the round budget.

"""


class VertexCoverNaiveMainPseudocodeOnly(VertexCover, MainPseudocodeOnly):
    def __init__(self, graph, rounds, problem_level, model_name="gpt-4o-mini", model_provider="openai",
                 chain_of_thought=True):
        VertexCover.__init__(self, graph, rounds, problem_level, model_name, model_provider, chain_of_thought,
                          is_must_send=True)
        MainPseudocodeOnly.__init__(self)

    def get_symbolic_pseudo_code(self):
        return """
    GLOBAL — Priority, messages, and promises
    - Priority: earlier names = higher priority (lexicographic order).
    - State: cover_state is either YES (in the cover) or NO (out of the cover).
    - Messages: STATUS(YES/NO), FLIP_NOTICE, DROP_REQ, DROP_ACK, DROP_NACK.
    - Promise: if you send DROP_ACK, you must stay YES for the rest of that round and cannot start your own drop.

    STEP 0 — Initialize your state
    Start off safe: assume you are part of the cover and know who your neighbors are. Clear your memory of past messages.  

    STEP 1 — Share what you’re doing (each round)
    At the start of a round, tell neighbors if you are YES or NO, and refresh your view of what they are doing.  

    STEP 2 — Repair uncovered edges
    If you and a neighbor are both NO, one of you must flip to YES. The higher-priority endpoint fixes the problem.  

    STEP 3 — Detect local redundancy
    If all your neighbors are YES, you are not needed in the cover right now and could potentially drop out.  
   
    STEP 4A — Start a safe drop (your side)
    If you are redundant and outrank all YES neighbors, you can ask them for permission to drop. You only leave if everyone agrees.  

    STEP 4B — Answer a drop request (neighbor side)
    When a neighbor asks to drop, only allow it if you can safely hold the cover this round.  

    STEP 5 — Continue until calm
    Keep repeating the cycle of sharing, repairing, and dropping until nothing changes.  
  

    """


class VertexCoverNaiveFunctionPseudoCodeOnly(VertexCover, FunctionPseudoCodeOnly):
    def __init__(self, graph, rounds, problem_level, model_name="gpt-4o-mini", model_provider="openai",
                 chain_of_thought=True):
        VertexCover.__init__(self, graph, rounds, problem_level, model_name, model_provider, chain_of_thought,
                          is_must_send=True)
        FunctionPseudoCodeOnly.__init__(self)

    def get_symbolic_functions(self):
        return """
    GLOBAL — Priority, messages, and promises
    - Priority: earlier names = higher priority (lexicographic order).
    - State: cover_state is either YES (in the cover) or NO (out of the cover).
    - Messages: STATUS(YES/NO), FLIP_NOTICE, DROP_REQ, DROP_ACK, DROP_NACK.
    - Promise: if you send DROP_ACK, you must stay YES for the rest of that round and cannot start your own drop.

    1) init_variables
    - Set cover_state := YES.
    - Record the fixed priority order (names sorted lexicographically).
    - inbox := {} (last seen STATUS/messages from neighbors).
    - DROPPED_THIS_ROUND := False (flag for cooldown).
    - will_hold_yes := False (tracks if you promised to stay YES this round).

    2) share_status
    - Reset will_hold_yes := False.
    - If DROPPED_THIS_ROUND is True, clear it back to False (cooldown handled).
    - BROADCAST STATUS(cover_state).
    - Update inbox with all neighbors’ current STATUS and any messages.

    3) repair_uncovered_edges
    - If cover_state == NO and you see any neighbor with cover_state == NO:
      • Pick one such neighbor (only one per round).
      • If you outrank that neighbor:
          - SEND FLIP_NOTICE to them.
          - Set cover_state := YES.
          - BROADCAST STATUS(YES).
      • Otherwise, do nothing and let the higher neighbor handle it.

    4) detect_redundant_yes
    - Set redundant := True if:
      • cover_state == YES, and
      • every neighbor in inbox is also YES.
    - Special case: if you have no neighbors at all, treat yourself as redundant.

    5) drop_redundant_yes_originator
    - Check:
      • You are redundant.
      • You are higher than every YES neighbor.
      • DROPPED_THIS_ROUND == False.
    - Then:
      • SEND DROP_REQ to every YES neighbor.
      • WAIT for replies this round (DROP_ACK or DROP_NACK).
    - Decide:
      • If everyone replied with DROP_ACK → set cover_state := NO; set DROPPED_THIS_ROUND := True; BROADCAST STATUS(NO).
      • Otherwise → stay YES.

    6) handle_drop_req_as_neighbor
    - On receiving DROP_REQ from neighbor u:
      1) If you are NO → SEND DROP_NACK to u.
      2) If you already promised someone else this round (will_hold_yes == True) → SEND DROP_NACK.
      3) If a higher-priority neighbor also asked you this round → SEND DROP_NACK.
      4) Otherwise:
          - SEND DROP_ACK to u.
          - Set will_hold_yes := True (promise to remain YES and not drop this round).

    7) continue_until_stable
    - Each round do:
      1) share_status
      2) repair_uncovered_edges
      3) detect_redundant_yes → drop_redundant_yes_originator
      4) handle_drop_req_as_neighbor
    - Stop if a full round finishes with:
      • No FLIP_NOTICE,
      • No DROP_REQ,
      • No DROP_ACK or DROP_NACK,
      • No changes to cover_state.
    - Otherwise keep going, or stop if you hit the round budget.

    """


class VertexCoverNaiveFunctionDescriptionOnly(VertexCover, FunctionDescriptionOnly):
    def __init__(self, graph, rounds, problem_level, model_name="gpt-4o-mini", model_provider="openai",
                 chain_of_thought=True):
        VertexCover.__init__(self, graph, rounds, problem_level, model_name, model_provider, chain_of_thought,
                          is_must_send=True)
        FunctionDescriptionOnly.__init__(self)

    # Only override the MODE block; keep Coloring flow in _intro_problem_unique_rules()

    def get_symbolic_functions(self):
        return """
    1) init_variables
    - Start in the cover (YES), record who your neighbors are, and clear your memory and flags.

    2) share_status
    - At the beginning of each round, tell neighbors your state (YES/NO) and update what you know about them.

    3) repair_uncovered_edges
    - If you and a neighbor are both out (NO), the higher-priority one returns to YES so the edge is covered.

    4) detect_redundant_yes
    - If you are YES but all your neighbors are also YES, you might not be needed and can consider dropping.

    5) drop_redundant_yes_originator
    - If you want to drop, ask all your YES neighbors. Only leave if everyone agrees to hold the cover.

    6) handle_drop_req_as_neighbor
    - When a neighbor asks to drop, allow it only if you can safely stay YES and no higher-priority neighbor also asked.

    7) continue_until_stable
    - Keep repeating the share → repair → redundancy check → drop cycle until nothing changes for a full round, or the round budget runs out.
    """


class VertexCoverNaiveHeuristicHintGuidedAgentV2(VertexCover, HeuristicHintGuided):
    def __init__(self, graph, rounds, problem_level, model_name="gpt-4o-mini", model_provider="openai",
                 chain_of_thought=True):
        VertexCover.__init__(self, graph, rounds, problem_level, model_name, model_provider, chain_of_thought,
                          is_must_send=True)
        HeuristicHintGuided.__init__(self)

    def get_intro(self):
        return """
Continuously share your current YES/NO state with neighbors and keep an up-to-date view of theirs. If you ever observe an uncovered edge (NO–NO), the higher-priority endpoint (earlier names = higher priority) should flip to YES to immediately restore coverage. When you are YES but all neighbors are YES, coordinate—respecting priority—to drop safely without simultaneous adjacent drops, and repeat this repair/trim cycle until no changes occur.
"""



class VertexCoverTrackCode(VertexCover):
    def __init__(self, graph: nx.Graph, rounds: int, **kwargs):
        super().__init__(graph=graph, rounds=rounds, **kwargs)
        self._order = list(sorted(self.graph.nodes()))
        self.model_name = "python"

        #self.rounds =10000

    async def bootstrap(self):
        return

    async def pass_messages(self):
        # Oracle: compute exact maximum matching once
        solver = SolveVertexCoverBacktrack(self.graph, rounds=self.rounds)
        answers = solver.run()  # List[Optional[str]] aligned with solver.order
        # Ensure alignment with this agent's ordering if you rely on it elsewhere
        if solver.order != self._order:
            # remap to this class's order
            index_in_solver = {u: i for i, u in enumerate(solver.order)}
            remapped: List[Optional[str]] = []
            for u in self._order:
                remapped.append(answers[index_in_solver[u]])
            return remapped
        return answers




class VertexCoverBackTrackCompletePseudoCode(VertexCover, CompletePseudoCode):
    def __init__(self, graph, rounds, problem_level, model_name="gpt-4o-mini", model_provider="openai",
                 chain_of_thought=True):
        VertexCover.__init__(self, graph, rounds, problem_level, model_name, model_provider, chain_of_thought,
                             is_must_send=True)
        CompletePseudoCode.__init__(self)
        self.problem_level = problem_level

    def get_symbolic_pseudo_code(self):
        main_function = """
GLOBAL — Priority, state, and messages
- Priority: alphabetical order; “Higher” = earlier name; “Lower” = later name.
- Local label ∈ {UNDECIDED, YES, NO}. Start UNDECIDED.
- Messages: STATUS(label, phase_id), BRANCH(edge=(u,v), choice), COVER_REQ(edge), COVER_LOCK(edge), REJECT_BRANCH, BACKTRACK, INCUMBENT(best_size, signature).
- phase_id: bump whenever your label/active branch changes; neighbors ignore older versions.
- Parent: your closest Higher neighbor (or none if you’re the root). Backtracking flows upward.

STEP 0 — Establish order and a single escalation parent above you
How: establish_order_and_parent
• Sort all names alphabetically; mark which neighbors are Higher/Lower.
• Your parent is the closest Higher; if none exist, you are the root.

STEP 1 — Initialize your search notebook
How: init_state
• label := UNDECIDED; phase_id := 0
• inbox := {} (last seen STATUS/phase_id from neighbors)
• tried_here := {}  # which branch choices you already tried at this node
• incumbent := None  # best (size, signature) you’ve heard so far

STEP 2 — After hearing Higher, pick a local witness edge and start a branch
How: choose_witness_and_start
• Wait until you’ve seen STATUS from all Higher (or a short timeout).
• Find an incident edge (u,v) that looks uncovered in your current view (no endpoint firmly YES).
• If you are the earlier endpoint on that edge (“edge captain”):
    - Try branches in this fixed order to ensure completeness:
      (A) BRANCH(edge=(u,v), choice = YOU=YES)
      (B) BRANCH(edge=(u,v), choice = OTHER=YES)  # implemented via COVER_REQ to the other endpoint
    - Start with (A) unless already tried; record the choice; phase_id += 1; SEND STATUS.

STEP 3 — Apply your current branch choice and its immediate implications
How: apply_branch_choice
• If choice is YOU=YES:
    - Set label := YES; phase_id += 1; SEND STATUS.
• If choice is OTHER=YES:
    - Set label := NO (for this branch only).
    - SEND COVER_REQ(edge) to the other endpoint asking them to hold YES on this edge.
    - Wait for COVER_LOCK or REJECT_BRANCH.

STEP 4 — Share your state and check a safe lower bound for pruning
How: share_status_and_prune
• On any change: SEND STATUS(label, phase_id).
• Compute LB := size of a greedy matching on currently uncovered edges (admissible).
• Let current_yes = number of vertices you believe are YES so far (in your visible partial).
• If incumbent exists with size B* and current_yes + LB ≥ B*:
    - SEND REJECT_BRANCH for this branch (can’t beat the incumbent); mark tried_here; move to the alternative or BACKTRACK.

STEP 5 — Let progress flow downward; escalate if blocked
How: descend_or_escalate
• If your branch holds (no contradiction) and Lower neighbors can keep branching, wait and observe.
• If you’re blocked locally (constraints can’t be satisfied under current Higher choices), SEND BACKTRACK to parent and pause until the context changes.

STEP 6 — When you receive a COVER_REQ, either lock the edge or reject the branch
How: on_cover_req(edge=(u,v), from=u_or_v)
• If you can set label := YES without contradicting your current branch, do it now, SEND COVER_LOCK(edge), SEND STATUS.
• If you cannot (e.g., you’re already NO under a stronger decision), SEND REJECT_BRANCH to the requester.

STEP 7 — React to feedback to either keep the branch or try the sibling
How: on_branch_feedback
• If you get REJECT_BRANCH on your current choice:
    - Mark it tried_here. If the sibling choice is untried, switch to it (Step 3) and continue.
    - If both choices are tried_here: SEND BACKTRACK to parent.
• If you get COVER_LOCK on OTHER=YES and everything remains consistent, keep going (Steps 4–5).

STEP 8 — Maintain and use the shared incumbent to prune
How: incumbent_updates
• If you learn INCUMBENT(B*, sig*): store it; use Step 4 pruning against B*.
• If you complete a full cover with size S and (incumbent is None or S < incumbent.size):
    - BROADCAST INCUMBENT(S, signature); SEND ACCEPT_BRANCH upward.

STEP 9 — When a child backtracks to you, try your sibling choice or pass it up
How: on_backtrack_from_child(child)
• If you are the “edge captain” and your sibling choice here is untried, switch to it (Step 3).
• Otherwise, forward BACKTRACK to parent (or, if you’re the root and stuck, finish with the incumbent).

STEP 10 — Finish cleanly when stable or out of time
How: finalize_now
• If a full round passes with no STATUS/BRANCH/BACKTRACK traffic and all visible edges are covered, ACCEPT_BRANCH upward.
• If time ends, return the best incumbent you know (root decides).
"""
        return main_function


class VertexCoverBackTrackFunctionPseudoCodeOnly(VertexCover, FunctionPseudoCodeOnly):
    def __init__(self, graph, rounds, problem_level, model_name="gpt-4o-mini", model_provider="openai",
                 chain_of_thought=True):
        VertexCover.__init__(self, graph, rounds, problem_level, model_name, model_provider, chain_of_thought,
                             is_must_send=True)

    def get_symbolic_functions(self):
        main_function = """
1) establish_order_and_parent
• Use alphabetical order; tag Higher/Lower; choose the closest Higher as parent (or none if root).

2) init_state
• label := UNDECIDED; phase_id := 0; inbox := {}; tried_here := {}; incumbent := None.

3) choose_witness_and_start
• After hearing Higher, pick a nearby uncovered edge you touch. If you are the earlier endpoint, start branching:
  try YOU=YES first, then OTHER=YES as the sibling later.

4) apply_branch_choice
• For YOU=YES: set label YES and announce.
• For OTHER=YES: set label NO, SEND COVER_REQ(edge) to the other endpoint, and wait for COVER_LOCK or REJECT_BRANCH.

5) share_status_and_prune
• On change, SEND STATUS(label, phase_id). Compute LB via greedy matching on uncovered edges.
• If current_yes + LB ≥ incumbent.size (when known), REJECT_BRANCH this branch and try the sibling or backtrack.

6) descend_or_escalate
• If your branch remains feasible, let Lower neighbors continue. If blocked, SEND BACKTRACK to parent and pause.

7) on_cover_req
• If consistent, set YES and SEND COVER_LOCK(edge); otherwise SEND REJECT_BRANCH to the requester.

8) on_branch_feedback
• If REJECT_BRANCH arrives for your current choice, mark tried and switch to the sibling if untried; else BACKTRACK.
• If COVER_LOCK arrives and nothing contradicts, continue the search.

9) incumbent_updates
• Use INCUMBENT(best_size, signature) to prune; publish a better one when you find a smaller cover.

10) on_backtrack_from_child
• If a sibling choice is untried here, switch to it; otherwise forward BACKTRACK to parent (root decides on failure).

11) finalize_now
• If quiet and all edges covered, ACCEPT_BRANCH upward; if time ends, output the best incumbent known.
"""
        return main_function


class VertexCoverBackTrackMainPseudocodeOnly(VertexCover, MainPseudocodeOnly):
    def __init__(self, graph, rounds, problem_level, model_name="gpt-4o-mini", model_provider="openai",
                 chain_of_thought=True):
        VertexCover.__init__(self, graph, rounds, problem_level, model_name, model_provider, chain_of_thought,
                             is_must_send=True)
        MainPseudocodeOnly.__init__(self)

    def get_symbolic_pseudo_code(self):
        main_function = """
STEP 0 — Establish order and parent (single escalation path upward).

STEP 1 — Init: label UNDECIDED; clear tried notes; phase_id=0.

STEP 2 — After hearing Higher, pick a local uncovered edge and start a branch (YOU=YES first, then OTHER=YES).

STEP 3 — Apply your current branch: set YES, or set NO and request the neighbor to cover (COVER_REQ).

STEP 4 — Announce and prune: use greedy-matching lower bound against the shared incumbent.

STEP 5 — Let Lower neighbors advance; if blocked, BACKTRACK to parent.

STEP 6 — On COVER_REQ: lock YES if possible, else REJECT_BRANCH.

STEP 7 — On feedback: keep a working branch or try the sibling; if both fail, BACKTRACK.

STEP 8 — Maintain a shared INCUMBENT; prune branches that can’t beat it; publish new bests.

STEP 9 — When children backtrack, try your sibling choice or pass the backtrack upward.

STEP 10 — Finish cleanly: accept when quiet and covered; otherwise return the best incumbent at time limit.
"""
        return main_function


class VertexCoverBackTrackFunctionDescriptionOnly(VertexCover, FunctionDescriptionOnly):
    def __init__(self, graph, rounds, problem_level, model_name="gpt-4o-mini", model_provider="openai",
                 chain_of_thought=True):
        VertexCover.__init__(self, graph, rounds, problem_level, model_name, model_provider, chain_of_thought,
                             is_must_send=True)
        FunctionDescriptionOnly.__init__(self)
        self.problem_level = problem_level

    def get_symbolic_functions(self):
        main_function = """
1) establish_order_and_parent
• Agree on lexical priority, and pick one Higher neighbor to escalate to.

2) init_state
• Start undecided; clear notes; track versions; prepare to record tried choices.

3) choose_witness_and_start
• Use a nearby uncovered edge to branch in a fixed order so every possibility is explored.

4) apply_branch_choice
• Enforce the chosen case locally and send a cover request when you defer coverage to your neighbor.

5) share_status_and_prune
• Speak only on change; compute an admissible lower bound and prune branches that can’t beat the best known.

6) descend_or_escalate
• Let progress flow downward; if stuck, backtrack to your parent.

7) on_cover_req
• Either take responsibility for the edge now or reject the branch cleanly.

8) on_branch_feedback
• Keep a branch that works; otherwise try the sibling or backtrack if both are exhausted.

9) incumbent_updates
• Share a new best cover when you find one; use it to cut search safely.

10) on_backtrack_from_child
• Help locally if you can; otherwise pass the backtrack up the chain.

11) finalize_now
• End with a consistent cover (accept) or with the best incumbent (time limit).
"""
        return main_function


class VertexCoverBackTrackHeuristicHintGuided(VertexCover, HeuristicHintGuided):
    def __init__(self, graph, rounds, problem_level, model_name="gpt-4o-mini", model_provider="openai",
                 chain_of_thought=True):
        VertexCover.__init__(self, graph, rounds, problem_level, model_name, model_provider, chain_of_thought,
                             is_must_send=True)
        HeuristicHintGuided.__init__(self)

    def get_intro(self):
        return (
        "Follow the shared alphabetical order: listen to higher-priority neighbors first, then make a consistent local choice and announce only when your state changes. "
        "Prefer simple local progress; if your choice runs into a conflict, try the alternative once, and if neither works, escalate to your designated higher neighbor and pause. "
        "As a group, remember the best complete cover seen so far and converge when a full round passes with no changes."
        )








class Coloring(LiteralMessagePassing):
    def __init__(self, graph: nx.Graph, rounds: int, problem_level, model_name="gpt-4o-mini",
                 model_provider="openai", chain_of_thought=True, is_must_send=False):
        LiteralMessagePassing.__init__(self, graph=graph, model_name=model_name, model_provider=model_provider,
                                       chain_of_thought=chain_of_thought, is_must_send=is_must_send,
                                       problem_level=problem_level)
        self.rounds = rounds

        if self.problem_level == ProblemLevel.poly:
            self.num_colors = max(graph.degree(n) for n in graph.nodes) + 1
            self.short_task_description = "Your task is to partition yourselves into groups such that agents who are neighbors are never in the same group."

        else:
            self.num_colors = 3
            self.short_task_description = (
                "Assign yourselves to groups so that neighboring agents are not in the same group when possible; "
                "otherwise, aim to minimize the number of neighboring pairs that share a group."
            )
            print("3 colors")
        self.colors = [f"Group {i + 1}" for i in range(self.num_colors)]

        self.long_problem_description = self.get_basic_long_prob_description()
        self.question_for_prediction = f"Which group do you assign yourself to?"

    def get_required_calculations(self,answers):
        valid_edges = sum(
            answers[u] is not None and
            answers[v] is not None and
            answers[u] != answers[v]
            for (u, v) in self.graph.edges
        )
        return {"valid_edges":valid_edges,"opt_valid_edges":_solve_max_3color_edges(self.graph)["satisfied_edges"]}


    def get_score_opt(self, answers):
        if self.problem_level == ProblemLevel.poly:
            return self.graph.number_of_edges()
        else:
            valid_edges = self.get_required_calculations(answers)["valid_edges"]
            # valid_edges = [answers[u] != answers[v] for (u, v) in self.graph.edges].count(True)
            max_valid_edges = _solve_max_3color_edges(self.graph)["satisfied_edges"]
            valid_ratio = valid_edges / max_valid_edges
            return valid_ratio


    def get_score(self, answers: list[str]) -> float:

        # all_valid = all(x in self.colors for x in answers)

        valid_edges = self.get_required_calculations(answers)["valid_edges"]
        # valid_edges = [answers[u] != answers[v] for (u, v) in self.graph.edges].count(True)
        valid_ratio = valid_edges / self.graph.number_of_edges()
        return valid_ratio

    def get_SSE_easy(self, answers):
        valid_edges = self.get_required_calculations(answers)["valid_edges"]
        invalid_valid_edges = self.graph.number_of_edges()- valid_edges
        return invalid_valid_edges**2


    def get_SSE_hard(self, answers):
        valid_edges = self.get_required_calculations(answers)["valid_edges"]
        if self.problem_level == ProblemLevel.hard:
            opt_edges = self.get_required_calculations(answers)["opt_valid_edges"]
        else:
            opt_edges = self.graph.number_of_edges()

        #invalid_valid_edges = self.graph.number_of_edges() - valid_edges
        return (opt_edges-valid_edges)**2 #+ invalid_valid_edges ** 2

    def get_valid_answers(self, node_id):
        return self.colors

    def get_basic_long_prob_description(self):
        return (
            f"You will be requested to state which group you assign yourself to. There are exactly {self.num_colors} groups available: Group 1,...,Group {self.num_colors}. You should assign yourself to exactly one of these groups. "
            "The final result should be such that any two agents who are neighbors are in different groups. In particular, you should assign yourself to a group that is different from all of your neighbors' groups. "
        )

        # ---------- problem-specific insert (Coloring flow only here) ----------


class ColoringTentativeCommitCode(Coloring):
    def __init__(self, graph: nx.Graph, rounds: int, problem_level, **kwargs):
        super().__init__(graph=graph, rounds=rounds, problem_level=problem_level, **kwargs)
        # stable node order across the pipeline
        self._order = list(sorted(self.graph.nodes()))
        # Δ+1 default
        self.model_name = "python"
        if run_until_converge:
            self.rounds = 1000

    async def bootstrap(self):  # no-op for parity with your API
        return

    async def pass_messages(self):
        if self.problem_level == ProblemLevel.poly:
            answers = SolvePolyColoringTentativeCommitPython(
                self.graph, self.rounds, colors=self.num_colors
            ).answers
        else:
            answers = SolveNPColoringTentativeCommitPython(
                self.graph, self.rounds, colors=self.num_colors
            ).answers
        return answers


class ColoringTentativeCommitCompletePseudoCode(Coloring, CompletePseudoCode):
    """
    Emits a plain-language, agent-facing script for the priority-based
    group assignment (Δ+1 budget) protocol. Uses "neighbor" terminology.
    Adds a clear fallback when you have no option in P(me): choose the group
    with the fewest current clashes (ties broken by the shared priority order,
    then by the master list S).
    """

    def __init__(
            self,
            graph: nx.Graph,
            rounds: int,
            problem_level,
            model_name: str = "gpt-4o-mini",
            model_provider: str = "openai",
            chain_of_thought: bool = True,
            is_must_send: bool = True,
    ):
        Coloring.__init__(
            self,
            graph=graph,
            rounds=rounds,
            problem_level=problem_level,
            model_name=model_name,
            model_provider=model_provider,
            chain_of_thought=chain_of_thought,
            is_must_send=is_must_send,
        )
        CompletePseudoCode.__init__(self)

    # * Add exactly one backup to P(me): the least-frequent committed group (if tie, choose the first that appears in S). – If P(me) is now empty: * Count frequencies of neighbors’ committed groups. – If your list has become empty, look at what your neighbors are tentatively aiming for and pick the group with the fewest aiming at it.
    # * If several groups tie for “least contested,” keep the one that favors you by the shared priority order.
    # * If still tied, fall back to whichever comes first in the shared sequence S.

    # -------- NON-POLY (< Δ+1) TEXT --------
    def get_np_symbolic_pseudo_code(self) -> str:
        return f"""
STEP 0 — Initialize your own variables to keep track upon your decision and the decisions of your neighbors
How: init_variables
• committed(me): your final locked group (start “none”).
• tentative(me): your current try for this round (start “none”).
• P(me): your personal list of allowed groups (may be SMALLER than your neighbors; it can become empty).
• neighbors_latest: the most recent committed/tentative per neighbor.
• priority_order: fixed tie-break (names in lexicographic order; earlier name wins).

STEP 1 — Make an initial random tentative choice (if possible)
How: warm_start
• pick one random group from P(me) as your tentative for round 0.

STEP 2 — Share your status (every round)
How: share_status
• Send to all neighbors:
  – Committed = committed(me) or “none”.
  – Tentative = tentative(me) or “none”.

STEP 3 — Receive neighbors and prune options of groups neighbors are commited to (every round)
How: update_choices_from_neighbors
• Refresh neighbors_latest with new messages.
• Remove from P(me) any group a neighbor has committed to (keep the original order of what remains).


STEP 4 — Propose a tentative group (if not committed), and if not possible pick your tentative as the group that is LEAST FREQUENT among your neighbors’ COMMITTED
How: propose_group_or_least_frequent_fallback
• If committed(me) is set: leave tentative empty (no proposal).
• Else if P(me) is NOT empty: pick a random group from P(me) as your tentative.
• Else (P(me) IS empty): pick your tentative as the group that is LEAST FREQUENT among your neighbors’ COMMITTED groups:
  – Count how many neighbors have committed to each group.
  – Choose the group with the smallest count.
  – If several tie: pick the first that appears in S.
• Then: share_status (send your update) and update_choices_from_neighbors (ingest neighbors’ updates).

STEP 5 — Resolve clashes by priority and commit if you are first lexicographically within neighbors you clash with
How: resolve_conflicts_by_priority
• Look only at neighbors whose tentative matches yours this round.
• If your name is first lexicographically within that set (including you):
  – Commit now: set committed(me) = tentative(me); clear tentative(me).
• Otherwise:
  – Withdraw: clear tentative(me) and try again in the next proposal step (Step 4).

STEP 6 — Final round behavior
How: final_choice
• If it’s your last round and you’re still uncommitted, make your best distinct choice:
  – Prefer a group not seen in neighbors’ committed/tentative this round.
  – select the LEAST FREQUENT group among your neighbors
    """

    def get_poly_symbolic_psuedo_code(self):
        return """
STEP 0 — Initialize your own variables to keep track upon your decision and the decisions of your neighbors
How: init_variables
• committed(me): your final locked group (start “none”).
• tentative(me): your current try for this round (start “none”).
• P(me): your personal list of allowed groups (shrinks as neighbors commit).
• neighbors_latest: the most recent committed/tentative per neighbor.
• priority_order: fixed tie-break (names in lexicographic order; earlier name wins).

STEP 1 — Make an initial random tentative choice
How: warm_start
• Pick one random group from P(me) as your tentative group for round 0.
• Keep committed(me) = “none”.

STEP 2 — Share your status (every round)
How: share_status
• Send to all neighbors:
  – Committed = committed(me) or “none”.
  – Tentative = tentative(me) or “none”.

STEP 3 — Receive neighbors and prune options of groups neighbors are commited to (every round)
How: update_choices_from_neighbors
• Refresh neighbors_latest with new messages.
• Remove from P(me) any group a neighbor has committed to (keep the original order of what remains).

STEP 4 — Propose a tentative group (if not committed)
How: propose_group
• If committed(me) is set: leave tentative empty (no proposal).
• Otherwise: pick a random group from the current P(me) as your tentative.
• Then: share_status (send your update) and update_choices_from_neighbors (ingest neighbors’ updates).

STEP 5 — Resolve clashes by priority and commit if you are first lexicographically within neighbors you clash with
How: resolve_conflicts_by_priority
• Look only at neighbors whose tentative matches yours this round.
• If your name is first lexicographically within that set (including you):
  – Commit now: set committed(me) = tentative(me); clear tentative(me).
• Otherwise:
  – Withdraw: clear tentative(me) and try a different tentative in the next proposal step.

STEP 6 — Final round behavior
How: final_choice
• If it’s your last round and you’re still uncommitted, make your best distinct choice:
  – Prefer a group not seen in neighbors’ committed/tentative this round.
  – select the LEAST FREQUENT group among your neighbors
"""

    def get_symbolic_pseudo_code(self) -> str:
        if self.problem_level == ProblemLevel.poly:
            return self.get_poly_symbolic_psuedo_code()
        else:
            return self.get_np_symbolic_pseudo_code()


class ColoringTentativeCommitMainPseudocodeOnly(Coloring, MainPseudocodeOnly):
    def __init__(
            self,
            graph: nx.Graph,
            rounds: int,
            problem_level,
            model_name: str = "gpt-4o-mini",
            model_provider: str = "openai",
            chain_of_thought: bool = True,
            is_must_send: bool = True,
    ):
        Coloring.__init__(
            self,
            graph=graph,
            rounds=rounds,
            problem_level=problem_level,
            model_name=model_name,
            model_provider=model_provider,
            chain_of_thought=chain_of_thought,
            is_must_send=is_must_send,
        )
        MainPseudocodeOnly.__init__(self)

    def get_np_symbolic_pseudo_code(self) -> str:
        return f"""
    STEP 0 — Initialize your own variables to keep track upon your decision and the decisions of your neighbors

    STEP 1 — Make an initial random tentative choice (if possible)

    STEP 2 — Share your status (every round)

    STEP 3 — Receive neighbors and prune options of groups neighbors are commited to (every round)

    STEP 4 — Propose a tentative group (if not committed), and if not possible pick your tentative as the group that is LEAST FREQUENT among your neighbors’ COMMITTED

    STEP 5 — Resolve clashes by priority and commit if you are first lexicographically within neighbors you clash with, if you are not first go back step 4

    STEP 6 — In the final round if not commited yet select the LEAST FREQUENT group among your neighbors

        """

    def get_poly_symbolic_psuedo_code(self):
        return """
    STEP 0 — Initialize your own variables to keep track upon your decision and the decisions of your neighbors

    STEP 1 — Make an initial random tentative choice (if possible)

    STEP 2 — Share your status (every round)

    STEP 3 — Receive neighbors and prune options of groups neighbors are commited to (every round)

    STEP 4 — Propose a tentative group (if not committed)

    STEP 5 — Resolve clashes by priority and commit if you are first lexicographically within neighbors you clash with, if you are not first go back step 4

    STEP 6 — In the final round if not commited yet select the LEAST FREQUENT group among your neighbors
    """

    def get_symbolic_pseudo_code(self) -> str:
        if self.problem_level == ProblemLevel.poly:
            return self.get_poly_symbolic_psuedo_code()
        else:
            return self.get_np_symbolic_pseudo_code()


class ColoringTentativeCommitFunctionPseudoCodeOnly(Coloring, FunctionPseudoCodeOnly):
    def __init__(
            self,
            graph: nx.Graph,
            rounds: int,
            problem_level,
            model_name: str = "gpt-4o-mini",
            model_provider: str = "openai",
            chain_of_thought: bool = True,
            is_must_send: bool = True,
    ):
        Coloring.__init__(
            self,
            graph=graph,
            rounds=rounds,
            problem_level=problem_level,
            model_name=model_name,
            model_provider=model_provider,
            chain_of_thought=chain_of_thought,
            is_must_send=is_must_send,
        )

        FunctionPseudoCodeOnly.__init__(self)

    def get_np_symbolic_pseudo_code(self) -> str:
        return f"""
1) init_variables
    • committed(me): your final locked group (start “none”).
    • tentative(me): your current try for this round (start “none”).
    • P(me): your personal list of allowed groups (may be SMALLER than your neighbors; it can become empty).
    • neighbors_latest: the most recent committed/tentative per neighbor.
    • priority_order: fixed tie-break (names in lexicographic order; earlier name wins).

2) warm_start
    • pick one random group from P(me) as your tentative for round 0.

    STEP 2 — Share your status (every round)
    How: share_status
    • Send to all neighbors:
      – Committed = committed(me) or “none”.
      – Tentative = tentative(me) or “none”.

3) update_choices_from_neighbors
    • Refresh neighbors_latest with new messages.
    • Remove from P(me) any group a neighbor has committed to (keep the original order of what remains).


4) propose_group_or_least_frequent_fallback
    • If committed(me) is set: leave tentative empty (no proposal).
    • Else if P(me) is NOT empty: pick a random group from P(me) as your tentative.
    • Else (P(me) IS empty): pick your tentative as the group that is LEAST FREQUENT among your neighbors’ COMMITTED groups:
      – Count how many neighbors have committed to each group.
      – Choose the group with the smallest count.
      – If several tie: pick the first that appears in S.
    • Then: share_status (send your update) and update_choices_from_neighbors (ingest neighbors’ updates).

5) resolve_conflicts_by_priority
    • Look only at neighbors whose tentative matches yours this round.
    • If your name is first lexicographically within that set (including you):
      – Commit now: set committed(me) = tentative(me); clear tentative(me).
    • Otherwise:
      – Withdraw: clear tentative(me) and try again in the next proposal step (Step 4).

6) final_choice
    • If it’s your last round and you’re still uncommitted, make your best distinct choice:
      – Prefer a group not seen in neighbors’ committed/tentative this round.
      – select the LEAST FREQUENT group among your neighbors
        """

    def get_poly_symbolic_psuedo_code(self):
        return """
1) init_variables
    • committed(me): your final locked group (start “none”).
    • tentative(me): your current try for this round (start “none”).
    • P(me): your personal list of allowed groups (shrinks as neighbors commit).
    • neighbors_latest: the most recent committed/tentative per neighbor.
    • priority_order: fixed tie-break (names in lexicographic order; earlier name wins).

2) warm_start
    • Pick one random group from P(me) as your tentative group for round 0.
    • Keep committed(me) = “none”.

3)share_status
    • Send to all neighbors:
      – Committed = committed(me) or “none”.
      – Tentative = tentative(me) or “none”.

4) update_choices_from_neighbors
    • Refresh neighbors_latest with new messages.
    • Remove from P(me) any group a neighbor has committed to (keep the original order of what remains).

5) propose_group
    • If committed(me) is set: leave tentative empty (no proposal).
    • Otherwise: pick a random group from the current P(me) as your tentative.
    • Then: share_status (send your update) and update_choices_from_neighbors (ingest neighbors’ updates).

6) resolve_conflicts_by_priority
    • Look only at neighbors whose tentative matches yours this round.
    • If your name is first lexicographically within that set (including you):
      – Commit now: set committed(me) = tentative(me); clear tentative(me).
    • Otherwise:
      – Withdraw: clear tentative(me) and try a different tentative in the next proposal step.

7) final_choice
    • If it’s your last round and you’re still uncommitted, make your best distinct choice:
      – Prefer a group not seen in neighbors’ committed/tentative this round.
      – select the LEAST FREQUENT group among your neighbors
    """

    def get_symbolic_functions(self):
        if self.problem_level == ProblemLevel.poly:
            return self.get_poly_symbolic_psuedo_code()
        else:
            return self.get_np_symbolic_pseudo_code()


class ColoringTentativeCommitFunctionDescriptionOnly(Coloring, FunctionDescriptionOnly):
    def __init__(
            self,
            graph: nx.Graph,
            rounds: int,
            problem_level,
            model_name: str = "gpt-4o-mini",
            model_provider: str = "openai",
            chain_of_thought: bool = True,
            is_must_send: bool = True,
    ):
        Coloring.__init__(
            self,
            graph=graph,
            rounds=rounds,
            problem_level=problem_level,
            model_name=model_name,
            model_provider=model_provider,
            chain_of_thought=chain_of_thought,
            is_must_send=is_must_send,
        )
        FunctionDescriptionOnly.__init__(self)

    # -------- Non-Poly (k < Δ+1): function descriptions only --------
    def get_np_symbolic_pseudo_code(self) -> str:
        return """
1) init_variables
   • Sets up your personal state: your final choice (committed), your current try (tentative), your option list (P), 
     an inbox of neighbors' latest messages, and a fixed tie-break order for resolving clashes.

2) warm_start
   • If you have options in P, pick one at random as your initial tentative choice for round 0.
   • If you have no options at the start, leave tentative empty for now.

3) share_status
   • In every round, broadcast your current committed choice (or “none”) and your tentative choice (or “none”) 
     so neighbors can update their own options and detect clashes.

4) update_choices_from_neighbors
   What it does:
   • Refresh your view of neighbors' latest committed/tentative choices.
   • Remove from your option list P any groups already committed by neighbors (keep the original order for the rest).

5) propose_group_or_least_frequent_fallback
   • If you are already committed, do nothing this round.
   • If P is non-empty, pick a random option from P as your tentative choice.
   • If P is empty, choose your tentative as the group that appears LEAST OFTEN among neighbors’ COMMITTED choices.
     (If several tie, prefer the earliest in the shared sequence S.)

6) resolve_conflicts_by_priority
   • If some neighbors chose the same tentative as you, compare IDs by the shared lexicographic rule.
   • If you are first by that rule, you lock in (commit) your tentative; otherwise you clear your tentative and try again later.

7) final_choice
   • If this is the last round and you are still uncommitted, select the most distinct option you can:
     prefer a group not used in neighbors’ committed/tentative this round; if none, pick the least frequent among neighbors.
"""

    # -------- Poly (k ≥ Δ+1): function descriptions only --------
    def get_poly_symbolic_psuedo_code(self):
        return """
1) init_variables
   • Prepares your personal state: committed (final choice), tentative (current try), option list P (which shrinks as neighbors commit),
     neighbors' latest messages, and a fixed lexicographic tie-break order.

2) warm_start
   • Pick one random option from your current P as your initial tentative for round 0 (you are not committed yet).

3) share_status
   • In every round, announce your committed choice (or “none”) and your tentative choice (or “none”) so neighbors can update and check clashes.

4) update_choices_from_neighbors
   • Refresh your view of neighbors’ committed/tentative choices.
   • Remove from P any groups that neighbors have already committed to (preserve the order of remaining options).

5) propose_group
   • If you are already committed, you skip proposing this round.
   • Otherwise, pick a random option from your current P as your tentative and proceed.

6) resolve_conflicts_by_priority
   • If neighbors share your tentative, compare IDs by the shared lexicographic rule.
   • If you are first in that order, you commit to your tentative; otherwise, you clear your tentative and try a new one next time.

7) final_choice
   • If this is the last round and you remain uncommitted, pick the most distinct option:
     prefer a group not seen in neighbors’ committed/tentative; if none, pick the least frequent among neighbors.
"""

    def get_symbolic_functions(self):
        if self.problem_level == ProblemLevel.poly:
            return self.get_poly_symbolic_psuedo_code()
        else:
            return self.get_np_symbolic_pseudo_code()

class ColoringTentativeCommitHeuristicHintGuidedAgentV2(Coloring, HeuristicHintGuided):
    def __init__(
            self,
            graph: nx.Graph,
            rounds: int,
            problem_level,
            model_name: str = "gpt-4o-mini",
            model_provider: str = "openai",
            chain_of_thought: bool = True,
            is_must_send: bool = False,
    ):
        Coloring.__init__(
            self,
            graph=graph,
            rounds=rounds,
            problem_level=problem_level,
            model_name=model_name,
            model_provider=model_provider,
            chain_of_thought=chain_of_thought,
            is_must_send=is_must_send,
        )
        HeuristicHintGuided.__init__(self)

    def get_intro(self):
        return """
Hold a tentative choice each round, announce it to your neighbors, and update your options as others commit.
If multiple agents choose the same option, the one whose name comes first alphabetically (the higher priority) keeps it, while the others withdraw and try again.
If you run out of options or it’s the final round, fall back to the least-used option among your neighbors so the choices stay balanced.
"""


###########---------- Coloring complete BackTrack ----------###########
class ColoringBackTrackCode(Coloring):
    def __init__(
            self,
            graph: nx.Graph,
            rounds: int,
            problem_level,
            model_name: str = "gpt-4o-mini",
            model_provider: str = "openai",
            chain_of_thought: bool = True,
            is_must_send: bool = True,
    ):
        Coloring.__init__(
            self,
            graph=graph,
            rounds=rounds,
            problem_level=problem_level,
            model_name=model_name,
            model_provider=model_provider,
            chain_of_thought=chain_of_thought,
            is_must_send=is_must_send,
        )
        # keep a consistent node order across the pipeline
        self._order = list(sorted(self.graph.nodes()))
        # default upper bound on colors: Δ+1
        # nice-to-have bookkeeping
        self.model_name = "python"
        if run_until_converge:
            self.rounds = 1000

    async def bootstrap(self):  # no-op
        return

    async def pass_messages(self):
        answers = SolveColoringBacktrackPython(self.graph, self.rounds, self.num_colors).answers
        return answers


class ColoringBackTrackCompletePseudoCode(Coloring, CompletePseudoCode):
    def __init__(
            self,
            graph: nx.Graph,
            rounds: int,
            problem_level,
            model_name: str = "gpt-4o-mini",
            model_provider: str = "openai",
            chain_of_thought: bool = True,
            is_must_send: bool = True,
    ):
        Coloring.__init__(
            self,
            graph=graph,
            rounds=rounds,
            problem_level=problem_level,
            model_name=model_name,
            model_provider=model_provider,
            chain_of_thought=chain_of_thought,
            is_must_send=is_must_send,
        )
        CompletePseudoCode.__init__(self)

    def get_symbolic_pseudo_code(self):
        main_function = """
STEP 0 — Identify which neighbors have higher priority, which have lower priority, and who you should escalate to if you run out of options.  
How: establish_priority_and_parent  
• Sort all names alphabetically.  
• Mark neighbors before you as Higher, after you as Lower.  
• If Higher is not empty, set escalation_contact = the last in Higher; otherwise set escalation_contact = None.  

STEP 1 — Prepare your local state by starting with no chosen group, the full list of possible groups, a record of neighbors’ choices, and a log of what you’ve already tried.  
How: init_state  
• Set context = {} to store neighbor → group.  
• Set tried = {} to remember which (Higher context, group) you already used.  
• Set current_group = None.  
• Set palette = all available groups.  

STEP 2 — Decide if you move first or wait, then make your initial choice.  
How: choose_first_consistent_group  
• If Higher is empty: pick the smallest group not used, set current_group = it, and SEND VALUE to all Lower.  
• If Higher is not empty: wait until you receive VALUE from every Higher, update context, then pick the smallest non-conflicting group and SEND VALUE.  

STEP 3 — Handle the situation when no valid group is available by sending BACKTRACK to your escalation_contact.  
How: backtrack_if_needed  
• If no group is feasible, SEND BACKTRACK to escalation_contact.  
• If escalation_contact = None, announce FAILURE and stop.  

STEP 4 — Adjust your choice whenever a Higher neighbor changes theirs.  
How: on_value_from_higher  
• When VALUE(h, g_h) arrives, update context[h] = g_h.  
• If g_h == current_group, try the next unused feasible group.  
• If none exists, SEND BACKTRACK to escalation_contact (or FAIL if none).  

STEP 5 — Respond when a Lower neighbor reports they cannot proceed under your current choice and change your choice accordingly.  
How: on_backtrack_from_child  
• When BACKTRACK(l) arrives, try the next unused feasible group.  
• If you find one, set current_group = it and SEND VALUE to all Lower.  
• If not, SEND BACKTRACK to escalation_contact (or FAIL if none).  

STEP 6 — Finalize your decision when time runs out or the process must stop.  
How: finalize_now  
• If time runs out or you are forced to stop, choose in order:  
  – A group unused by any neighbor.  
  – Else a group unused by Higher.  
  – Else the smallest group in palette.  
• SEND VALUE(final_group) once to all Lower and then stop.  

        """

        return main_function


class ColoringBackTrackMainPseudocodeOnly(Coloring, MainPseudocodeOnly):
    def __init__(
            self,
            graph: nx.Graph,
            rounds: int,
            problem_level,
            model_name: str = "gpt-4o-mini",
            model_provider: str = "openai",
            chain_of_thought: bool = True,
            is_must_send: bool = True,
    ):
        Coloring.__init__(
            self,
            graph=graph,
            rounds=rounds,
            problem_level=problem_level,
            model_name=model_name,
            model_provider=model_provider,
            chain_of_thought=chain_of_thought,
            is_must_send=is_must_send,
        )
        MainPseudocodeOnly.__init__(self)

    def get_symbolic_pseudo_code(self):
        main_function = """
STEP 0 — Identify which neighbors have higher priority, which have lower priority, and who you should escalate to if you run out of options.  

STEP 1 — Prepare your local state by starting with no chosen group, the full list of possible groups, a record of neighbors’ choices, and a log of what you’ve already tried.  

STEP 2 — Decide if you move first or wait, then make your initial choice.  

STEP 3 — Handle the situation when no valid group is available by sending BACKTRACK to your escalation_contact.  

STEP 4 — Adjust your choice whenever a Higher neighbor changes theirs.  

STEP 5 — Respond when a Lower neighbor reports they cannot proceed under your current choice and change your choice accordingly.  

STEP 6 — Finalize your decision when time runs out or the process must stop.  

        """

        return main_function


class ColoringBackTrackFunctionPseudoCodeOnly(Coloring, FunctionPseudoCodeOnly):
    def __init__(
            self,
            graph: nx.Graph,
            rounds: int,
            problem_level,
            model_name: str = "gpt-4o-mini",
            model_provider: str = "openai",
            chain_of_thought: bool = True,
            is_must_send: bool = True,
    ):
        Coloring.__init__(
            self,
            graph=graph,
            rounds=rounds,
            problem_level=problem_level,
            model_name=model_name,
            model_provider=model_provider,
            chain_of_thought=chain_of_thought,
            is_must_send=is_must_send,
        )

        FunctionPseudoCodeOnly.__init__(self)

    def get_symbolic_functions(self):
        main_function = """
1) establish_priority_and_parent
• Sort all names alphabetically.
• Mark neighbors before you as Higher, after you as Lower.
• If Higher is not empty, set escalation_contact = the last in Higher; otherwise set escalation_contact = None.

2) init_state
• Set context = {} to store neighbor → group.
• Set tried = {} to remember which (Higher context, group) you already used.
• Set current_group = None.
• Set palette = all available groups.

3) choose_first_consistent_group
• If Higher is empty: pick the smallest group not used, set current_group = it, and SEND VALUE to all Lower.
• If Higher is not empty: wait until you receive VALUE from every Higher, update context, then pick the smallest non-conflicting group and SEND VALUE.

4) backtrack_if_needed
• If no group is feasible, SEND BACKTRACK to escalation_contact.
• If escalation_contact = None, announce FAILURE and stop.

5) on_value_from_higher
• When VALUE(h, g_h) arrives, update context[h] = g_h.
• If g_h == current_group, try the next unused feasible group.
• If none exists, SEND BACKTRACK to escalation_contact (or FAIL if none).

6) on_backtrack_from_child
• When BACKTRACK(l) arrives, try the next unused feasible group.
• If you find one, set current_group = it and SEND VALUE to all Lower.
• If not, SEND BACKTRACK to escalation_contact (or FAIL if none).

7) finalize_now
• If time runs out or you are forced to stop, choose in order:
  – A group unused by any neighbor.
  – Else a group unused by Higher.
  – Else the smallest group in palette.
• SEND VALUE(final_group) once to all Lower and then stop. """

        return main_function


class ColoringBackTrackFunctionDescriptionOnly(Coloring, FunctionDescriptionOnly):
    def __init__(
            self,
            graph: nx.Graph,
            rounds: int,
            problem_level,
            model_name: str = "gpt-4o-mini",
            model_provider: str = "openai",
            chain_of_thought: bool = True,
            is_must_send: bool = True,
    ):
        Coloring.__init__(
            self,
            graph=graph,
            rounds=rounds,
            problem_level=problem_level,
            model_name=model_name,
            model_provider=model_provider,
            chain_of_thought=chain_of_thought,
            is_must_send=is_must_send,
        )
        FunctionDescriptionOnly.__init__(self)

    # Only override the MODE block; keep Coloring flow in _intro_problem_unique_rules()

    def get_symbolic_functions(self):
        main_function = """
1) establish_priority_and_parent
   Figures out your place in the order by sorting all names, marking who is above (Higher) and below (Lower), and picking the one Higher neighbor you escalate to if needed.

2) init_state
   Prepares your local memory with an empty choice, a full list of groups, a record of neighbors’ choices, and a log of what you’ve already tried.

3) choose_first_consistent_group
   Makes your initial choice by picking the smallest group not used by Higher neighbors, waiting for their announcements first if necessary, and then sending it to those below.

4) backtrack_if_needed
   If no valid group is available under the current situation, you ask your escalation contact to change; if none exists, you declare failure and stop.

5) on_value_from_higher
   Updates what you know when a Higher neighbor changes, and if their choice clashes with yours you either switch to a new group or escalate if no option remains.

6) on_backtrack_from_child
   Handles a Lower neighbor’s backtrack by trying a different group yourself, and if you cannot find one you pass the backtrack upward.

7) finalize_now
   When the process must stop, you settle on the best available group—prefer one unused by any neighbor, otherwise unused by Higher, otherwise the smallest—and announce it as final.
"""
        return main_function


class ColoringBackTrackHeuristicHintGuidedAgentV2(Coloring, HeuristicHintGuided):
    def __init__(
            self,
            graph: nx.Graph,
            rounds: int,
            problem_level,
            model_name: str = "gpt-4o-mini",
            model_provider: str = "openai",
            chain_of_thought: bool = True,
            is_must_send: bool = True,
    ):
        Coloring.__init__(
            self,
            graph=graph,
            rounds=rounds,
            problem_level=problem_level,
            model_name=model_name,
            model_provider=model_provider,
            chain_of_thought=chain_of_thought,
            is_must_send=is_must_send,
        )
        HeuristicHintGuided.__init__(self)

    def get_intro(self):
        return """
Choose the smallest color that does not clash with your higher-priority neighbors—these are the ones whose names come before yours alphabetically—then announce your choice forward to your lower-priority neighbors.
If you later run into a conflict or if one of your lower neighbors signals they are stuck, you adjust your choice; if no valid option remains, you send a backward request to your escalation contact (a higher neighbor) asking them to reconsider.
"""

##########################
