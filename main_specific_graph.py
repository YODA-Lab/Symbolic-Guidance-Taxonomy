from langchain.agents import Agent

import LiteralMessagePassing as lmp
import argparse
import asyncio
import datetime
import os
import json
import random
import networkx as nx
from networkx.readwrite import json_graph
import subprocess

from globals_ import AgentType, AlgoType, Problems

# -------------------------
# Task registry
# -------------------------
TASKS = {
    AlgoType.algo_independent:{"coloring":{AgentType.FreeForm: lmp.Coloring},
                        "leader_election":{AgentType.FreeForm: lmp.LeaderElection},
                        "consensus":{AgentType.FreeForm: lmp.Consensus},
                        "matching":{AgentType.FreeForm: lmp.Matching}

},
    AlgoType.complete:{"coloring": {
        AgentType.Code:lmp.ColoringPriorityGreedyCode,
        AgentType.CompletePseudoCode: lmp.ColoringPriorityGreedyCompletePseudoCode,
        AgentType.MainPseudocodeOnly: lmp.ColoringPriorityGreedyMainPseudocodeOnly,
        AgentType.FunctionPseudoCodeOnly: lmp.ColoringPriorityGreedyFunctionPseudoCodeOnly,
        AgentType.FunctionDescriptionOnly: lmp.ColoringPriorityGreedyFunctionDescriptionOnly,
        AgentType.HeuristicHintGuided: lmp.ColoringBackTrackHeuristicHintGuidedAgentV2,
    },"matching": {
        AgentType.Code:lmp.MatchingEdmondsCode,
        AgentType.CompletePseudoCode: lmp.MatchingEdmondsCompletePseudoCode,
        AgentType.MainPseudocodeOnly: lmp.MatchingEdmondsMainPseudocodeOnly,
        AgentType.FunctionPseudoCodeOnly: lmp.MatchingEdmondsFunctionPseudoCodeOnly,
        AgentType.FunctionDescriptionOnly: lmp.MatchingEdmondsFunctionDescriptionOnly,
        AgentType.HeuristicHintGuided: lmp.MatchingEdmondsHintGuidedAgent,
    }
    },
    AlgoType.incomplete:{"coloring": {
        AgentType.Code: lmp.ColoringDsaCode,
        AgentType.CompletePseudoCode: lmp.ColoringDsaCompletePseudoCode,
        AgentType.MainPseudocodeOnly: lmp.ColoringDsaMainPseudocodeOnly,
        AgentType.FunctionPseudoCodeOnly: lmp.ColoringDsaFunctionPseudoCodeOnly,
        AgentType.FunctionDescriptionOnly: lmp.ColoringDsaFunctionDescriptionOnly,
        AgentType.HeuristicHintGuided: lmp.ColoringDsaHeuristicHintGuidedAgentV2,
    }, "matching":{
        AgentType.Code:lmp.MatchingGreedyMaximalCode,
        AgentType.CompletePseudoCode: lmp.MatchingGreedyMaximalCompletePseudoCode,
        AgentType.MainPseudocodeOnly: lmp.MatchingGreedyMaximalMainPseudocodeOnly,
        AgentType.FunctionPseudoCodeOnly: lmp.MatchingGreedyMaximalFunctionPseudoCodeOnly,
        AgentType.FunctionDescriptionOnly: lmp.MatchingGreedyMaximalFunctionDescriptionOnly,
        AgentType.HeuristicHintGuided: lmp.MatchingGreedyMaximalHeuristicHintGuidedAgentV2,
    }

    }
}

# -------------------------
# Model provider mapping
# -------------------------
MODEL_PROVIDER = {
    "gpt-4o-mini": "openai",
    "gpt-4.1-mini": "openai",
    "gpt-4o": "openai",
    "o1": "openai",
    "o3-mini": "openai",
    "o4-mini": "openai",
    "llama3": "ollama",
    "gemini-2.0-flash": "google-genai",
    "gemini-2.0-flash-lite": "google-genai",
    "claude-3-5-haiku-20241022": "anthropic",
    "claude-3-opus-20240229": "anthropic",
    "claude-3-7-sonnet-20250219": "anthropic",
    "claude-3-7-sonnet-20250219-thinking": "anthropic",
    "gemini-2.5-flash-preview-04-17": "google-genai",
    "gemini-2.5-flash-preview-04-17-thinking": "google-genai",
    "gemini-2.5-pro-exp-03-25": "google-genai",
    "gemini-2.5-pro-preview-03-25": "google-genai",
    "gemini-2.5-pro-preview-05-06": "google-genai",
    "gemini-1.5-pro": "google-genai",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "together",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": "together",
    "meta-llama/llama-3.1-8b-instruct:free": "openai",
    "google/gemini-2.5-flash": "openai",   # <-- add this line
    # OpenRouter is OpenAI-compatible

}

# -------------------------
# Utilities
# -------------------------
def get_git_commit_hash():
    """Failsafe even if git is not installed."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "None"

def _safe_diameter(G: nx.Graph) -> int:
    """Diameter; if disconnected, use the largest connected component."""
    if G.number_of_nodes() == 0:
        return 0
    if nx.is_connected(G):
        return nx.diameter(G)
    cc = max(nx.connected_components(G), key=len)
    return nx.diameter(G.subgraph(cc).copy())

def get_graph_path(local_dir: str, graph_model: str, graph_size: int, num_sample: int) -> str:
    return os.path.join(local_dir, f"graph_{graph_model}_{graph_size}_{num_sample}.json")

def get_graph(graph_model: str, graph_size: int, num_sample: int, local_dir: str) -> nx.Graph:
    """Load a pre-generated local graph JSON: {'graph': <node-link>, ...}."""
    filepath = get_graph_path(local_dir, graph_model, graph_size, num_sample)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Local graph not found: {filepath}")
    print(f"[INFO] Loading local graph JSON: {filepath}")
    with open(filepath, "r") as f:
        graph_dict = json.load(f)

    print(
        f"[INFO] Loaded {graph_model}_{graph_size}_{num_sample} "
        f"({graph_dict.get('num_nodes', 'unknown')} nodes) from {filepath}"
    )
    return json_graph.node_link_graph(graph_dict["graph"])

def determine_rounds(task: str, graph: nx.Graph) -> int:
    """Always use 2*D + 1 rounds, where D is (safe) graph diameter."""
    D = _safe_diameter(graph)
    rounds = 2 * D + 1
    # Optional: never drop below 1 round
    return max(1, rounds)

def save_results(
    answers,
    transcripts,
    graph: nx.Graph,
    rounds: int,
    model_name: str,
    task: str,
    score,
    commit_hash: str,
    graph_generator: str,
    graph_index: int,
    successful: bool,
    error_message: str | None,
    chain_of_thought: bool,
    num_fallbacks: int,
    num_failed_json_parsings_after_retry: int,
    num_failed_answer_parsings_after_retry: int,
    output_folder: str,
):
    """
    Save one run's outputs under results/<output_folder>/...
    output_folder is '<prob_type>,auto,<agent_type>' when using diameter-based rounds.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = (
        f"{task}_results_{timestamp}_g{graph_index}_{graph_generator}_"
        f"rounds{rounds}_{model_name.split('/')[-1]}_nodes{graph.number_of_nodes()}.json"
    )
    output_dir = os.path.join("results", output_folder)
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(
            {
                "answers": answers,
                "transcripts": transcripts,
                "graph": json_graph.node_link_data(graph),
                "num_nodes": graph.number_of_nodes(),
                "diameter": _safe_diameter(graph),
                "max_degree": max(dict(graph.degree()).values(), default=0),
                "rounds": rounds,
                "model_name": model_name,
                "task": task,
                "score": score,
                "commit_hash": commit_hash,
                "graph_generator": graph_generator,
                "graph_index": graph_index,
                "successful": successful,
                "error_message": error_message,
                "chain_of_thought": chain_of_thought,
                "num_fallbacks": num_fallbacks,
                "num_failed_json_parsings_after_retry": num_failed_json_parsings_after_retry,
                "num_failed_answer_parsings_after_retry": num_failed_answer_parsings_after_retry,
            },
            f,
            indent=4,
        )

    print(f"[INFO] Results saved → {filepath}")

# -------------------------
# Main async runner
# -------------------------
async def run(args):
    commit_hash = get_git_commit_hash()
    random.seed(args.seed)

    # Subfolder indicates automatic rounds (2D+1)
    output_folder_name = f"{str(args.graph_size)},{args.algo_type.name},{args.task},{args.symbolic_level.name}"
    print(f"[INFO] Writing results under: results/{output_folder_name}")

    if args.missing_run_file is not None:
        recovery_mode = True
        import pandas as pd  # lazy import
        missing_run_df = pd.read_csv(args.missing_run_file)
    else:
        recovery_mode = False

    for graph_model in args.graph_models:
        for i in range(args.start_from_sample, args.start_from_sample+args.samples_per_graph_model):
            graph = get_graph(graph_model, args.graph_size, i, args.local_graphs_dir)
            rounds = determine_rounds(args.task, graph)
            print(f"[INFO] Selected {rounds} rounds for {graph_model}_{args.graph_size}_{i}")

            os.environ["OLLAMA_URI"] = "http://localhost:11434"
            task_class = TASKS[args.algo_type][args.task][args.symbolic_level]
            model_provider = MODEL_PROVIDER[args.model]
            chain_of_thought = not args.disable_chain_of_thought

            runs_to_execute = 1
            if recovery_mode:
                print("### Entered recovery mode")
                graph_string = str(json_graph.node_link_data(graph))
                filtered_df = missing_run_df[
                    (missing_run_df.num_nodes == graph.number_of_nodes())
                    & (missing_run_df.task == args.task)
                    & (missing_run_df.graph_generator == graph_model)
                    & (missing_run_df.graph == graph_string)
                    & (missing_run_df.model_name == args.model)
                ]
                if len(filtered_df) > 0:
                    print("### Running recovery!")
                    runs_to_execute = int(filtered_df.iloc[0].missing_runs)
                else:
                    print("### Skipping run!")
                    runs_to_execute = 0

            for _ in range(runs_to_execute):
                lmp_model: lmp.LiteralMessagePassing = task_class(
                    graph=graph,
                    rounds=rounds,
                    model_name=args.model,
                    model_provider=model_provider,
                    chain_of_thought=chain_of_thought,
                )

                for attempt in range(max_retries):
                    await lmp_model.bootstrap()
                    try:
                        answers = await lmp_model.pass_messages()

                        # Check if any value in answers is None
                        if isinstance(answers, list) and any(v is None for v in answers):
                            print(f"[WARN] Answers contained None values (attempt {attempt + 1}/{max_retries})")
                            if attempt < max_retries - 1:
                                continue  # retry bootstrap + pass_messages
                            else:
                                raise ValueError("Answers still contained None values after max retries")

                        score = lmp_model.get_score(answers)
                        successful = True
                        error_message = None
                        break  # success, exit retry loop

                    except (ValueError, KeyError) as e:
                        print(f"[ERROR] Failed run (attempt {attempt + 1}/{max_retries}): {repr(e)}")
                        if attempt < max_retries - 1:
                            continue  # try again
                        answers = {k: None for k in range(graph.order())} if isinstance(answers, dict) else [None for _
                                                                                                             in range(
                                graph.order())]
                        score = None
                        successful = False
                        error_message = repr(e)

                save_results(
                    answers=answers,
                    transcripts=lmp_model.get_transcripts(),
                    graph=lmp_model.graph,
                    rounds=rounds,
                    model_name=lmp_model.model_name,
                    task=args.task,
                    score=score,
                    commit_hash=commit_hash,
                    graph_generator=graph_model,
                    graph_index=i,
                    successful=successful,
                    error_message=error_message,
                    chain_of_thought=chain_of_thought,
                    num_fallbacks=lmp_model.num_fallbacks,
                    num_failed_json_parsings_after_retry=lmp_model.num_failed_json_parsings_after_retry,
                    num_failed_answer_parsings_after_retry=lmp_model.num_failed_answer_parsings_after_retry,
                    output_folder=output_folder_name,
                )


def get_dict_for_main():
    # return {
    #         AlgoType.algo_independent: {Problems.coloring: [AgentType.FreeForm],
    #                                     Problems.leader_election: [AgentType.FreeForm],
    #                                     Problems.consensus: [AgentType.FreeForm],
    #                                     Problems.matching: [AgentType.FreeForm]},
    #
    #         AlgoType.incomplete: {Problems.coloring: [AgentType.Code,AgentType.CompletePseudoCode, AgentType.MainPseudocodeOnly, AgentType.FunctionPseudoCodeOnly, AgentType.FunctionDescriptionOnly, AgentType.HeuristicHintGuided],
    #                               Problems.leader_election: [AgentType.Code,AgentType.CompletePseudoCode, AgentType.MainPseudocodeOnly, AgentType.FunctionPseudoCodeOnly, AgentType.FunctionDescriptionOnly, AgentType.HeuristicHintGuided],
    #                               Problems.consensus: [AgentType.Code,AgentType.CompletePseudoCode, AgentType.MainPseudocodeOnly, AgentType.FunctionPseudoCodeOnly, AgentType.FunctionDescriptionOnly, AgentType.HeuristicHintGuided],
    #                               Problems.matching: [AgentType.Code,AgentType.CompletePseudoCode, AgentType.MainPseudocodeOnly, AgentType.FunctionPseudoCodeOnly, AgentType.FunctionDescriptionOnly, AgentType.HeuristicHintGuided]},
    #
    #
    #         AlgoType.complete:  {Problems.coloring: [AgentType.Code,AgentType.CompletePseudoCode, AgentType.MainPseudocodeOnly, AgentType.FunctionPseudoCodeOnly, AgentType.FunctionDescriptionOnly, AgentType.HeuristicHintGuided],
    #                              Problems.leader_election: [AgentType.Code,AgentType.CompletePseudoCode, AgentType.MainPseudocodeOnly, AgentType.FunctionPseudoCodeOnly, AgentType.FunctionDescriptionOnly, AgentType.HeuristicHintGuided],
    #                              Problems.consensus: [AgentType.Code,AgentType.CompletePseudoCode, AgentType.MainPseudocodeOnly, AgentType.FunctionPseudoCodeOnly, AgentType.FunctionDescriptionOnly, AgentType.HeuristicHintGuided],
    #                              Problems.matching: [AgentType.Code,AgentType.CompletePseudoCode, AgentType.MainPseudocodeOnly, AgentType.FunctionPseudoCodeOnly, AgentType.FunctionDescriptionOnly, AgentType.HeuristicHintGuided]}
    # }

    #return

    #return    {AlgoType.incomplete: {"matching": [ AgentType.Code,AgentType.MainPseudocodeOnly,AgentType.FunctionPseudoCodeOnly, AgentType.FunctionDescriptionOnly, AgentType.HeuristicHintGuided]}}

    return {
            #AlgoType.algo_independent: {
             #                           "coloring": [AgentType.FreeForm]},

            AlgoType.incomplete: {
                                  "coloring": [AgentType.CompletePseudoCode]}#[AgentType.MainPseudocodeOnly, AgentType.FunctionPseudoCodeOnly]}#, AgentType.FunctionDescriptionOnly, AgentType.HeuristicHintGuided]}
    }
# -------------------------
# CLI entry
# -------------------------
if __name__ == "__main__":
    max_retries = 3  # how many times to retry if answers contain None

    dict1 = get_dict_for_main()# Configure the loops you want to run here
    for graph_s in [8]:

        for algo_type,dict2 in dict1.items():
            for prob,list_of_types_of_agents in dict2.items():
                for agent_type in list_of_types_of_agents:
        #for prob_type in ["consensus"]:#["matching", "coloring"]:
            #for agent_type in AgentType.list_all():
                    parser = argparse.ArgumentParser()
                    parser.add_argument("--model", type=str, default="google/gemini-2.5-flash")

                    parser.add_argument("--task", type=str, default=prob)
                    parser.add_argument("--algo_type", type=AgentType, default=algo_type)

                    parser.add_argument("--graph_models", type=str, nargs="+", default=[ "dt"])#default=["ws", "ba", "dt"])
                    parser.add_argument("--start_from_sample", type=int, default=1)#default=0)
                    parser.add_argument("--samples_per_graph_model", type=int, default=2)
                    parser.add_argument("--graph_size", type=int, default=graph_s)
                    # NOTE: No --rounds argument; we compute 2D+1 per-graph automatically.
                    parser.add_argument("--seed", type=int, default=42)
                    parser.add_argument("--disable_chain_of_thought", action="store_true")
                    parser.add_argument("--missing_run_file", type=str, default=None)
                    parser.add_argument("--symbolic_level", type=AgentType, default=agent_type)
                    parser.add_argument("--local_graphs_dir", type=str, default="graphs")
                    args = parser.parse_args()
                    asyncio.run(run(args))