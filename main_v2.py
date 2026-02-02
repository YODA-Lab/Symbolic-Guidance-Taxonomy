# -*- coding: utf-8 -*-
# Fixed runner: problem_level from get_dict_for_main → iterated in __main__ → passed to LMP constructors
# FreeForm remap rules:
#   - poly:    run as AlgoType.complete only
#   - hard:    run as BOTH AlgoType.complete and AlgoType.incomplete
from langchain.agents import Agent

# Removed conflicting/unneeded Agent imports:
# from huggingface_hub import Agent
# from langchain.agents import Agent

import LiteralMessagePassing as lmp
import argparse
import asyncio
import datetime
import os
import json
import random
import re
import subprocess
import networkx as nx
from networkx.readwrite import json_graph

from globals_ import AgentType, AlgoType, Problems, ProblemLevel

# -------------------------
# Task registry (kept minimal; FreeForm lives under algo_independent)
# -------------------------

from dotenv import load_dotenv
load_dotenv(override=True)

TASKS = {
    "coloring": {
        ProblemLevel.poly: {
            # FreeForm defined here under algo_independent
            AlgoType.algo_independent: {AgentType.FreeForm: lmp.Coloring},
            AlgoType.complete:{
                AgentType.Code:lmp.ColoringTentativeCommitCode,
                AgentType.CompletePseudoCode: lmp.ColoringTentativeCommitCompletePseudoCode,
                AgentType.MainPseudocodeOnly: lmp.ColoringTentativeCommitMainPseudocodeOnly,
                AgentType.FunctionPseudoCodeOnly: lmp.ColoringTentativeCommitFunctionPseudoCodeOnly,
                AgentType.FunctionDescriptionOnly: lmp.ColoringTentativeCommitFunctionDescriptionOnly,
                AgentType.HeuristicHintGuided: lmp.ColoringTentativeCommitHeuristicHintGuidedAgentV2}
                },
        ProblemLevel.hard: {
            # you can add explicit algo-specific agents here if you like
            AlgoType.algo_independent: {AgentType.FreeForm: lmp.Coloring},
            AlgoType.incomplete:{
                AgentType.Code:lmp.ColoringTentativeCommitCode,
                AgentType.CompletePseudoCode: lmp.ColoringTentativeCommitCompletePseudoCode,
                AgentType.MainPseudocodeOnly: lmp.ColoringTentativeCommitMainPseudocodeOnly,
                AgentType.FunctionPseudoCodeOnly: lmp.ColoringTentativeCommitFunctionPseudoCodeOnly,
                AgentType.FunctionDescriptionOnly: lmp.ColoringTentativeCommitFunctionDescriptionOnly,
                AgentType.HeuristicHintGuided: lmp.ColoringTentativeCommitHeuristicHintGuidedAgentV2}
                ,
                AlgoType.complete:{
                AgentType.Code:lmp.ColoringBackTrackCode,
                AgentType.CompletePseudoCode: lmp.ColoringBackTrackCompletePseudoCode,
                AgentType.MainPseudocodeOnly: lmp.ColoringBackTrackMainPseudocodeOnly,
                AgentType.FunctionPseudoCodeOnly: lmp.ColoringBackTrackFunctionPseudoCodeOnly,
                AgentType.FunctionDescriptionOnly: lmp.ColoringBackTrackFunctionDescriptionOnly,
                AgentType.HeuristicHintGuided: lmp.ColoringBackTrackHeuristicHintGuidedAgentV2}
                }
            },
    "matching":{

                ProblemLevel.hard: {AlgoType.algo_independent: {AgentType.FreeForm: lmp.Matching},
                AlgoType.incomplete:{AgentType.Code:lmp.MatchingNaiveCode,
                AgentType.CompletePseudoCode: lmp.MatchingNaiveCompletePseudoCode,
                AgentType.MainPseudocodeOnly: lmp.MatchingNaiveMainPseudocodeOnly,
                AgentType.FunctionPseudoCodeOnly: lmp.MatchingNaiveFunctionPseudoCodeOnly,
                AgentType.FunctionDescriptionOnly: lmp.MatchingNaiveFunctionDescriptionOnly,
                AgentType.HeuristicHintGuided: lmp.MatchingNaiveHeuristicHintGuidedAgentV2},
                AlgoType.complete:{

                AgentType.Code:lmp.MatchingBackTrackCode,
                AgentType.CompletePseudoCode: lmp.MatchingBackTrackCompletePseudoCodeV2,
                AgentType.MainPseudocodeOnly: lmp.MatchingBackTrackMainPseudocodeOnly,
                AgentType.FunctionPseudoCodeOnly: lmp.MatchingBackTrackFunctionPseudoCodeOnlyV2,
                AgentType.FunctionDescriptionOnly: lmp.MatchingBackTracFunctionDescriptionOnly,
                AgentType.HeuristicHintGuided: lmp.MatchingBackTrackHeuristicHintGuidedAgentV2}


                                    }},
                "vertex_cover": {

                ProblemLevel.hard: {AlgoType.algo_independent: {AgentType.FreeForm: lmp.VertexCover},
                                    AlgoType.incomplete: {AgentType.Code: lmp.VertexCoverPriorityCode,
                                                          AgentType.CompletePseudoCode: lmp.VertexCoverNaiveCompletePseudoCode,
                                                          AgentType.MainPseudocodeOnly: lmp.VertexCoverNaiveMainPseudocodeOnly,
                                                          AgentType.FunctionPseudoCodeOnly: lmp.VertexCoverNaiveFunctionPseudoCodeOnly,
                                                          AgentType.FunctionDescriptionOnly: lmp.VertexCoverNaiveFunctionDescriptionOnly,
                                                          AgentType.HeuristicHintGuided: lmp.VertexCoverNaiveHeuristicHintGuidedAgentV2},
                                    AlgoType.complete: {AgentType.Code: lmp.VertexCoverTrackCode,
                                                          AgentType.CompletePseudoCode: lmp.VertexCoverBackTrackCompletePseudoCode,
                                                          AgentType.MainPseudocodeOnly: lmp.VertexCoverBackTrackMainPseudocodeOnly,
                                                          AgentType.FunctionPseudoCodeOnly: lmp.VertexCoverBackTrackFunctionPseudoCodeOnly,
                                                          AgentType.FunctionDescriptionOnly: lmp.VertexCoverBackTrackFunctionDescriptionOnly,
                                                          AgentType.HeuristicHintGuided: lmp.VertexCoverBackTrackHeuristicHintGuided}
                            }}



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

    #### This is what i used:
    "google/gemini-2.5-flash": "openai",
    "deepseek/deepseek-r1": "openai",
    "deepseek/deepseek-r1:free": "openai",  # optional

    "qwen2.5:7b-instruct": "ollama",
    "llama3": "ollama",


}

# -------------------------
# Utilities
# -------------------------
def safe_fs_label(s: str, maxlen: int = 80) -> str:
    s = str(s or "unknown")
    s = s.split("/")[-1]                 # drop any path-like prefix
    s = re.sub(r'[\\/:*?"<>|]', "-", s)  # replace Windows-reserved chars
    s = re.sub(r"\s+", "_", s)           # normalize whitespace
    return s[:maxlen]

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
    return max(1, rounds)

def resolve_task_class(TASKS, task: str, problem_level, algo_type, symbolic_level):
    """
    Returns the LMP class for (task, problem_level, algo_type, symbolic_level),
    supporting both shapes under TASKS[task][problem_level]:
      - {AlgoType: {AgentType: Class}}
      - {AgentType: Class}  (implicit AlgoType.algo_independent)
    Also: if requested algo_type is complete/incomplete but only algo_independent exists,
    we map to that implementation (useful for FreeForm remapping).
    """
    if task not in TASKS:
        raise KeyError(f"Unknown task '{task}' in TASKS.")
    if problem_level not in TASKS[task]:
        raise KeyError(f"Task '{task}' has no problem_level '{problem_level}'.")
    node = TASKS[task][problem_level]

    # Case 1: explicit AlgoType dict present
    if isinstance(node, dict) and any(isinstance(k, AlgoType) for k in node.keys()):
        # If the requested algo_type exists, use it
        if algo_type in node:
            sub = node[algo_type]
            if symbolic_level not in sub:
                raise KeyError(
                    f"No class for task='{task}', level='{problem_level}', "
                    f"algo_type='{algo_type}', symbolic='{symbolic_level}'."
                )
            return sub[symbolic_level]
        # Otherwise, if algo_independent exists, fall back to it
        if AlgoType.algo_independent in node:
            sub = node[AlgoType.algo_independent]
            if symbolic_level in sub:
                return sub[symbolic_level]
        # Else: not found
        raise KeyError(
            f"No class for task='{task}', level='{problem_level}', algo_type='{algo_type}', "
            f"symbolic='{symbolic_level}' (and no algo_independent fallback)."
        )

    # Case 2: flat {AgentType: Class} dict (treat as algo_independent)
    if symbolic_level not in node:
        raise KeyError(
            f"No class for task='{task}', level='{problem_level}', "
            f"(flat node) symbolic='{symbolic_level}'."
        )
    return node[symbolic_level]

def save_results(
    answers,
    transcripts,
    graph: nx.Graph,
    rounds: int,
    model_name: str,
    task: str,
    score,
    score_opt,
    SSE_easy,
    SSE_hard,
    norm_SSE_hard,
    norm_SSE_easy,

    required_calculations,

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
    base_results_dir: str = "results",
    algorithm_type_name: str | None = None,
    agent_type_name: str | None = None,
    problem_level_name: str | None = None,
):
    """
    Save one run's outputs under {base_results_dir}/{output_folder}/...
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    raw_model_label = (model_name or "").split("/")[-1]
    model_label = safe_fs_label(raw_model_label)

    filename = (
        f"{task}_results_{timestamp}_g{graph_index}_{graph_generator}_"
        f"rounds{rounds}_{model_label}_nodes{graph.number_of_nodes()}.json"
    )

    output_dir = os.path.join(base_results_dir, output_folder)
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
                "score_opt":score_opt,
                "SSE_easy":SSE_easy,
                "SSE_hard":SSE_hard,
                "norm_SSE_easy": norm_SSE_easy,
                "norm_SSE_hard": norm_SSE_hard,
                "required_calculations":required_calculations,
                "commit_hash": commit_hash,
                "graph_generator": graph_generator,
                "graph_index": graph_index,
                "successful": successful,
                "error_message": error_message,
                "chain_of_thought": chain_of_thought,
                "num_fallbacks": num_fallbacks,
                "num_failed_json_parsings_after_retry": num_failed_json_parsings_after_retry,
                "num_failed_answer_parsings_after_retry": num_failed_answer_parsings_after_retry,
                "algorithm_type": algorithm_type_name,
                "agent_type": agent_type_name,
                "problem_level": problem_level_name,
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

    # Folder name includes problem level
    output_folder_name = (
        f"{str(args.graph_size)},"
        f"{args.algo_type.name},"
        f"{args.task},"
        f"{args.problem_level.name},"
        f"{args.symbolic_level.name}"
    )

    base_results_dir = "results" if args.start_from_sample > 2 else "results"
    print(f"[INFO] Writing results under: {base_results_dir}/{output_folder_name}")

    if args.missing_run_file is not None:
        recovery_mode = True
        import pandas as pd  # lazy import
        missing_run_df = pd.read_csv(args.missing_run_file)
    else:
        recovery_mode = False

    for current_model in args.models:
        if current_model not in MODEL_PROVIDER:
            print(f"[WARN] Model '{current_model}' not found in MODEL_PROVIDER. Skipping.")
            continue

        model_provider = MODEL_PROVIDER[current_model]
        print(f"[INFO] Running model: {current_model} (provider: {model_provider})")

        for graph_model in args.graph_models:
            for i in range(args.start_from_sample, args.start_from_sample + args.samples_per_graph_model):
                graph = get_graph(graph_model, args.graph_size, i, args.local_graphs_dir)
                rounds = determine_rounds(args.task, graph)
                print(f"[INFO] Selected {rounds} rounds for {graph_model}_{args.graph_size}_{i}")

                os.environ["OLLAMA_URI"] = "http://localhost:11434"

                # Resolve correct class for (task, problem_level, algo_type, agent_type)
                task_class = resolve_task_class(
                    TASKS,
                    task=args.task,
                    problem_level=args.problem_level,
                    algo_type=args.algo_type,
                    symbolic_level=args.symbolic_level,
                )

                # If flag is present, disable CoT; here we keep it False by default (your prior change)
                chain_of_thought = False  # or: not args.disable_chain_of_thought

                runs_to_execute = 1
                if recovery_mode:
                    print("### Entered recovery mode")
                    graph_string = str(json_graph.node_link_data(graph))
                    filtered_df = missing_run_df[
                        (missing_run_df.num_nodes == graph.number_of_nodes())
                        & (missing_run_df.task == args.task)
                        & (missing_run_df.graph_generator == graph_model)
                        & (missing_run_df.graph == graph_string)
                        & (missing_run_df.model_name == current_model)
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

                        problem_level=args.problem_level,     # ← pass level
                        model_name=current_model,
                        model_provider=model_provider,
                        chain_of_thought=chain_of_thought,
                    )

                    for attempt in range(max_retries):
                        await lmp_model.bootstrap()

                        answers = await lmp_model.pass_messages()

                        # Validate answers

                        score = lmp_model.get_score(answers)
                        score_opt = lmp_model.get_score_opt(answers)
                        SSE_hard = lmp_model.get_SSE_hard(answers)
                        SSE_easy = lmp_model.get_SSE_easy(answers)
                        norm_SSE_easy = lmp_model.get_norm_SSE_easy(answers)
                        norm_SSE_hard = lmp_model.get_norm_SSE_hard(answers)
                        required_calculations = lmp_model.get_required_calculations(answers)
                        successful = True
                        error_message = None



                    if score is not None:
                        save_results(
                            answers=answers,
                            transcripts=lmp_model.get_transcripts(),
                            graph=lmp_model.graph,
                            rounds=rounds,
                            model_name=lmp_model.model_name,
                            task=args.task,
                            score=score,
                            score_opt = score_opt,
                            SSE_hard = SSE_hard,
                            SSE_easy = SSE_easy,
                            norm_SSE_easy = norm_SSE_easy,
                            norm_SSE_hard=norm_SSE_hard,

                            required_calculations =required_calculations,
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
                            base_results_dir=base_results_dir,
                            algorithm_type_name=args.algo_type.name,
                            agent_type_name=args.symbolic_level.name,
                            problem_level_name=args.problem_level.name,
                        )
                    else:
                        t = (
                            args.problem_level.name
                            + ","
                            + args.symbolic_level.name
                            + ","
                            + graph_model
                            + ","
                            + str(i)
                            + ","
                            + lmp_model.model_name
                        )
                        with open("output.txt", "a") as f:
                            f.write(t + "\n")

def get_dict_for_main():


    ### For figure 2
    run_plan = {
        "coloring": {

            ProblemLevel.poly: {
                AlgoType.algo_independent: [AgentType.FreeForm],
                AlgoType.incomplete: [AgentType.CompletePseudoCode, AgentType.MainPseudocodeOnly,
                                      AgentType.FunctionPseudoCodeOnly
                                      ]}},
        "matching": {

           ProblemLevel.hard: {
                            AlgoType.algo_independent: [AgentType.FreeForm],
                            AlgoType.incomplete: [AgentType.CompletePseudoCode, AgentType.MainPseudocodeOnly,AgentType.FunctionPseudoCodeOnly
                              ]}},
       "vertex_cover": {

           ProblemLevel.hard: {
               AlgoType.algo_independent: [AgentType.FreeForm],
               AlgoType.incomplete: [AgentType.CompletePseudoCode, AgentType.MainPseudocodeOnly,
                                     AgentType.FunctionPseudoCodeOnly
                                     ]}}}


    #### Figure 4
    # run_plan = {
    #     "coloring": {
    #
    #         ProblemLevel.hard: {
    #             AlgoType.algo_independent: [AgentType.FreeForm],
    #             AlgoType.incomplete: [AgentType.CompletePseudoCode, AgentType.MainPseudocodeOnly,
    #                                 AgentType.FunctionPseudoCodeOnly
    #                                 ]}}}



    ##### For figure 5
    # run_plan = {
    #     "coloring": {
    #
    #         ProblemLevel.hard: {
    #             AlgoType.algo_independent: [AgentType.FreeForm],
    #             AlgoType.complete: [AgentType.CompletePseudoCode, AgentType.MainPseudocodeOnly,
    #                                   AgentType.FunctionPseudoCodeOnly
    #                                   ]}},
    #     "matching": {
    #
    #         ProblemLevel.hard: {
    #             AlgoType.algo_independent: [AgentType.FreeForm],
    #             AlgoType.complete: [AgentType.CompletePseudoCode, AgentType.MainPseudocodeOnly,
    #                                   AgentType.FunctionPseudoCodeOnly
    #                                   ]}},
    #     "vertex_cover": {
    #
    #         ProblemLevel.hard: {
    #             AlgoType.algo_independent: [AgentType.FreeForm],
    #             AlgoType.complete: [AgentType.CompletePseudoCode, AgentType.MainPseudocodeOnly,
    #                                   AgentType.FunctionPseudoCodeOnly
    #                                   ]}}}
    #



    return run_plan

# -------------------------
# CLI entry
# -------------------------
if __name__ == "__main__":
    max_retries = 1  # how many times to retry if answers contain None

    dict1 = get_dict_for_main()  # Configure the loops you want to run here

    for graph_s in [16]:
        for prob, level_map in dict1.items():
            for problem_level, algo_types_map in level_map.items():
                for algo_type, agents_type_list in algo_types_map.items():
                    for agent_type in agents_type_list:
                        parser = argparse.ArgumentParser()
                        # "deepseek/deepseek-r1": "openai",
                        #"deepseek/deepseek-r1:free": "openai",
                        parser.add_argument("--models", type=str, nargs="+", default=[ "deepseek/deepseek-r1"]) #"llama3", "qwen2.5:7b-instruct", "google/gemini-2.5-flash"
                        parser.add_argument("--task", type=str, default=prob)
                        # must be AlgoType (not AgentType)
                        parser.add_argument("--algo_type", type=AlgoType, default=algo_type)
                        parser.add_argument("--graph_models", type=str, nargs="+", default=["ws", "ba", "dt"])
                        parser.add_argument("--start_from_sample", type=int, default=0)
                        parser.add_argument("--samples_per_graph_model", type=int, default=5)
                        parser.add_argument("--graph_size", type=int, default=graph_s)
                        parser.add_argument("--seed", type=int, default=42)
                        parser.add_argument("--disable_chain_of_thought", action="store_true")
                        parser.add_argument("--missing_run_file", type=str, default=None)
                        parser.add_argument("--symbolic_level", type=AgentType, default=agent_type)
                        parser.add_argument("--local_graphs_dir", type=str, default="graphs")

                        args = parser.parse_args()
                        # Make CLI the source of truth
                        if args.models and len(args.models) > 0:
                            os.environ["OPENROUTER_MODEL"] = args.models[0]
                        # Inject problem_level directly (no CLI flag)
                        setattr(args, "problem_level", problem_level)

                        asyncio.run(run(args))
