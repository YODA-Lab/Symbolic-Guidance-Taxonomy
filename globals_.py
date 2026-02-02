from enum import Enum

run_until_converge =True
class AgentType(Enum):
    Code = 7
    FreeForm = 1
    HeuristicHintGuided = 2
    FunctionDescriptionOnly = 3
    FunctionPseudoCodeOnly = 4
    MainPseudocodeOnly = 5
    CompletePseudoCode = 6



class ProblemLevel(Enum):
    poly = 1
    hard = 2




class Problems(Enum):
    coloring = 1
    leader_election = 2
    consensus = 3
    matching = 4

class AlgoType(Enum):
    complete = 1
    incomplete = 2
    algo_independent = 3

    @classmethod
    def list_all(cls):
        return list(cls)
