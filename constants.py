TASK_NAMES = [
    "mnli",
    "rte",
    "sst5",
    "mrpc",
    "dbpedia_14",
    "hellaswag",
    "xsum",
    # "nq", # Needs OpenAI API
]


QUERYLESS_SUBMODLIB_FUNCTIONS = [
    # "ConcaveOverModularFunction",
    "DisparityMinFunction",
    "DisparitySumFunction",
    # "FacilityLocationConditionalGainFunction",
    # "FacilityLocationConditionalMutualInformationFunction",
    "FacilityLocationFunction",
    # "FacilityLocationMutualInformationFunction",
    # "FacilityLocationVariantMutualInformationFunction",
    # "FeatureBasedFunction",
    # "GraphCutConditionalGainFunction",
    "GraphCutFunction",
    # "GraphCutMutualInformationFunction",
    # "LogDeterminantConditionalGainFunction",
    # "LogDeterminantConditionalMutualInformationFunction",
    "LogDeterminantFunction",
    # "LogDeterminantMutualInformationFunction",
    # "ProbabilisticSetCoverConditionalGainFunction",
    # "ProbabilisticSetCoverConditionalMutualInformationFunction",
    # "ProbabilisticSetCoverFunction",
    # "ProbabilisticSetCoverMutualInformationFunction",
    # "SetCoverConditionalGainFunction",
    # "SetCoverConditionalMutualInformationFunction",
    # "SetCoverFunction",
    # "SetCoverMutualInformationFunction",
]


def generate_combinations(lst):
    # List to store the combinations
    result = []
    # Loop through each element in the list
    for i in lst:
        for j in lst:
            # Create the combination and append to the result list
            result.append(f"{i}-{j}")
    return result


QUERYLESS_SUBMODLIB_FUNCTIONS_TWO_SEQUENCE = generate_combinations(
    QUERYLESS_SUBMODLIB_FUNCTIONS
)

QUERYFULL_SUBMODLIB_FUNCTIONS = [
    # "ConcaveOverModularFunction",
    # "DisparityMinFunction",
    # "DisparitySumFunction",
    # "FacilityLocationConditionalGainFunction",
    # "FacilityLocationConditionalMutualInformationFunction",
    # "FacilityLocationFunction",
    "FacilityLocationMutualInformationFunction",
    "FacilityLocationVariantMutualInformationFunction",
    # "FeatureBasedFunction",
    # "GraphCutConditionalGainFunction",
    # "GraphCutFunction",
    "GraphCutMutualInformationFunction",
    # "LogDeterminantConditionalGainFunction",
    # "LogDeterminantConditionalMutualInformationFunction",
    # "LogDeterminantFunction",
    "LogDeterminantMutualInformationFunction",
    # "ProbabilisticSetCoverConditionalGainFunction",
    # "ProbabilisticSetCoverConditionalMutualInformationFunction",
    # "ProbabilisticSetCoverFunction",
    # "ProbabilisticSetCoverMutualInformationFunction",
    # "SetCoverConditionalGainFunction",
    # "SetCoverConditionalMutualInformationFunction",
    # "SetCoverFunction",
    # "SetCoverMutualInformationFunction",
]


SELECTIVE_ANNOTATION_METHODS = [
    *QUERYLESS_SUBMODLIB_FUNCTIONS,
    *QUERYLESS_SUBMODLIB_FUNCTIONS_TWO_SEQUENCE,
    "random",
    "diversity",
    "fast_votek",
    # "mfl",
    "votek",
    "least_confidence",
    # *QUERYFULL_SUBMODLIB_FUNCTIONS,
]
