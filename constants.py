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
    "random",
    *QUERYLESS_SUBMODLIB_FUNCTIONS,
    "diversity",
    "fast_votek",
    #"mfl",
    "votek",
    "least_confidence",
    #*QUERYFULL_SUBMODLIB_FUNCTIONS,
]
