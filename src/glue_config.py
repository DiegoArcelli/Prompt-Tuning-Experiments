glue_config = {

    # "ax": {
    #     "attribute_1": "premise",
    #     "attribute_2": "hypothesis"
    # },

    "cola": {
        "attribute_1": "sentence",
        "num_to_text": {
            0: "unacceptable",
            1: "acceptable"
        }
    },

    "mnli": {
        "attribute_1": "premise",
        "attribute_2": "hypothesis",
        "num_to_text": {
            0: "entailment",
            1: "neutral",
            2: "not entailment"
        }
    },

    "mrpc": {
        "attribute_1": "sentence1",
        "attribute_2": "sentence2",
        "num_to_text": {
            0: "not equivalent",
            1: "equivalent"
        }
    },

    "qnli": {
        "attribute_1": "question",
        "attribute_2": "sentence",
        "num_to_text": {
            0: "not entailment",
            1: "entailment"
        }
    },

    "qqp": {
        "attribute_1": "question1",
        "attribute_2": "question2",
        "num_to_text": {
            0: "not duplicate",
            1: "duplicate"
        }
    },

    "rte": {
        "attribute_1": "sentence1",
        "attribute_2": "sentence2",
        "num_to_text": {
            0: "entailment",
            1: "not entailment"
        }
    },

    "sst2": {
        "attribute_1": "sentence",
        "num_to_text": {
            0: "negative",
            1: "positive"
        }
    },
    
    # "stsb": {
    #     "attribute_1": "sentence1",
    #     "attribute_2": "sentence2"
    # },

    "wnli": {
        "attribute_1": "sentence1",
        "attribute_2": "sentence2",
        "num_to_text": {
            0: "not entailment",
            1: "entailment"
        }
    },
}
