import os
import json
import random
from tqdm import tqdm
from datasets import load_dataset

DEBUG_ANNOTATION_SIZE = 2
DEBUG_BATCH_SIZE = 1
DEBUG_TRAIN_SAMPLES = 5
DEBUG_EVAL_SAMPLES = 2


def format_dataset(sample):
    question = sample["question"]["text"]
    context = sample["document"]["tokens"]["token"]
    is_html = sample["document"]["tokens"]["is_html"]
    long_answers = sample["annotations"]["long_answer"]
    short_answers = sample["annotations"]["short_answers"]

    context_string = " ".join(
        [context[i] for i in range(len(context)) if not is_html[i]]
    )

    # 0 - No ; 1 - Yes
    for answer in sample["annotations"]["yes_no_answer"]:
        if answer == 0 or answer == 1:
            return {
                "question": question,
                "short": ["no" if answer == 0 else "yes"],
                "long": [],
                "category": "no" if answer == 0 else "yes",
            }

    short_targets = []
    for s in short_answers:
        short_targets.extend(s["text"])
    short_targets = list(set(short_targets))

    long_targets = []
    for s in long_answers:
        if s["start_token"] == -1:
            continue
        answer = context[s["start_token"] : s["end_token"]]
        html = is_html[s["start_token"] : s["end_token"]]
        new_answer = " ".join([answer[i] for i in range(len(answer)) if not html[i]])
        if new_answer not in long_targets:
            long_targets.append(new_answer)

    category = "other" if len(short_targets) > 0 else "null"

    return {
        "question": question,
        "short": short_targets,
        "long": long_targets,
        "category": category,
    }


def process_mnli_examples(examples):
    processed_examples = []
    idx = 0
    for raw_data in tqdm(examples, desc="process mnli examples"):
        processed_examples.append(
            {
                "id": idx,
                "label": raw_data["label"],
                "premise": raw_data["premise"],
                "hypothesis": raw_data["hypothesis"],
            }
        )
        idx += 1
    return processed_examples


def process_rte_examples(examples):
    processed_examples = []
    idx = 0
    for raw_data in tqdm(examples, desc="process rte examples"):
        processed_examples.append(
            {
                "id": idx,
                "label": raw_data["label"],
                "sentence1": raw_data["sentence1"],
                "sentence2": raw_data["sentence2"],
            }
        )
        idx += 1
    return processed_examples


def process_sst5_examples(examples):
    processed_examples = []
    idx = 0
    for raw_data in tqdm(examples, desc="process sst5 examples"):
        processed_examples.append(
            {
                "id": idx,
                "label": raw_data["label"],
                "text": raw_data["text"],
            }
        )
        idx += 1
    return processed_examples


def process_mrpc_examples(examples):
    processed_examples = []
    idx = 0
    for raw_data in tqdm(examples, desc="process mrpc examples"):
        processed_examples.append(
            {
                "id": idx,
                "label": raw_data["label"],
                "sentence1": raw_data["sentence1"],
                "sentence2": raw_data["sentence2"],
            }
        )
        idx += 1
    return processed_examples


def process_dbpedia_examples(examples):
    processed_examples = []
    idx = 0
    for raw_data in tqdm(examples, desc="process dbpedia_14 examples"):
        processed_examples.append(
            {
                "id": idx,
                "label": raw_data["label"],
                "title": raw_data["title"],
                "content": raw_data["content"],
            }
        )
        idx += 1
    return processed_examples


def process_hellaswag_examples(examples):
    processed_examples = []
    idx = 0
    for raw_data in tqdm(examples, desc="process hellaswag examples"):
        processed_examples.append(
            {
                "id": idx,
                "ctx_a": raw_data["ctx_a"],
                "ctx_b": raw_data["ctx_b"],
                "ctx": raw_data["ctx"],
                "endings": raw_data["endings"],
                "label": int(raw_data["label"]),
                "activity_label": raw_data["activity_label"],
            }
        )
        idx += 1
    return processed_examples


def process_xsum_examples(examples):
    processed_examples = []
    for i, e in enumerate(examples):
        processed_examples.append(
            {
                "id": i,
                "document": e["document"],
                "summary": e["summary"],
                "label": e["summary"],
            }
        )
    return processed_examples


def process_nq_examples(examples):
    processed_examples = []
    for idx, e in enumerate(examples):
        processed_examples.append(
            {
                "id": idx,
                "question": e["question"],
                "short_targets": e["short"],
                "category": e["category"],
                "long": e["long"],
                "label": e["short"],
            }
        )
    return processed_examples


def get_task(args):
    task_name = args.task_name
    data_cache_dir = args.data_cache_dir
    
    if task_name == "mnli":
        mnli_datasets = load_dataset("glue", "mnli", cache_dir=data_cache_dir)
        if args.subsample:
            train_examples_file = os.path.join(args.output_dir, f"train_examples_seed_{args.seed}.json")
            eval_examples_file = os.path.join(args.output_dir, f"eval_examples_seed_{args.seed}.json")
            
            if os.path.isfile(train_examples_file) and os.path.isfile(eval_examples_file):
                print("use cached examples")
                with open(train_examples_file) as f:
                    total_train_examples = json.load(f)
                with open(eval_examples_file) as f:
                    total_eval_examples = json.load(f)
            else:
                total_train_examples = random.sample([e for e in mnli_datasets["train"]], 3000)
                total_eval_examples = random.sample([e for e in mnli_datasets["validation_matched"]], 256)
                
                total_train_examples = process_mnli_examples(total_train_examples)
                total_eval_examples = process_mnli_examples(total_eval_examples)
                
                with open(train_examples_file, "w") as f:
                    json.dump(total_train_examples, f, indent=4)
                with open(eval_examples_file, "w") as f:
                    json.dump(total_eval_examples, f, indent=4)
        else:
            total_train_examples = process_mnli_examples([e for e in mnli_datasets["train"]])
            total_eval_examples = process_mnli_examples([e for e in mnli_datasets["validation_matched"]])            
        if args.debug:
            args.annotation_size = DEBUG_ANNOTATION_SIZE
            args.batch_size = DEBUG_BATCH_SIZE
            total_train_examples = total_train_examples[:DEBUG_TRAIN_SAMPLES]
            total_eval_examples = total_eval_examples[:DEBUG_EVAL_SAMPLES]

        if args.model_name == "mistralai/Mistral-7B-Instruct-v0.2":
            def format_example(example, label_map, **kwargs):
                instruction = f"[INST] {example['premise']}. Based on that information, is the claim {example['hypothesis']} \"True\", " \
                            f'"False", or "Inconclusive"? [/INST]'
                label = f"{label_map[example['label']]}</s>"
                return (instruction, label)
        else:
            def format_example(example, label_map, **kwargs):
                instruction = f"{example['premise']}. Based on that information, is the claim {example['hypothesis']} \"True\", " \
                            f'"False", or "Inconclusive"?\\nanswer:'
                label = f"{label_map[example['label']]}"
                return (instruction, label)

        all_train_text_to_encode = [
            '{}. Based on that information, is the claim {} "True", "False", or "Inconclusive"?'.format(
                raw_item["premise"], raw_item["hypothesis"]
            )
            for raw_item in total_train_examples
        ]
        all_eval_text_to_encode = [
            '{}. Based on that information, is the claim {} "True", "False", or "Inconclusive"?'.format(
                raw_item["premise"], raw_item["hypothesis"]
            )
            for raw_item in total_eval_examples
        ]
        label_map = {0: "True", 1: "Inconclusive", 2: "False"}
    elif task_name == "rte":
        if os.path.isfile(
            os.path.join(args.output_dir, f"train_examples_seed_{args.seed}.json")
        ) and os.path.isfile(
            os.path.join(args.output_dir, f"eval_examples_seed_{args.seed}.json")
        ):
            print("use cached examples")
            with open(
                os.path.join(args.output_dir, f"train_examples_seed_{args.seed}.json")
            ) as f:
                total_train_examples = json.load(f)
            with open(
                os.path.join(args.output_dir, f"eval_examples_seed_{args.seed}.json")
            ) as f:
                total_eval_examples = json.load(f)
        else:
            rte_datasets = load_dataset("glue", "rte", cache_dir=data_cache_dir)
            total_train_examples = [e for e in rte_datasets["train"]]
            total_train_examples = process_rte_examples(total_train_examples)
            total_eval_examples = [e for e in rte_datasets["validation"]]
            if args.subsample:
                total_eval_examples = random.sample(total_eval_examples, 256)
            total_eval_examples = process_rte_examples(total_eval_examples)
            with open(
                os.path.join(args.output_dir, f"train_examples_seed_{args.seed}.json"),
                "w",
            ) as f:
                json.dump(total_train_examples, f, indent=4)
            with open(
                os.path.join(args.output_dir, f"eval_examples_seed_{args.seed}.json"),
                "w",
            ) as f:
                json.dump(total_eval_examples, f, indent=4)
        if args.debug:
            args.annotation_size = DEBUG_ANNOTATION_SIZE
            args.batch_size = DEBUG_BATCH_SIZE
            total_train_examples = total_train_examples[:DEBUG_TRAIN_SAMPLES]
            total_eval_examples = total_eval_examples[:DEBUG_EVAL_SAMPLES]

        if args.model_name == "mistralai/Mistral-7B-Instruct-v0.2":
            def format_example(example, label_map, **kwargs):
                instruction = f"[INST] {example['sentence1']}.\\nquestion: {example['sentence2']}. True or False? [/INST]"
                label = f"{label_map[example['label']]}</s>"
                return (instruction, label)
        else:
            def format_example(example, label_map, **kwargs):
                instruction = f"{example['sentence1']}.\\nquestion: {example['sentence2']}. True or False?\\nanswer:"
                label = f"{label_map[example['label']]}"
                return (instruction, label)

        all_train_text_to_encode = [
            "{}.\nquestion: {}".format(raw_item["sentence1"], raw_item["sentence2"])
            for raw_item in total_train_examples
        ]
        all_eval_text_to_encode = [
            "{}.\nquestion: {}".format(raw_item["sentence1"], raw_item["sentence2"])
            for raw_item in total_eval_examples
        ]
        label_map = {0: "True", 1: "False"}
    elif task_name == "sst5":
        sst5_datasets = load_dataset("SetFit/sst5", cache_dir=data_cache_dir)
        if args.subsample:
            train_examples_file = os.path.join(args.output_dir, f"train_examples_seed_{args.seed}.json")
            eval_examples_file = os.path.join(args.output_dir, f"eval_examples_seed_{args.seed}.json")
            
            if os.path.isfile(train_examples_file) and os.path.isfile(eval_examples_file):
                print("use cached examples")
                with open(train_examples_file) as f:
                    total_train_examples = json.load(f)
                with open(eval_examples_file) as f:
                    total_eval_examples = json.load(f)
            else:
                total_train_examples = random.sample([e for e in sst5_datasets["train"]], 3000)
                total_eval_examples = random.sample([e for e in sst5_datasets["test"]], 256)
                
                total_train_examples = process_sst5_examples(total_train_examples)
                total_eval_examples = process_sst5_examples(total_eval_examples)
                
                with open(train_examples_file, "w") as f:
                    json.dump(total_train_examples, f, indent=4)
                with open(eval_examples_file, "w") as f:
                    json.dump(total_eval_examples, f, indent=4)
        else:
            total_train_examples = process_sst5_examples([e for e in sst5_datasets["train"]])
            total_eval_examples = process_sst5_examples([e for e in sst5_datasets["test"]])
        if args.debug:
            args.annotation_size = DEBUG_ANNOTATION_SIZE
            args.batch_size = DEBUG_BATCH_SIZE
            total_train_examples = total_train_examples[:DEBUG_TRAIN_SAMPLES]
            total_eval_examples = total_eval_examples[:DEBUG_EVAL_SAMPLES]

        if args.model_name == "mistralai/Mistral-7B-Instruct-v0.2":
            def format_example(example, label_map, **kwargs):
                instruction = f"[INST] How do you feel about the following sentence?\\n{example['text']} [/INST]"
                label = f"{label_map[example['label']]}</s>"
                return (instruction, label)
        else:
            def format_example(example, label_map, **kwargs):
                instruction = f"How do you feel about the following sentence?\\n{example['text']}\\nanswer:"
                label = f"{label_map[example['label']]}"
                return (instruction, label)

        all_train_text_to_encode = [
            raw_item["text"] for raw_item in total_train_examples
        ]
        all_eval_text_to_encode = [raw_item["text"] for raw_item in total_eval_examples]
        label_map = {
            0: "very negative",
            1: "negative",
            2: "neutral",
            3: "positive",
            4: "very positive",
        }
    elif task_name == "mrpc":
        mrpc_datasets = load_dataset("glue", "mrpc", cache_dir=data_cache_dir)
        if args.subsample:
            train_examples_file = os.path.join(args.output_dir, f"train_examples_seed_{args.seed}.json")
            eval_examples_file = os.path.join(args.output_dir, f"eval_examples_seed_{args.seed}.json")
            
            if os.path.isfile(train_examples_file) and os.path.isfile(eval_examples_file):
                print("use cached examples")
                with open(train_examples_file) as f:
                    total_train_examples = json.load(f)
                with open(eval_examples_file) as f:
                    total_eval_examples = json.load(f)
            else:
                total_train_examples = random.sample([e for e in mrpc_datasets["train"]], 3000)
                total_eval_examples = random.sample([e for e in mrpc_datasets["validation"]], 256)
                
                total_train_examples = process_mrpc_examples(total_train_examples)
                total_eval_examples = process_mrpc_examples(total_eval_examples)
                
                with open(train_examples_file, "w") as f:
                    json.dump(total_train_examples, f, indent=4)
                with open(eval_examples_file, "w") as f:
                    json.dump(total_eval_examples, f, indent=4)
        else:
            total_train_examples = process_mrpc_examples([e for e in mrpc_datasets["train"]])
            total_eval_examples = process_mrpc_examples([e for e in mrpc_datasets["validation"]])

        if args.debug:
            args.annotation_size = DEBUG_ANNOTATION_SIZE
            args.batch_size = DEBUG_BATCH_SIZE
            total_train_examples = total_train_examples[:DEBUG_TRAIN_SAMPLES]
            total_eval_examples = total_eval_examples[:DEBUG_EVAL_SAMPLES]

        if args.model_name == "mistralai/Mistral-7B-Instruct-v0.2":
            def format_example(example, label_map, **kwargs):
                instruction = f"[INST] Are the following two sentences 'equivalent' or 'not equivalent'?\\n" \
                            f"{example['sentence1']}.\\n{example['sentence2']}. [/INST]"
                label = f"{label_map[example['label']]}</s>"
                return (instruction, label)
        else:
            def format_example(example, label_map, **kwargs):
                instruction = f"Are the following two sentences 'equivalent' or 'not equivalent'?\\n" \
                            f"{example['sentence1']}.\\n{example['sentence2']}.\\nanswer:"
                label = f"{label_map[example['label']]}"
                return (instruction, label)
            
        all_train_text_to_encode = [
            "{}.\n{}".format(raw_item["sentence1"], raw_item["sentence2"])
            for raw_item in total_train_examples
        ]
        all_eval_text_to_encode = [
            "{}.\n{}".format(raw_item["sentence1"], raw_item["sentence2"])
            for raw_item in total_eval_examples
        ]
        label_map = {0: "not equivalent", 1: "equivalent"}
    elif task_name == "dbpedia_14":
        dbpedia_datasets = load_dataset("fancyzhx/dbpedia_14", cache_dir=data_cache_dir)
        if args.subsample:
            train_examples_file = os.path.join(args.output_dir, f"train_examples_seed_{args.seed}.json")
            eval_examples_file = os.path.join(args.output_dir, f"eval_examples_seed_{args.seed}.json")
            
            if os.path.isfile(train_examples_file) and os.path.isfile(eval_examples_file):
                print("use cached examples")
                with open(train_examples_file) as f:
                    total_train_examples = json.load(f)
                with open(eval_examples_file) as f:
                    total_eval_examples = json.load(f)
            else:
                total_train_examples = random.sample([e for e in dbpedia_datasets["train"]], 3000)
                total_eval_examples = random.sample([e for e in dbpedia_datasets["test"]], 256)
                
                total_train_examples = process_dbpedia_examples(total_train_examples)
                total_eval_examples = process_dbpedia_examples(total_eval_examples)
                
                with open(train_examples_file, "w") as f:
                    json.dump(total_train_examples, f, indent=4)
                with open(eval_examples_file, "w") as f:
                    json.dump(total_eval_examples, f, indent=4)
        else:
            total_train_examples = process_dbpedia_examples([e for e in dbpedia_datasets["train"]])
            total_eval_examples = process_dbpedia_examples([e for e in dbpedia_datasets["test"]])
        if args.debug:
            args.annotation_size = DEBUG_ANNOTATION_SIZE
            args.batch_size = DEBUG_BATCH_SIZE
            total_train_examples = total_train_examples[:DEBUG_TRAIN_SAMPLES]
            total_eval_examples = total_eval_examples[:DEBUG_EVAL_SAMPLES]

        if args.model_name == "mistralai/Mistral-7B-Instruct-v0.2":
            def format_example(example, label_map, **kwargs):
                instruction = f"[INST] title: {example['title']}; content: {example['content']} [/INST]"
                label = f"{label_map[example['label']]}</s>"
                return (instruction, label)
        else:
            def format_example(example, label_map, **kwargs):
                instruction = f"title: {example['title']}; content: {example['content']}"
                label = f"{label_map[example['label']]}"
                return (instruction, label)

        all_train_text_to_encode = [
            "title: {} ; content: {}".format(raw_item["title"], raw_item["content"])
            for raw_item in total_train_examples
        ]
        all_eval_text_to_encode = [
            "title: {} ; content: {}".format(raw_item["title"], raw_item["content"])
            for raw_item in total_eval_examples
        ]
        label_map = {
            0: "company",
            1: "educational institution",
            2: "artist",
            3: "athlete",
            4: "office holder",
            5: "mean of transportation",
            6: "building",
            7: "natural place",
            8: "village",
            9: "animal",
            10: "plant",
            11: "album",
            12: "film",
            13: "written work",
        }
    elif task_name == "hellaswag":
        hellaswag_datasets = load_dataset("hellaswag", cache_dir=data_cache_dir)
        if args.subsample:
            train_examples_file = os.path.join(args.output_dir, f"train_examples_seed_{args.seed}.json")
            eval_examples_file = os.path.join(args.output_dir, f"eval_examples_seed_{args.seed}.json")
            
            if os.path.isfile(train_examples_file) and os.path.isfile(eval_examples_file):
                print("use cached examples")
                with open(train_examples_file) as f:
                    total_train_examples = json.load(f)
                with open(eval_examples_file) as f:
                    total_eval_examples = json.load(f)
            else:
                total_train_examples = random.sample([e for e in hellaswag_datasets["train"]], 3000)
                total_eval_examples = random.sample([e for e in hellaswag_datasets["validation"]], 256)
                
                total_train_examples = process_hellaswag_examples(total_train_examples)
                total_eval_examples = process_hellaswag_examples(total_eval_examples)
                
                with open(train_examples_file, "w") as f:
                    json.dump(total_train_examples, f, indent=4)
                with open(eval_examples_file, "w") as f:
                    json.dump(total_eval_examples, f, indent=4)
        else:
            total_train_examples = process_hellaswag_examples([e for e in hellaswag_datasets["train"]])
            total_eval_examples = process_hellaswag_examples([e for e in hellaswag_datasets["validation"]])
        if args.debug:
            args.annotation_size = DEBUG_ANNOTATION_SIZE
            args.batch_size = DEBUG_BATCH_SIZE
            total_train_examples = total_train_examples[:DEBUG_TRAIN_SAMPLES]
            total_eval_examples = total_eval_examples[:DEBUG_EVAL_SAMPLES]

        if args.model_name == "mistralai/Mistral-7B-Instruct-v0.2":
            def format_example(example, label_map, **kwargs):
                instruction = f"[INST] The topic is {example['activity_label']}. {example['ctx_a']} " \
                            f"{example['ctx_b']} [/INST]"
                label = f"{example['endings'][example['label']]}</s>"
                return (instruction, label)
        else:
            def format_example(example, label_map, **kwargs):
                instruction = f"The topic is {example['activity_label']}. {example['ctx_a']} " \
                            f"{example['ctx_b']} "
                label = f"{example['endings'][example['label']]}"
                return (instruction, label)

        all_train_text_to_encode = [
            f"The topic is {raw_item['activity_label']}. {raw_item['ctx_a']} {raw_item['ctx_b']} | "
            f"{raw_item['endings'][0]} | "
            f"{raw_item['endings'][1]} | "
            f"{raw_item['endings'][2]} | "
            f"{raw_item['endings'][3]}"
            for raw_item in total_train_examples
        ]
        all_eval_text_to_encode = [
            f"The topic is {raw_item['activity_label']}. {raw_item['ctx_a']} {raw_item['ctx_b']} | "
            f"{raw_item['endings'][0]} | "
            f"{raw_item['endings'][1]} | "
            f"{raw_item['endings'][2]} | "
            f"{raw_item['endings'][3]}"
            for raw_item in total_eval_examples
        ]
        label_map = None
    elif task_name == "xsum":
        xsum_dataset = load_dataset("xsum", cache_dir=data_cache_dir)
        if args.subsample:
            train_examples_file = os.path.join(args.output_dir, f"train_examples_seed_{args.seed}.json")
            eval_examples_file = os.path.join(args.output_dir, f"eval_examples_seed_{args.seed}.json")
            
            if os.path.isfile(train_examples_file) and os.path.isfile(eval_examples_file):
                print("use cached examples")
                with open(train_examples_file) as f:
                    total_train_examples = json.load(f)
                with open(eval_examples_file) as f:
                    total_eval_examples = json.load(f)
            else:
                total_train_examples = random.sample([e for e in xsum_dataset["train"]], 3000)
                total_eval_examples = random.sample([e for e in xsum_dataset["test"]], 256)
                
                total_train_examples = process_xsum_examples(total_train_examples)
                total_eval_examples = process_xsum_examples(total_eval_examples)
                
                with open(train_examples_file, "w") as f:
                    json.dump(total_train_examples, f, indent=4)
                with open(eval_examples_file, "w") as f:
                    json.dump(total_eval_examples, f, indent=4)
        else:
            total_train_examples = process_xsum_examples([e for e in xsum_dataset["train"]])
            total_eval_examples = process_xsum_examples([e for e in xsum_dataset["test"]])
        if args.debug:
            args.annotation_size = DEBUG_ANNOTATION_SIZE
            args.batch_size = DEBUG_BATCH_SIZE
            total_train_examples = total_train_examples[:DEBUG_TRAIN_SAMPLES]
            total_eval_examples = total_eval_examples[:DEBUG_EVAL_SAMPLES]

        if args.model_name == "mistralai/Mistral-7B-Instruct-v0.2":
            def format_example(example, label_map, **kwargs):
                instruction = f"[INST] write a short summary:\\n{example['document']} [/INST]"
                label = f"{example['summary']}</s>"
                return (instruction, label)
        else:
            def format_example(example, label_map, **kwargs):
                instruction = f"write a short summary:\\n{example['document']}\\nTL;DR:"
                label = f"{example['summary']}"
                return (instruction, label)

        all_train_text_to_encode = [
            raw_item["document"] for raw_item in total_train_examples
        ]
        all_eval_text_to_encode = [
            raw_item["document"] for raw_item in total_eval_examples
        ]
        label_map = None
    elif task_name == "nq":
        nq_dataset = load_dataset("natural_questions", cache_dir=data_cache_dir)

        if args.subsample:
            train_examples_file = os.path.join(args.output_dir, f"train_examples_seed_{args.seed}.json")
            eval_examples_file = os.path.join(args.output_dir, f"eval_examples_seed_{args.seed}.json")

            if os.path.isfile(train_examples_file) and os.path.isfile(eval_examples_file):
                print("use cached examples")
                with open(train_examples_file) as f:
                    total_train_examples = json.load(f)
                with open(eval_examples_file) as f:
                    total_eval_examples = json.load(f)
            else:
                first_sub_sample_indices = random.sample(range(len(nq_dataset["train"])), 12000)
                train_data = nq_dataset["train"].select(first_sub_sample_indices).map(format_dataset)
                total_train_examples = train_data.remove_columns(["annotations", "document", "id"]).filter(lambda x: x["category"] != "null")
                total_train_examples = [e for e in total_train_examples]
                total_train_examples = random.sample(total_train_examples, 3000)
                total_train_examples = process_nq_examples(total_train_examples)

                total_eval_examples = (
                    nq_dataset["validation"]
                    .map(format_dataset)
                    .remove_columns(["annotations", "document", "id"])
                    .filter(lambda x: x["category"] != "null")
                )
                total_eval_examples = [e for e in total_eval_examples]
                total_eval_examples = random.sample(total_eval_examples, 256)
                total_eval_examples = process_nq_examples(total_eval_examples)

                with open(train_examples_file, "w") as f:
                    json.dump(total_train_examples, f, indent=4)
                with open(eval_examples_file, "w") as f:
                    json.dump(total_eval_examples, f, indent=4)
        else:
            train_data = nq_dataset["train"].map(format_dataset)
            total_train_examples = train_data.remove_columns(["annotations", "document", "id"]).filter(lambda x: x["category"] != "null")
            total_train_examples = [e for e in total_train_examples]
            total_train_examples = process_nq_examples(total_train_examples)

            total_eval_examples = (
                nq_dataset["validation"]
                .map(format_dataset)
                .remove_columns(["annotations", "document", "id"])
                .filter(lambda x: x["category"] != "null")
            )
            total_eval_examples = [e for e in total_eval_examples]
            total_eval_examples = process_nq_examples(total_eval_examples)
        
        if args.debug:
            args.annotation_size = DEBUG_ANNOTATION_SIZE
            args.batch_size = DEBUG_BATCH_SIZE
            total_train_examples = total_train_examples[:DEBUG_TRAIN_SAMPLES]
            total_eval_examples = total_eval_examples[:DEBUG_EVAL_SAMPLES]

        if args.model_name == "mistralai/Mistral-7B-Instruct-v0.2":
            def format_example(example, label_map, **kwargs):
                if example["category"] in ["yes", "no"]:
                    instruction = f"[INST] Write an answer: {example['question']} [/INST]"
                    label = f"{example['category']}</s>"
                    return (instruction, label)
                assert example["category"] == "other", example["category"]
                assert len(example["short_targets"]) > 0, f"{example['short_targets']}"
                instruction = f"[INST] Write an answer: {example['question']} [/INST]"
                label = f"{example['short_targets'][0]}</s>"
                return (instruction, label)
        else:
            def format_example(example, label_map, **kwargs):
                if example["category"] in ["yes", "no"]:
                    instruction = f"Write an answer: {example['question']}\\nclass"
                    label = f"{example['category']}"
                    return (instruction, label)
                assert example["category"] == "other", example["category"]
                assert len(example["short_targets"]) > 0, f"{example['short_targets']}"
                instruction = f"Write an answer: {example['question']}\\n{example['category']} "
                label = f"{example['short_targets'][0]}"
                return (instruction, label)
            
        all_train_text_to_encode = [
            raw_item["question"] for raw_item in total_train_examples
        ]
        all_eval_text_to_encode = [
            raw_item["question"] for raw_item in total_eval_examples
        ]
        label_map = None
    else:
        raise ValueError(f"{args.task_name} is not supported")
    return (
        total_train_examples,
        total_eval_examples,
        all_train_text_to_encode,
        all_eval_text_to_encode,
        format_example,
        label_map,
    )
