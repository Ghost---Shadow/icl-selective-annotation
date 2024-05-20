from pathlib import Path
import shutil
import unittest
import torch
from MetaICL.metaicl.data import MetaICLData
from MetaICL.metaicl.model import MetaICLModel
from constants import (
    QUERYFULL_SUBMODLIB_FUNCTIONS,
    QUERYLESS_SUBMODLIB_FUNCTIONS,
    QUERYLESS_SUBMODLIB_FUNCTIONS_TWO_SEQUENCE,
)
from get_task import get_task
from two_steps import selective_annotation
from transformers import AutoModelForCausalLM, AutoTokenizer


class TestSelectiveAnnotation(unittest.TestCase):
    # python -m unittest two_steps_test.TestSelectiveAnnotation.test_facility_location_selection -v
    def test_facility_location_selection(self):
        # Generate synthetic embeddings using PyTorch
        num_embeddings = 300
        embedding_dim = 128
        embeddings = torch.randn(num_embeddings, embedding_dim)

        # Define args and kwargs to mimic your actual function call
        args = type("", (), {})()
        args.selective_annotation_method = "FacilityLocationFunction"
        args.annotation_size = 150
        kwargs = {"embeddings": embeddings}

        # Call the selective_annotation function
        selected_indices = selective_annotation(args, **kwargs)

        assert type(selected_indices) == list
        assert type(selected_indices[0]) == int
        assert len(selected_indices) == args.annotation_size

    # python -m unittest two_steps_test.TestSelectiveAnnotation.test_submodlib_queryless -v
    def test_submodlib_queryless(self):
        num_embeddings = 300
        embedding_dim = 128
        embeddings = torch.randn(num_embeddings, embedding_dim)

        for method in QUERYLESS_SUBMODLIB_FUNCTIONS:
            print(method)
            with self.subTest(method=method):
                args = type("", (), {})()
                args.selective_annotation_method = method
                args.annotation_size = 150
                kwargs = {"embeddings": embeddings}

                selected_indices = selective_annotation(args, **kwargs)

                assert type(selected_indices) == list
                assert type(selected_indices[0]) == int
                assert len(selected_indices) == args.annotation_size

    # python -m unittest two_steps_test.TestSelectiveAnnotation.test_submodlib_queryless_two_step -v
    def test_submodlib_queryless_two_step(self):
        num_embeddings = 300
        embedding_dim = 128
        embeddings = torch.randn(num_embeddings, embedding_dim)

        for method in QUERYLESS_SUBMODLIB_FUNCTIONS_TWO_SEQUENCE:
            print("-" * 80)
            print(method)
            with self.subTest(method=method):
                args = type("", (), {})()
                args.selective_annotation_method = method
                args.annotation_size = 50
                args.two_step_budget_multiplier = 2
                kwargs = {"embeddings": embeddings}

                selected_indices = selective_annotation(args, **kwargs)

                assert type(selected_indices) == list
                assert type(selected_indices[0]) == int
                assert len(selected_indices) == args.annotation_size

    # python -m unittest two_steps_test.TestSelectiveAnnotation.test_flmi -v
    def test_flmi(self):
        args = type("", (), {})()
        args.task_name = "dbpedia_14"
        args.model_cache_dir = "models"
        args.data_cache_dir = "datasets"
        args.output_dir = "outputs/tests/FacilityLocationMutualInformationFunction"
        args.model_key = None
        args.prompt_retrieval_method = "similar"
        args.model_name = "EleutherAI/gpt-neo-125m"
        args.selective_annotation_method = "FacilityLocationMutualInformationFunction"
        args.annotation_size = 2
        args.seed = 42
        args.batch_size = 1
        args.debug = True
        try:
            shutil.rmtree(args.output_dir)
        except FileNotFoundError:
            ...
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        (
            train_examples,
            eval_examples,
            train_text_to_encode,
            eval_text_to_encode,
            format_example,
            label_map,
        ) = get_task(args=args)

        num_embeddings = len(train_examples)
        embedding_dim = 128
        embeddings = torch.randn(num_embeddings, embedding_dim)

        inference_data_module = MetaICLData(
            method="direct",
            max_length=1024,
            max_length_per_example=256,
        )

        inference_model = MetaICLModel(args=args)
        inference_model.load()
        inference_model.cuda()
        inference_model.eval()
        tokenizer_gpt = AutoTokenizer.from_pretrained(args.model_name)

        # Prepare kwargs for the selective_annotation function
        kwargs = {
            "embeddings": embeddings,
            "train_examples": train_examples,
            "eval_examples": eval_examples,
            "train_text_to_encode": train_text_to_encode,
            "eval_text_to_encode": eval_text_to_encode,
            "format_example": format_example,
            "label_map": label_map,
            "return_string": False,
            "maximum_input_len": 1000,
            "single_context_example_len": 250,
            "inference_model": inference_model,
            "inference_data_module": inference_data_module,
            "tokenizer_gpt": tokenizer_gpt,
        }

        # Call the selective_annotation function
        selected_indices = selective_annotation(args, **kwargs)

        assert type(selected_indices) == list
        assert type(selected_indices[0]) == int
        assert len(selected_indices) == args.annotation_size

    # python -m unittest two_steps_test.TestSelectiveAnnotation.test_queryfull_submodlib -v
    def test_queryfull_submodlib(self):
        for method in QUERYFULL_SUBMODLIB_FUNCTIONS:
            print(method)
            with self.subTest(method=method):
                args = type("", (), {})()
                args.task_name = "dbpedia_14"
                args.model_cache_dir = "models"
                args.data_cache_dir = "datasets"
                args.output_dir = f"outputs/tests/{method}"
                args.model_key = None
                args.prompt_retrieval_method = "similar"
                args.model_name = "EleutherAI/gpt-neo-125m"
                args.selective_annotation_method = method
                args.annotation_size = 2
                args.seed = 42
                args.batch_size = 1
                args.debug = True
                try:
                    shutil.rmtree(args.output_dir)
                except FileNotFoundError:
                    ...
                Path(args.output_dir).mkdir(parents=True, exist_ok=True)

                (
                    train_examples,
                    eval_examples,
                    train_text_to_encode,
                    eval_text_to_encode,
                    format_example,
                    label_map,
                ) = get_task(args=args)

                num_embeddings = len(train_examples)
                embedding_dim = 128
                embeddings = torch.randn(num_embeddings, embedding_dim)

                inference_data_module = MetaICLData(
                    method="direct",
                    max_length=1024,
                    max_length_per_example=256,
                )

                inference_model = MetaICLModel(args=args)
                inference_model.load()
                inference_model.cuda()
                inference_model.eval()
                tokenizer_gpt = AutoTokenizer.from_pretrained(args.model_name)

                # Prepare kwargs for the selective_annotation function
                kwargs = {
                    "embeddings": embeddings,
                    "train_examples": train_examples,
                    "eval_examples": eval_examples,
                    "train_text_to_encode": train_text_to_encode,
                    "eval_text_to_encode": eval_text_to_encode,
                    "format_example": format_example,
                    "label_map": label_map,
                    "return_string": False,
                    "maximum_input_len": 1000,
                    "single_context_example_len": 250,
                    "inference_model": inference_model,
                    "inference_data_module": inference_data_module,
                    "tokenizer_gpt": tokenizer_gpt,
                }

                # Call the selective_annotation function
                selected_indices = selective_annotation(args, **kwargs)

                assert type(selected_indices) == list
                assert type(selected_indices[0]) == int
                assert len(selected_indices) == args.annotation_size

    # python -m unittest two_steps_test.TestSelectiveAnnotation.test_mfl_selection -v
    def test_mfl_selection(self):
        args = type("", (), {})()
        args.selective_annotation_method = "mfl"
        args.annotation_size = 10

        num_data_points = 100  # Example number of data points
        embedding_dim = 128  # Example embedding dimension
        embeddings = torch.rand(num_data_points, embedding_dim)

        kwargs = {"embeddings": embeddings}

        selected_indices = selective_annotation(args, **kwargs)

        assert type(selected_indices) == list
        assert len(selected_indices) == args.annotation_size


if __name__ == "__main__":
    unittest.main()
