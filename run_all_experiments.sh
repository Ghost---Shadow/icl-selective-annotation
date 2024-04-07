#!/bin/bash

python main.py --task_name mnli --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mnli/random --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mnli/diversity --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mnli/fast_votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method mfl --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mnli/mfl --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mnli/votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method least_confidence --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mnli/least_confidence --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method DisparityMinFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mnli/DisparityMinFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method DisparitySumFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mnli/DisparitySumFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method FacilityLocationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mnli/FacilityLocationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method GraphCutFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mnli/GraphCutFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method LogDeterminantFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mnli/LogDeterminantFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method FacilityLocationMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mnli/FacilityLocationMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method FacilityLocationVariantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mnli/FacilityLocationVariantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method GraphCutMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mnli/GraphCutMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method LogDeterminantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mnli/LogDeterminantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/rte/random --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/rte/diversity --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/rte/fast_votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method mfl --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/rte/mfl --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/rte/votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method least_confidence --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/rte/least_confidence --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method DisparityMinFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/rte/DisparityMinFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method DisparitySumFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/rte/DisparitySumFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method FacilityLocationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/rte/FacilityLocationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method GraphCutFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/rte/GraphCutFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method LogDeterminantFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/rte/LogDeterminantFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method FacilityLocationMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/rte/FacilityLocationMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method FacilityLocationVariantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/rte/FacilityLocationVariantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method GraphCutMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/rte/GraphCutMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method LogDeterminantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/rte/LogDeterminantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/sst5/random --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/sst5/diversity --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/sst5/fast_votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method mfl --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/sst5/mfl --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/sst5/votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method least_confidence --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/sst5/least_confidence --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method DisparityMinFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/sst5/DisparityMinFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method DisparitySumFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/sst5/DisparitySumFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method FacilityLocationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/sst5/FacilityLocationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method GraphCutFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/sst5/GraphCutFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method LogDeterminantFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/sst5/LogDeterminantFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method FacilityLocationMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/sst5/FacilityLocationMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method FacilityLocationVariantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/sst5/FacilityLocationVariantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method GraphCutMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/sst5/GraphCutMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method LogDeterminantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/sst5/LogDeterminantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mrpc/random --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mrpc/diversity --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mrpc/fast_votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method mfl --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mrpc/mfl --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mrpc/votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method least_confidence --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mrpc/least_confidence --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method DisparityMinFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mrpc/DisparityMinFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method DisparitySumFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mrpc/DisparitySumFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method FacilityLocationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mrpc/FacilityLocationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method GraphCutFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mrpc/GraphCutFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method LogDeterminantFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mrpc/LogDeterminantFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method FacilityLocationMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mrpc/FacilityLocationMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method FacilityLocationVariantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mrpc/FacilityLocationVariantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method GraphCutMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mrpc/GraphCutMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method LogDeterminantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mrpc/LogDeterminantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/dbpedia_14/random --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/dbpedia_14/diversity --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/dbpedia_14/fast_votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method mfl --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/dbpedia_14/mfl --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/dbpedia_14/votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method least_confidence --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/dbpedia_14/least_confidence --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method DisparityMinFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/dbpedia_14/DisparityMinFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method DisparitySumFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/dbpedia_14/DisparitySumFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method FacilityLocationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/dbpedia_14/FacilityLocationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method GraphCutFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/dbpedia_14/GraphCutFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method LogDeterminantFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/dbpedia_14/LogDeterminantFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method FacilityLocationMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/dbpedia_14/FacilityLocationMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method FacilityLocationVariantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/dbpedia_14/FacilityLocationVariantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method GraphCutMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/dbpedia_14/GraphCutMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method LogDeterminantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/dbpedia_14/LogDeterminantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/hellaswag/random --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/hellaswag/diversity --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/hellaswag/fast_votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method mfl --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/hellaswag/mfl --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/hellaswag/votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method least_confidence --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/hellaswag/least_confidence --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method DisparityMinFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/hellaswag/DisparityMinFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method DisparitySumFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/hellaswag/DisparitySumFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method FacilityLocationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/hellaswag/FacilityLocationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method GraphCutFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/hellaswag/GraphCutFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method LogDeterminantFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/hellaswag/LogDeterminantFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method FacilityLocationMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/hellaswag/FacilityLocationMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method FacilityLocationVariantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/hellaswag/FacilityLocationVariantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method GraphCutMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/hellaswag/GraphCutMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method LogDeterminantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/hellaswag/LogDeterminantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/xsum/random --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/xsum/diversity --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/xsum/fast_votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method mfl --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/xsum/mfl --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/xsum/votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method least_confidence --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/xsum/least_confidence --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method DisparityMinFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/xsum/DisparityMinFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method DisparitySumFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/xsum/DisparitySumFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method FacilityLocationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/xsum/FacilityLocationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method GraphCutFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/xsum/GraphCutFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method LogDeterminantFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/xsum/LogDeterminantFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method FacilityLocationMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/xsum/FacilityLocationMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method FacilityLocationVariantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/xsum/FacilityLocationVariantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method GraphCutMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/xsum/GraphCutMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method LogDeterminantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/xsum/LogDeterminantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B

python exelify_results.py
