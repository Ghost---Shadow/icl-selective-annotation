# Autogenerated with generate_run_py.py
from cvar_pyutils.ccc import submit_dependant_jobs

run_commands = [
    "python main.py --task_name mnli --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mnli_full/random --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name mnli --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mnli_full/diversity --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name mnli --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mnli_full/fast_votek --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name mnli --selective_annotation_method DisparityMinFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mnli_full/DisparityMinFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name mnli --selective_annotation_method DisparitySumFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mnli_full/DisparitySumFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name mnli --selective_annotation_method FacilityLocationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mnli_full/FacilityLocationFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name mnli --selective_annotation_method GraphCutFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mnli_full/GraphCutFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name mnli --selective_annotation_method LogDeterminantFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mnli_full/LogDeterminantFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name rte --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/rte_full/random --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name rte --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/rte_full/diversity --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name rte --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/rte_full/fast_votek --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name rte --selective_annotation_method DisparityMinFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/rte_full/DisparityMinFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name rte --selective_annotation_method DisparitySumFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/rte_full/DisparitySumFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name rte --selective_annotation_method FacilityLocationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/rte_full/FacilityLocationFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name rte --selective_annotation_method GraphCutFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/rte_full/GraphCutFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name rte --selective_annotation_method LogDeterminantFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/rte_full/LogDeterminantFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name sst5 --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/sst5_full/random --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name sst5 --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/sst5_full/diversity --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name sst5 --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/sst5_full/fast_votek --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name sst5 --selective_annotation_method DisparityMinFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/sst5_full/DisparityMinFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name sst5 --selective_annotation_method DisparitySumFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/sst5_full/DisparitySumFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name sst5 --selective_annotation_method FacilityLocationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/sst5_full/FacilityLocationFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name sst5 --selective_annotation_method GraphCutFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/sst5_full/GraphCutFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name sst5 --selective_annotation_method LogDeterminantFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/sst5_full/LogDeterminantFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name mrpc --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mrpc_full/random --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name mrpc --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mrpc_full/diversity --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name mrpc --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mrpc_full/fast_votek --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name mrpc --selective_annotation_method DisparityMinFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mrpc_full/DisparityMinFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name mrpc --selective_annotation_method DisparitySumFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mrpc_full/DisparitySumFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name mrpc --selective_annotation_method FacilityLocationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mrpc_full/FacilityLocationFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name mrpc --selective_annotation_method GraphCutFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mrpc_full/GraphCutFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name mrpc --selective_annotation_method LogDeterminantFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/mrpc_full/LogDeterminantFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name dbpedia_14 --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/dbpedia_14_full/random --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name dbpedia_14 --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/dbpedia_14_full/diversity --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name dbpedia_14 --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/dbpedia_14_full/fast_votek --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name dbpedia_14 --selective_annotation_method DisparityMinFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/dbpedia_14_full/DisparityMinFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name dbpedia_14 --selective_annotation_method DisparitySumFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/dbpedia_14_full/DisparitySumFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name dbpedia_14 --selective_annotation_method FacilityLocationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/dbpedia_14_full/FacilityLocationFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name dbpedia_14 --selective_annotation_method GraphCutFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/dbpedia_14_full/GraphCutFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name dbpedia_14 --selective_annotation_method LogDeterminantFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/dbpedia_14_full/LogDeterminantFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name hellaswag --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/hellaswag_full/random --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name hellaswag --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/hellaswag_full/diversity --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name hellaswag --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/hellaswag_full/fast_votek --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name hellaswag --selective_annotation_method DisparityMinFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/hellaswag_full/DisparityMinFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name hellaswag --selective_annotation_method DisparitySumFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/hellaswag_full/DisparitySumFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name hellaswag --selective_annotation_method FacilityLocationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/hellaswag_full/FacilityLocationFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name hellaswag --selective_annotation_method GraphCutFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/hellaswag_full/GraphCutFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name hellaswag --selective_annotation_method LogDeterminantFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/hellaswag_full/LogDeterminantFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name xsum --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/xsum_full/random --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name xsum --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/xsum_full/diversity --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name xsum --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/xsum_full/fast_votek --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name xsum --selective_annotation_method DisparityMinFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/xsum_full/DisparityMinFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name xsum --selective_annotation_method DisparitySumFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/xsum_full/DisparitySumFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name xsum --selective_annotation_method FacilityLocationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/xsum_full/FacilityLocationFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name xsum --selective_annotation_method GraphCutFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/xsum_full/GraphCutFunction --model_name EleutherAI/gpt-j-6B",
    "python main.py --task_name xsum --selective_annotation_method LogDeterminantFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/gpt-j-6B/xsum_full/LogDeterminantFunction --model_name EleutherAI/gpt-j-6B",
    "python exelify_results.py",
]

submit_dependant_jobs(
    number_of_rolling_jobs=57,
    command_to_run=run_commands,
    machine_type="x86",
    conda_env="cords",
    out_file="mistral_out.txt",
    err_file="mistral_err.txt",
    queue="nonstandard",
    time="12h",
    num_cores=12,
    num_gpus=1,
    mem="120g",
    gpu_type="a100_40gb",
    mail_log_file_when_done="krishnateja.k@ibm.com",
    mail_notification_on_start="krishnateja.k@ibm.com",
)
