import argparse
import json
import subprocess
from generate_run_all_experiments import generate_commands

IMPORT_COMMAND = """# Autogenerated with generate_run_py.py
from cvar_pyutils.ccc import submit_dependant_jobs

"""

INVOKE_COMMAND = """
submit_dependant_jobs(
    number_of_rolling_jobs={n_jobs},
    command_to_run=run_commands,
    machine_type="x86",
    conda_env="cords",
    out_file="{model_name}_out.txt",
    err_file="{model_name}_err.txt",
    queue="nonstandard",
    time="12h",
    num_cores=12,
    num_gpus=1,
    mem="400g",
    gpu_type="a100_80gb",
    mail_log_file_when_done="krishnateja.k@ibm.com",
    mail_notification_on_start="krishnateja.k@ibm.com",
)
"""


def main(model_name, subsample):
    with open("run.py", "w") as file:
        file.write(IMPORT_COMMAND)

        all_commands = list(generate_commands(model_name, subsample))
        all_commands.append("python exelify_results.py")

        file.write("run_commands = " + json.dumps(all_commands, indent=4))

        file.write("\n")
        file.write(
            INVOKE_COMMAND.format(
                n_jobs=len(all_commands), model_name=model_name.split("/")[-1]
            )
        )

    subprocess.Popen(["black", "run.py"]).communicate()


if __name__ == "__main__":
    # python generate_run_py.py --model_name=EleutherAI/gpt-neo-125m
    # python generate_run_py.py --model_name=EleutherAI/gpt-j-6B
    parser = argparse.ArgumentParser(description="Generate run.py script.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name to use for all experiments.",
    )
    parser.add_argument(
        "--subsample",
        type=bool,
        help="Subsample to use for all experiments.",
    )
    args = parser.parse_args()
    print(args)

    main(args.model_name, args.subsample)
