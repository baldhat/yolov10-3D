import argparse
import os
import distutils.dir_util
from datetime import datetime
import json
import shutil
import time
import subprocess
import numpy as np


EXP_DIR = "/home/stud/mijo/experiments/" # adjust
CODE_DIR = os.path.join(EXP_DIR, "code")
RESULTS_DIR = os.path.join(EXP_DIR, "results")
TEMP_SBATCH_DIR = os.path.join(EXP_DIR, "temp_sbatch")
RUNNING_JOBS_FILE = os.path.join(EXP_DIR, "runnings_jobs2.json")


def get_source_code_dirs(name: str):
    if name.lower().startswith("monodetr"):
        return [
            "/home/stud/mijo/dev/MonoDETR"
        ]
    elif name.lower().startswith("monolss"):
        return [
            "/home/stud/mijo/dev/MonoLSS"
        ]
    elif name.lower().startswith("yolo"):
        return [
            "/home/stud/mijo/dev/yolov10-3D"
        ]
    elif name.lower().startswith("dino"):
        return [
            "/home/stud/mijo/dev/yolov10-3D"
        ]
    else:
        raise NotImplementedError()


def delete_file_folder(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)


def update_running_jobs_list():
    if os.path.exists(RUNNING_JOBS_FILE):
        with open(RUNNING_JOBS_FILE) as f:
            jobs_list = json.load(f)
    else:
        jobs_list = []

    with os.popen('squeue --user mijo') as stream:    # adjust
        running_jobs_str = stream.read()

    new_jobs_list = [(id, name) for (id, name) in jobs_list if id in running_jobs_str]
    with open(RUNNING_JOBS_FILE, "w") as f:
        json.dump(jobs_list, f)

    return new_jobs_list


def append_to_running_jobs_list(new_id, name):
    jobs_list = update_running_jobs_list()
    jobs_list.append((new_id, name))
    with open(RUNNING_JOBS_FILE, "w") as f:
        json.dump(jobs_list, f)


def prepare_file_structure(name: str, sbatch_source_file: str, code_dir: str, results_dir: str, overwrite: bool, resume: bool):
    if not os.path.exists(EXP_DIR):
        print(f"Base directory {EXP_DIR} does not exist")
        return False

    if " " in name:
        print("Choose a name without spaces")

    if not os.path.exists(sbatch_source_file):
        print(f"Could not find the sbatch_file in {sbatch_source_file}")
        return False

    for i, path in enumerate([results_dir, code_dir]):
        if os.path.exists(path):
            if i == 0 and resume:
                pass  # Don't overwrite results, when resume
            elif i == 1 and resume and not overwrite:
                pass
            elif overwrite:
                delete_file_folder(path)
            else:
                print(f"Path {path} already exists. Please clean up first or rename.")
                return False

    os.makedirs(code_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    return True


def get_timestamp_name(name):
    return datetime.now().strftime(f"%Y_%m_%d_%H_%M_%S_{name}")


def create_sbatch(name, sbatch_file, results_dir, code_dir):
    with open(sbatch_file) as f:
        sbatch_source_text = f.read()

    if (
        not "[JOB_NAME]" in sbatch_source_text or
        not "[CODE_DIR]" in sbatch_source_text
    ):
        assert(False)

    sbatch_target_text = sbatch_source_text.replace("[JOB_NAME]", name)

    sbatch_target_text = sbatch_target_text.replace("[CODE_DIR]", code_dir)

    sbatch_target_text = sbatch_target_text.replace("[RESULTS_DIR]", results_dir)

    filename = get_timestamp_name(name) + ".sbatch"
    filename = os.path.join(TEMP_SBATCH_DIR, filename)
    with open(filename, "w") as f:
        f.write(sbatch_target_text)
    return filename


def extract_job_id(stream_output):
    id_beginning = str(stream_output).index("job ")+4
    id = stream_output[id_beginning:len(stream_output)-1]
    return id


def run_sbatch(sbatch_run_file):
    command = ["sbatch", sbatch_run_file]
    p = subprocess.Popen(command, stdout=subprocess.PIPE)
    time.sleep(2)
    stream_output = p.stdout.read().decode('UTF-8')
    p.terminate()
    print(stream_output)

    # Extract ID
    return extract_job_id(stream_output)


def run_experiment(name, overwrite, check, resume):
    # Ensure that we don't overwrite a running job
    jobs_list = update_running_jobs_list()
    for (id, name_) in jobs_list:
        if name == name_:
            print(f"Job is already running under ID {id}. Start under a different name or wait.")
            return

    if check:
        print("The Job has stopped and is not running anymore.")
        return

    # Prepare directories. Ensure that for all experiments there is a unique directory:
    code_dir = os.path.join(CODE_DIR, name)
    sbatch_source_file = os.path.join(EXP_DIR, "sbatch", name+".sbatch")
    results_dir = os.path.join(RESULTS_DIR, name)
    temp_sbatch_dir = os.path.join(TEMP_SBATCH_DIR)
    if not prepare_file_structure(name, sbatch_source_file, code_dir, results_dir, overwrite, resume):
        return

    # Copy code
    if not resume or overwrite:
        source_code_dirs = get_source_code_dirs(name)
        for path in source_code_dirs:
            base_folder = os.path.join(code_dir, os.path.basename(path))
            if os.path.isdir(path):
                os.mkdir(base_folder)
                distutils.dir_util.copy_tree(path, base_folder)
            else:
                shutil.copy(path, base_folder)

    # Create and run sbatch
    sbatch_run_file = create_sbatch(name, sbatch_source_file, results_dir, code_dir)
    run_id = run_sbatch(sbatch_run_file)
    append_to_running_jobs_list(run_id, name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run an experiment on the cluster')
    parser.add_argument('--name', type=str, help='Name of the experiment. Also name of the sbatch file in the sbatch folder without ".sbatch"', required=True)
    parser.add_argument('--overwrite', action='store_true', help='Overwrite the temporary data in case the experiment has alrady been run. Also overwrite the code.')
    parser.add_argument('--check', action='store_true', help='Check, if the experiment is still running')
    parser.add_argument('--resume', action='store_true', help='Keep the results directory. If used in combination with --overwrite, it will only override the results but not the code.')
    args = parser.parse_args()
    run_experiment(args.name, args.overwrite, args.check, args.resume)


