import os
import tempfile
import pandas
import numpy as np


def get_project_paths(script_file, to_tmp=False):
    project_name = get_project_name(script_file)
    work_dir = os.getcwd()

    if to_tmp:
        log_dir = os.path.join(tempfile.gettempdir(), "iNNspector", "log", project_name)
    else:
        log_dir = os.path.join(work_dir, "log", project_name)

    project_paths = {
        "checkpoints": os.path.join(log_dir, "checkpoints"),
        "graphs": os.path.join(log_dir, "graphs"),
        "plots": os.path.join(log_dir, "plots"),
        "weights": os.path.join(log_dir, "weights"),
        "tb": os.path.join(log_dir, "logs"),
        "code": os.path.join(log_dir, "code"),

    }

    # Make sure that directories exist
    for _, path in project_paths.items():
        os.makedirs(path, exist_ok=True)

    return project_paths


def get_project_name(script_file):
    file_name, _ = os.path.splitext(os.path.basename(script_file))

    return file_name


def add_epochs(filepath):
    files=os.listdir(filepath)
    for i in files:
        if (i.find('weights')):
            Data = pandas.read_csv(filepath + "/" + i)
            Data =  np.array(Data)
            
