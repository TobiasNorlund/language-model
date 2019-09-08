"""
Script that runs train.py for a number of steps, then runs evaluate.py on both train and test sets
"""
import subprocess
import sys
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", required=True, type=Path, help="Path to dir with train/test data + vocab")
parser.add_argument("--common_args", default="", help="Arguments to supply to all scripts")
parser.add_argument("--train_args", default="", help="Arguments specific to training")
parser.add_argument("--eval_train_args", default="", help="Arguments specific to evaluation of training set")
parser.add_argument("--eval_test_args", default="", help="Arguments specific to evaluation of test set")

args = parser.parse_args()

train_data_path = str(args.data_dir / "train.tfrecord")
test_data_path = str(args.data_dir / "test.tfrecord")
vocab_path = str(args.data_dir / "vocab.subwords")


def train_and_eval():
    # Training
    print("Starting/resuming training")
    training_cmd = ["python", "train.py",
                    "--data", train_data_path,
                    "--vocab", vocab_path,
                    "--nocontinuous"] + \
                    args.common_args.split() + \
                    args.train_args.split()
    training = subprocess.Popen(training_cmd, stdout=sys.stdout)
    print(" ".join(training_cmd))
    return_code = training.wait()
    if return_code != 0:
        exit(return_code)
    print("Training stopped")


    # Evaluation on train set
    evaluation_train_cmd = ["python", "evaluate.py",
                            "--data", train_data_path,
                            "--vocab", vocab_path] + \
                            args.common_args.split() + \
                            args.eval_train_args.split()
    evaluation_train = subprocess.Popen(evaluation_train_cmd, stdout=sys.stdout)
    print("Evaluating on training set")
    print(" ".join(evaluation_train_cmd))
    return_code = evaluation_train.wait()
    if return_code != 0:
        exit(return_code)
    print("Evaluation on training set finished")

    # Evaluation on test set
    evaluation_test_cmd = ["python", "evaluate.py",
                           "--data", test_data_path,
                           "--vocab", vocab_path] + \
                           args.common_args.split() + \
                           args.eval_test_args.split()
    evaluation_test = subprocess.Popen(evaluation_test_cmd, stdout=sys.stdout)
    print("Evaluating on test set")
    print(" ".join(evaluation_test_cmd))
    return_code = evaluation_test.wait()
    if return_code != 0:
        exit(return_code)
    print("Evaluation on test set finished")

try:
    while True:
        train_and_eval()
except KeyboardInterrupt:
    pass
