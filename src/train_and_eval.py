"""
Script that runs train.py for a number of steps, then runs evaluate.py on both train and test sets
"""
import subprocess
import sys
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", required=True, type=Path, help="Path to dir with train/test data + vocab")
parser.add_argument("--checkpoint_path", required=True, help="Checkpoint dir")
parser.add_argument("--checkpoint_every", required=True, type=int, help="Checkpoint every X step")
parser.add_argument("--eval_train_take", default=100, help="Num steps to take when evaluating on train set")
parser.add_argument("--eval_train_shuffle_buffer", default=1000,
                    help="Shuffle buffer size when evaluating on train set")

args, model_args = parser.parse_known_args()

train_data_path = str(args.data_dir / "train.tfrecord")
test_data_path = str(args.data_dir / "test.tfrecord")
vocab_path = str(args.data_dir / "vocab.subwords")


def train_and_eval():
    # Training
    print("Starting/resuming training")
    training_cmd = ["python", "train.py",
                    "--data", train_data_path,
                    "--vocab", vocab_path,
                    "--checkpoint_path", args.checkpoint_path,
                    "--checkpoint_every", str(args.checkpoint_every),
                    "--nocontinuous"] + model_args
    training = subprocess.Popen(training_cmd, stdout=sys.stdout)
    print(" ".join(training_cmd))
    return_code = training.wait()
    if return_code != 0:
        exit(return_code)
    print("Training stopped")


    # Evaluation on train set
    evaluation_train_cmd = ["python", "evaluate.py",
                            "--data", train_data_path,
                            "--vocab", vocab_path,
                            "--checkpoint_path", args.checkpoint_path,
                            "--take", str(args.eval_train_take),
                            "--shuffle_buffer", str(args.eval_train_shuffle_buffer)] + model_args
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
                           "--vocab", vocab_path,
                           "--checkpoint_path", args.checkpoint_path] + model_args
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
