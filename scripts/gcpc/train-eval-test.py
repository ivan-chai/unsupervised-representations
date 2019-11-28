#!/usr/bin/env python3
"""Train Contrastive Predictive Coding model."""
import argparse
import logging

from urep.estimators import GCPCEstimator as Estimator
from urep.data import MultivariateNormalDataset as Dataset
from urep.evaluate import Evaluator
from urep.io import read_json
from urep.parallel import DataParallel
from urep.train import Trainer


def parse_arguments():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model-dir", help="Path to output model dir", required=True)
    parser.add_argument("--model-config", help="Path to model config")
    parser.add_argument("--data-config", help="Path to data config")
    parser.add_argument("--train-config", help="Path to training config")
    parser.add_argument("--eval-config", help="Path to evaluation config")
    parser.add_argument("--num-train-steps", help="Number of batches to process during training", type=int)
    parser.add_argument("--num-eval-steps", help="Number of batches to process during evaluation", type=int)
    parser.add_argument("--train-batch-size", help="Batch size for training", type=int)
    parser.add_argument("--eval-batch-size", help="Batch size for training", type=int)
    parser.add_argument("--log-steps", help="Number of steps between log output", type=int)
    return parser.parse_args()


def main(args):
    train_dataset = Dataset(args.data_config)
    eval_dataset = Dataset(args.data_config)

    model = DataParallel(Estimator(train_dataset.out_channels, args.model_config))

    eval_config = read_json(args.eval_config) if args.eval_config else {}
    if args.num_eval_steps is not None:
        eval_config["num_steps"] = args.num_eval_steps
    if args.eval_batch_size is not None:
        eval_config["batch_size"] = args.eval_batch_size
    evaluator = Evaluator(eval_config)

    train_config = read_json(args.train_config) if args.train_config else {}
    if args.num_train_steps is not None:
        train_config["num_steps"] = args.num_train_steps
    if args.train_batch_size is not None:
        train_config["batch_size"] = args.train_batch_size
    if args.log_steps is not None:
        train_config["logging_steps"] = args.log_steps
    trainer = Trainer(train_config)

    evaluate_fn = lambda: evaluator.evaluation_hook(model, args.model_dir, eval_dataset)
    trainer.train(model, train_dataset, args.model_dir,
                  eval_hook=evaluate_fn)


if __name__ == "__main__":
    main(parse_arguments())
