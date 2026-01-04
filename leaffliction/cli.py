from __future__ import annotations

import argparse


class CLIBuilder:
    """
    Centralise argparse pour éviter duplication dans les entrypoints root.
    Tu peux faire 5 méthodes: distribution, augmentation, transformation, train, predict.
    """

    def build_distribution_parser(self) -> argparse.ArgumentParser:
        """
        Build and configure the argument parser for the Distribution script.

        This parser handles:
        - A mandatory dataset directory containing plant image classes.
        - Optional visualization modes for distribution analysis.
        - Optional bonus features such as statistics export and verbose output.

        :param self: Instance of the Distribution command handler.
        :return: Configured argument parser for the Distribution script.
        :rtype: argparse.ArgumentParser
        """
        parser = argparse.ArgumentParser(
            description="Leaffliction - Distribution script"
        )

        parser.add_argument(
            "dataset_dir",
            type=str,
            help="Directory containing the plant dataset"
        )

        parser.add_argument(
            "--mode",
            choices=["both", "bar", "pie"],
            default="both",
            help="Choose type of graph for distribution"
        )

        parser.add_argument(
            "--stats",
            action="store_true",
            help="Export distribution statistics to a file"
        )

        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Display detailed processing information"
        )

        return parser


    def build_augmentation_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Leaffliction - Augmentation script"
        )

        parser.add_argument(
            "image_path",
            type=str,
            help="Path to an input image (e.g., ./Apple/apple_healthy/image (1).JPG)"
        )

        parser.add_argument(
            "--output-dir",
            type=str,
            default="augmented_directory",
            help='Output directory for augmented images (default: "augmented_directory")'
        )

        parser.add_argument(
            "--show",
            action="store_true",
            help="Display augmented images"
        )

        parser.add_argument(
            "--display",
            choices=["full", "one"],
            default="full",
            help="Display mode: show all augmentations at once (full) or one-by-one (one)"
        )

        parser.add_argument(
            "--formats",
            nargs="+",
            choices=["jpg", "jpeg", "png"],
            default=["jpg"],
            help="Accepted image extensions"
        )

        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Display detailed processing information"
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Random seed for reproducible augmentations"
        )

        parser.add_argument(
            "--only",
            nargs="+",
            choices=["flip", "rotate", "skew", "shear", "crop", "distortion"],
            help="Apply only the selected augmentations (default: all)"
        )

        return parser


    def build_transformation_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Leaffliction - Transformation"
        )

        parser.add_argument(
            "img_path",
            nargs="?",
            type=str,
            help="Path to an input image (e.g., ./Apple/apple_healthy/image (1).JPG)"
        )

        parser.add_argument(
            "--src",
            type=str,
            help="Path to a directory containing images"
        )

        parser.add_argument(
            "--dst",
            type=str,
            help="Destination directory to save transformed images"
        )

        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Display detailed processing information"
        )

        parser.add_argument(
            "--histogram",
            action="store_true",
            help="Display color histogram"
        )

        parser.add_argument(
            "--formats",
            nargs="+",
            choices=["jpg", "jpeg", "png"],
            default=["jpg"],
            help="Accepted image extensions (batch mode)"
        )

        parser.add_argument(
            "--only",
            nargs="+",
            choices=["blur", "mask", "roi", "analyze", "pseudo"],
            help="Apply only the selected transformations (default: all)"
        )

        return parser



    def build_train_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Leaffliction - Train"
        )

        parser.add_argument(
            "dataset_dir",
            type=str,
            help="Dataset root directory (images are fetched from subdirectories)."
        )


        parser.add_argument(
            "--out-dir", "-o",
            type=str,
            default="training_artifacts",
            help='Directory to save modified images and model artifacts (default: "training_artifacts").'
        )

        parser.add_argument(
            "--out-zip",
            type=str,
            default="train_output.zip",
            help='Output zip filename to generate (default: "train_output.zip").'
        )


        parser.add_argument(
            "--valid-ratio",
            type=float,
            default=0.2,
            help="Validation ratio for the train/valid split (default: 0.2)."
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for splitting/shuffling (default: 42)."
        )


        parser.add_argument(
            "--learning-rate",
            type=float,
            default=0.0314,
            help="Learning rate (default: 0.0314)."
        )

        parser.add_argument(
            "--loss",
            type=str,
            default="cce",
            choices=["cce", "bce", "mse"],
            help="Loss function (default: cce)."
        )

        parser.add_argument(
            "--layers",
            type=int,
            nargs="+",
            help="Hidden layer sizes, e.g. --layers 24 24 24."
        )

        parser.add_argument(
            "--epochs",
            type=int,
            default=70,
            help="Number of training epochs (default: 70)."
        )

        parser.add_argument(
            "--batch-size",
            type=int,
            default=8,
            help="Batch size (default: 8)."
        )

        parser.add_argument(
            "--optimizer",
            type=str,
            default="Adam",
            choices=["SGD", "Adam"],
            help="Optimizer to use (default: Adam)."
        )

        parser.add_argument(
            "--metrics",
            type=str,
            nargs="*",
            default=["Accuracy"],
            help='Metrics to evaluate during training (default: ["Accuracy"]).'
        )

        parser.add_argument(
            "--early-stopping",
            action="store_true",
            help="Enable early stopping."
        )

        parser.add_argument(
            "--patience",
            type=int,
            default=5,
            help="Patience for early stopping (default: 5)."
        )

        parser.add_argument(
            "--min-delta",
            type=float,
            default=0.0,
            help="Minimum improvement to qualify as progress for early stopping (default: 0.0)."
        )

        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Display detailed processing information."
        )

        return parser


    def build_predict_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Leaffliction - Predict"
        )

        parser.add_argument(
            "image_path",
            type=str,
            help="Path to an input image to classify (e.g., ./Apple/apple_healthy/image (1).JPG)"
        )

        parser.add_argument(
            "--bundle-zip",
            type=str,
            default=None,
            help="Path to the training bundle zip produced by train (optional)."
        )

        parser.add_argument(
            "--model-path",
            type=str,
            default=None,
            help="Path to the model/artifacts directory (optional; overrides --bundle-zip if provided)."
        )

        parser.add_argument(
            "--show-transforms",
            action="store_true",
            help="Display intermediate transformed images (bonus)."
        )

        parser.add_argument(
            "--top-k",
            type=int,
            default=1,
            help="Show the top-k predicted classes (default: 1)."
        )

        parser.add_argument(
            "--proba",
            type=str,
            default=None,
            help="Save prediction probabilities/report to a .txt file (optional)."
        )

        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Display detailed processing information."
        )

        return parser
