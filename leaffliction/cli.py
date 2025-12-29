from __future__ import annotations

import argparse


class CLIBuilder:
    """
    Centralise argparse pour éviter duplication dans les entrypoints root.
    Tu peux faire 5 méthodes: distribution, augmentation, transformation, train, predict.
    """

    def build_distribution_parser(self) -> argparse.ArgumentParser:
        raise NotImplementedError

    def build_augmentation_parser(self) -> argparse.ArgumentParser:
        raise NotImplementedError

    def build_transformation_parser(self) -> argparse.ArgumentParser:
        raise NotImplementedError

    def build_train_parser(self) -> argparse.ArgumentParser:
        raise NotImplementedError

    def build_predict_parser(self) -> argparse.ArgumentParser:
        raise NotImplementedError
