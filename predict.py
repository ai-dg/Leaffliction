from __future__ import annotations

from pathlib import Path

from leaffliction.cli import CLIBuilder
from leaffliction.predict_pipeline import Predictor, PredictConfig, PredictionVisualiser
from leaffliction.transformations import TransformationEngine
from leaffliction.model import ModelBundle


def main() -> None:
    parser = CLIBuilder().build_predict_parser()
    args = parser.parse_args()

    bundle_zip = Path(args.bundle_zip)
    image_path = Path(args.image_path)

    cfg = PredictConfig(
        show_transforms=getattr(args, "show_transforms", True),
        top_k=getattr(args, "top_k", 1),
    )

    engine = TransformationEngine.default_six()

    # Ici bundle_loader peut être directement ModelBundle si tu exposes une méthode load_from_zip
    predictor = Predictor(bundle_loader=ModelBundle, transformations_engine=engine)

    label, probs = predictor.predict(bundle_zip=bundle_zip, image_path=image_path, cfg=cfg)

    # Affichage (optionnel)
    if cfg.show_transforms:
        vis = PredictionVisualiser()
        # TODO: tu fourniras original + transformed depuis predictor ou via utils
        vis.show(original=None, transformed={}, predicted_label=label)

    # Affichage résultat (toujours)
    print(label)


if __name__ == "__main__":
    main()
