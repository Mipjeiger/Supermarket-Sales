import joblib
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain constants — mirrors the actual directory layout under Models/
# ---------------------------------------------------------------------------
FRAUD_DIR = settings.MODEL_PATH / "fraud_ml_models"
SALES_DIR = settings.MODEL_PATH / "sales_ml_models"
SALES_SCALER_DIR = SALES_DIR / "scaler"


class ModelRegistry:
    """
    Centralized in-memory registry for managing and retrieving trained
    classical ML models.

    Directory layout expected under ``Models/``:
    ├── fraud_ml_models/
    │   ├── GradientBoostingClassifier.pkl
    │   ├── RandomForestClassifier.pkl
    │   └── XGBClassifier.pkl
    └── sales_ml_models/
        ├── CatboostRegressor_model.joblib
        ├── DecisionTreeRegressor_model.joblib
        ├── RandomForestRegressor_model.joblib
        ├── XGBRegressor_model.joblib
        └── scaler/          ← scaler artefact(s) for sales models
    """

    def __init__(self) -> None:
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_pkl(self, path: Path, label: str) -> Optional[Any]:
        """Load a scikit-learn-compatible binary serialised with pickle."""
        if not path.exists():
            logger.warning(f"⚠️  Model file not found: {path}")
            return None
        try:
            with open(path, "rb") as fh:
                obj = pickle.load(fh)
            logger.info(f"✅ Loaded (pickle) '{label}' from {path}")
            return obj
        except Exception as exc:
            logger.error(f"❌ Failed to load (pickle) '{label}': {exc}")
            return None

    def _load_joblib(self, path: Path, label: str) -> Optional[Any]:
        """Load a joblib-serialised artefact (model or scaler)."""
        if not path.exists():
            logger.warning(f"⚠️  Artefact file not found: {path}")
            return None
        try:
            obj = joblib.load(path)
            logger.info(f"✅ Loaded (joblib) '{label}' from {path}")
            return obj
        except Exception as exc:
            logger.error(f"❌ Failed to load (joblib) '{label}': {exc}")
            return None

    # ------------------------------------------------------------------
    # Public: single-model loaders
    # ------------------------------------------------------------------

    def load_fraud_model(self, model_name: str) -> None:
        """
        Load a single fraud model (.pkl) by bare name.
        e.g. ``"XGBClassifier"``  →  ``fraud_ml_models/XGBClassifier.pkl``
        """
        path = FRAUD_DIR / f"{model_name}.pkl"
        obj = self._load_pkl(path, model_name)
        if obj is not None:
            self.models[model_name] = obj

    def load_sales_model(self, model_name: str) -> None:
        """
        Load a single sales model (.joblib) by bare name.
        e.g. ``"CatboostRegressor_model"``
        """
        path = SALES_DIR / f"{model_name}.joblib"
        obj = self._load_joblib(path, model_name)
        if obj is not None:
            self.models[model_name] = obj

    def load_sales_scaler(self, scaler_name: str) -> None:
        """
        Load a scaler artefact from ``sales_ml_models/scaler/``.
        Tries .joblib first, then .pkl.
        """
        for ext, loader in [(".joblib", self._load_joblib), (".pkl", self._load_pkl)]:
            path = SALES_SCALER_DIR / f"{scaler_name}{ext}"
            if path.exists():
                obj = loader(path, scaler_name)
                if obj is not None:
                    self.scalers[scaler_name] = obj
                    return
        logger.warning(f"⚠️  Scaler '{scaler_name}' not found in {SALES_SCALER_DIR}")

    # ------------------------------------------------------------------
    # Bulk loader — called once during FastAPI lifespan startup
    # ------------------------------------------------------------------

    def load_all_models(self) -> None:
        """
        Discover and load every model file from both domain directories.

        • fraud_ml_models/        →  *.pkl    (loaded with pickle)
        • sales_ml_models/        →  *.joblib (loaded with joblib)
        • sales_ml_models/scaler/ →  any .joblib / .pkl scaler artefacts
        """
        # ── Fraud models ────────────────────────────────────────────────
        if not FRAUD_DIR.exists():
            logger.error(f"❌ Fraud model directory missing: {FRAUD_DIR}")
        else:
            for pkl_path in sorted(FRAUD_DIR.glob("*.pkl")):
                model_name = pkl_path.stem          # e.g. "XGBClassifier"
                obj = self._load_pkl(pkl_path, model_name)
                if obj is not None:
                    self.models[model_name] = obj

        # ── Sales models ─────────────────────────────────────────────────
        if not SALES_DIR.exists():
            logger.error(f"❌ Sales model directory missing: {SALES_DIR}")
        else:
            for jbl_path in sorted(SALES_DIR.glob("*.joblib")):
                model_name = jbl_path.stem          # e.g. "CatboostRegressor_model"
                obj = self._load_joblib(jbl_path, model_name)
                if obj is not None:
                    self.models[model_name] = obj

            # ── Sales scalers ──────────────────────────────────────────
            if SALES_SCALER_DIR.exists():
                for scaler_path in sorted(SALES_SCALER_DIR.iterdir()):
                    if scaler_path.suffix in {".joblib", ".pkl"}:
                        scaler_name = scaler_path.stem
                        loader = (
                            self._load_joblib
                            if scaler_path.suffix == ".joblib"
                            else self._load_pkl
                        )
                        obj = loader(scaler_path, scaler_name)
                        if obj is not None:
                            self.scalers[scaler_name] = obj
            else:
                logger.warning(
                    f"⚠️  Sales scaler directory not found: {SALES_SCALER_DIR}"
                )

        logger.info(
            f"📦 ModelRegistry ready — "
            f"{len(self.models)} model(s), {len(self.scalers)} scaler(s) loaded."
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_model(self, model_name: str) -> Optional[Any]:
        """Retrieve a cached model by its bare name (stem of the filename)."""
        model = self.models.get(model_name)
        if model is None:
            logger.warning(f"⚠️  Model '{model_name}' not found in registry.")
        return model

    def get_scaler(self, scaler_name: str) -> Optional[Any]:
        """Retrieve a cached sales scaler by its bare name."""
        scaler = self.scalers.get(scaler_name)
        if scaler is None:
            logger.warning(f"⚠️  Scaler '{scaler_name}' not found in registry.")
        return scaler

    def list_models(self) -> list:
        """Return a sorted list of all currently cached model names."""
        return sorted(self.models.keys())

    def list_scalers(self) -> list:
        """Return a sorted list of all currently cached scaler names."""
        return sorted(self.scalers.keys())


# ---------------------------------------------------------------------------
# Thread-safe module-level singleton — import this everywhere
# ---------------------------------------------------------------------------
model_registry = ModelRegistry()
