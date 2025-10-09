"""Feature based rerankers inspired by OpenSearch LTR and Vespa."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from math import exp

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - numpy optional
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import onnxruntime as ort
except Exception:  # pragma: no cover - onnxruntime optional
    ort = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import xgboost
except Exception:  # pragma: no cover - xgboost optional
    xgboost = None  # type: ignore

from .base import BaseReranker, BatchScore
from .features import FeaturePipeline, FeatureVector
from .models import QueryDocumentPair
from .utils import clamp


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def _bounded(value: float) -> float:
    """Ensure score is within valid range [0,1]."""
    return max(0.0, min(1.0, value))


@dataclass(slots=True)
class LambdaMARTModel:
    """Wrapper around LambdaMART/XGBoost style models."""

    name: str = "lambda-mart"
    version: str = "v1"
    feature_order: list[str] = field(default_factory=list)
    coefficients: Mapping[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    booster: xgboost.Booster | None = None  # type: ignore[name-defined]

    def score_many(self, vectors: Sequence[FeatureVector]) -> list[float]:
        if not self.feature_order and vectors:
            self.feature_order = list(vectors[0].values.keys())
        features = [vector.as_ordered(self.feature_order) for vector in vectors]
        if self.booster is not None and xgboost is not None:
            dmatrix = xgboost.DMatrix(features, feature_names=list(self.feature_order))
            predictions = self.booster.predict(dmatrix)
            return [clamp(float(value)) for value in predictions.tolist()]
        return [self._fallback_score(vector) for vector in vectors]

    def score_with_contributions(
        self, vectors: Sequence[FeatureVector]
    ) -> tuple[list[float], list[Mapping[str, float]]]:
        scores = self.score_many(vectors)
        contributions: list[Mapping[str, float]] = []
        for vector, score in zip(vectors, scores, strict=False):
            breakdown: MutableMapping[str, float] = {}
            for feature, value in vector.values.items():
                weight = self.coefficients.get(feature, 0.0)
                breakdown[feature] = float(value) * weight
            breakdown["intercept"] = self.intercept
            breakdown["score"] = score
            contributions.append(breakdown)
        return scores, contributions

    def _fallback_score(self, vector: FeatureVector) -> float:
        value = self.intercept
        for feature, weight in self.coefficients.items():
            value += float(vector.values.get(feature, 0.0)) * weight
        return clamp(_sigmoid(value))


@dataclass(slots=True)
class LTRDataset:
    """Container produced by the training pipeline."""

    features: list[FeatureVector]
    labels: list[float]
    feature_order: Sequence[str]
    groups: list[str] | None = None

    def to_dmatrix(self) -> xgboost.DMatrix | None:  # type: ignore[name-defined]
        if xgboost is None:
            return None
        matrix = [vector.as_ordered(self.feature_order) for vector in self.features]
        dmatrix = xgboost.DMatrix(matrix, label=self.labels, feature_names=list(self.feature_order))
        if self.groups:
            dmatrix.set_group([self.groups.count(group) for group in set(self.groups)])
        return dmatrix


@dataclass(slots=True)
class LTRTrainingPipeline:
    """Utility building datasets for LambdaMART style training."""

    feature_pipeline: FeaturePipeline
    label_getter: Callable[[QueryDocumentPair], float]
    group_getter: Callable[[QueryDocumentPair], str] | None = None

    def build_dataset(self, pairs: Sequence[QueryDocumentPair]) -> LTRDataset:
        vectors = [
            FeatureVector(doc_id=pair.doc_id, values=self.feature_pipeline.extract(pair))
            for pair in pairs
        ]
        labels = [float(self.label_getter(pair)) for pair in pairs]
        groups = [self.group_getter(pair) for pair in pairs] if self.group_getter else None
        return LTRDataset(
            features=vectors,
            labels=labels,
            feature_order=self.feature_pipeline.feature_names(),
            groups=groups,
        )


class OpenSearchLTRReranker(BaseReranker):
    """Learning-to-rank reranker compatible with OpenSearch sltr plugin."""

    def __init__(
        self,
        feature_pipeline: FeaturePipeline | None = None,
        model: LambdaMARTModel | None = None,
        *,
        feature_store: str = "medical-ltr",
        feature_set: str = "biomedical-default",
    ) -> None:
        self.feature_pipeline = feature_pipeline or FeaturePipeline.default()
        self.model = model or LambdaMARTModel()
        if not self.model.feature_order:
            self.model.feature_order = self.feature_pipeline.feature_names()
        self.feature_store = feature_store
        self.feature_set = feature_set
        super().__init__(
            identifier="opensearch-ltr",
            model_version=self.model.version,
            batch_size=32,
            requires_gpu=False,
        )

    def _score_batch(
        self, batch: Sequence[QueryDocumentPair], *, explain: bool = False
    ) -> BatchScore:
        vectors = [
            FeatureVector(doc_id=pair.doc_id, values=self.feature_pipeline.extract(pair))
            for pair in batch
        ]
        scores, contributions = self.model.score_with_contributions(vectors)
        metadata = None
        if explain:
            metadata = [
                {
                    "features": vector.values,
                    "contributions": breakdown,
                    "model": self.model.name,
                    "version": self.model.version,
                }
                for vector, breakdown in zip(vectors, contributions, strict=False)
            ]
        return BatchScore(scores=scores, extra_metadata=metadata)

    def build_sltr_query(
        self,
        query: str,
        *,
        doc_ids: Sequence[str],
        size: int | None = None,
    ) -> Mapping[str, object]:
        """Return an OpenSearch query using the stored feature set."""
        return {
            "size": size or len(doc_ids),
            "query": {
                "bool": {
                    "filter": [{"terms": {"_id": list(doc_ids)}}],
                    "must": {
                        "match": {
                            "_all": query,
                        }
                    },
                }
            },
            "rescore": {
                "window_size": size or len(doc_ids),
                "query": {
                    "score_mode": "total",
                    "rescore_query": {
                        "sltr": {
                            "params": {"keywords": query},
                            "model": {
                                "stored": {
                                    "store": self.feature_store,
                                    "name": self.feature_set,
                                }
                            },
                        }
                    },
                },
            },
        }

    def schema(self) -> Mapping[str, Sequence[str]]:
        return {"features": self.feature_pipeline.feature_names()}

    @classmethod
    def training_pipeline(
        cls,
        label_getter: Callable[[QueryDocumentPair], float],
        group_getter: Callable[[QueryDocumentPair], str] | None = None,
    ) -> LTRTrainingPipeline:
        return LTRTrainingPipeline(
            feature_pipeline=FeaturePipeline.default(),
            label_getter=label_getter,
            group_getter=group_getter,
        )


@dataclass(slots=True)
class VespaRankProfile:
    """Structured representation of a Vespa rank profile."""

    name: str
    first_phase: str = "nativeRank"
    second_phase: str | None = None
    onnx_model: str | None = None

    def to_dict(self) -> Mapping[str, object]:
        profile: MutableMapping[str, object] = {
            "name": self.name,
            "firstPhase": {"expression": self.first_phase},
        }
        if self.second_phase:
            profile["secondPhase"] = {"expression": self.second_phase}
        if self.onnx_model:
            profile["onnx"] = {"model": self.onnx_model}
        return profile


class VespaRankProfileReranker(BaseReranker):
    """Reranker that produces scores aligned with Vespa rank profiles."""

    def __init__(
        self,
        profile: VespaRankProfile | None = None,
        feature_pipeline: FeaturePipeline | None = None,
        model: LambdaMARTModel | None = None,
    ) -> None:
        self.profile = profile or VespaRankProfile(name="biomedical_ranker_v1")
        self.feature_pipeline = feature_pipeline or FeaturePipeline.default()
        self.model = model or LambdaMARTModel(name="vespa-ltr")
        if not self.model.feature_order:
            self.model.feature_order = self.feature_pipeline.feature_names()
        super().__init__(
            identifier=f"vespa:{self.profile.name}",
            model_version=self.model.version,
            batch_size=16,
            requires_gpu=False,
        )
        self._onnx_session: ort.InferenceSession | None = None  # type: ignore[name-defined]
        self._onnx_input: str | None = None

    def attach_onnx_model(self, model_path: str, input_name: str = "input") -> None:
        if ort is None or np is None:
            raise RuntimeError("onnxruntime and numpy are required for ONNX execution")
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._onnx_session = session
        self._onnx_input = input_name
        self.profile.onnx_model = model_path

    def with_second_phase(self, expression: str) -> None:
        self.profile.second_phase = expression

    def profile_definition(self) -> Mapping[str, object]:
        return self.profile.to_dict()

    def build_deployment_package(self) -> Mapping[str, object]:
        return {
            "models": [self.profile.onnx_model] if self.profile.onnx_model else [],
            "rank_profiles": [self.profile.to_dict()],
            "feature_names": self.feature_pipeline.feature_names(),
        }

    def _score_batch(
        self, batch: Sequence[QueryDocumentPair], *, explain: bool = False
    ) -> BatchScore:
        feature_order = self.feature_pipeline.feature_names()
        vectors = [
            FeatureVector(doc_id=pair.doc_id, values=self.feature_pipeline.extract(pair))
            for pair in batch
        ]
        if self._onnx_session is not None and np is not None and self._onnx_input is not None:
            matrix = np.array([vector.as_ordered(feature_order) for vector in vectors], dtype=np.float32)
            outputs = self._onnx_session.run(None, {self._onnx_input: matrix})
            scores = [_bounded(float(value)) for value in outputs[0].reshape(-1)]
            contributions = [vector.values for vector in vectors]
        else:
            scores, contributions = self.model.score_with_contributions(vectors)
        metadata = None
        if explain:
            metadata = [
                {
                    "features": vector.values,
                    "contributions": breakdown,
                    "profile": self.profile.to_dict(),
                }
                for vector, breakdown in zip(vectors, contributions, strict=False)
            ]
        return BatchScore(scores=scores, extra_metadata=metadata)
