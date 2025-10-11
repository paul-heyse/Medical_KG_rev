"""Medical text normalization service for biomedical document processing.

This module provides normalization capabilities for medical text including:
- Hyphenation at line breaks in medical terms
- Unicode harmonization (Greek letters, micro sign vs "u")
- Unit and dosage preservation
- Machine-only field with unit-standardized forms
"""

from dataclasses import dataclass
from enum import Enum
import logging
import re

import unicodedata


logger = logging.getLogger(__name__)


class NormalizationLevel(Enum):
    """Levels of text normalization."""

    MINIMAL = "minimal"  # Only critical medical terms
    STANDARD = "standard"  # Standard medical normalization
    AGGRESSIVE = "aggressive"  # Full normalization including units


@dataclass
class NormalizationResult:
    """Result of text normalization."""

    original_text: str
    normalized_text: str
    machine_text: str
    normalization_applied: list[str]
    confidence_score: float


class MedicalNormalizer:
    """Medical text normalizer for biomedical document processing.

    Handles:
    - Hyphenation at line breaks in medical terms
    - Unicode harmonization (Greek letters, micro sign vs "u")
    - Unit and dosage preservation
    - Machine-only field with unit-standardized forms
    """

    def __init__(self, level: NormalizationLevel = NormalizationLevel.STANDARD):
        """Initialize the medical normalizer.

        Args:
        ----
            level: Normalization level to apply

        """
        self.level = level
        self._init_patterns()
        self._init_replacement_maps()

    def _init_patterns(self) -> None:
        """Initialize regex patterns for normalization."""
        # Hyphenation patterns for medical terms
        self.hyphenation_patterns = [
            # Common medical term hyphenations
            (r"\b(\w+)-\s*\n\s*(\w+)\b", r"\1\2"),  # Line break hyphenation
            (r"\b(\w+)\s*-\s*\n\s*(\w+)\b", r"\1\2"),  # Space-hyphen-line break
            # Specific medical term patterns
            (r"\b(\w+)-\s*\n\s*(pharmacology|therapy|treatment)\b", r"\1\2"),
            (r"\b(\w+)-\s*\n\s*(syndrome|disease|disorder)\b", r"\1\2"),
            (r"\b(\w+)-\s*\n\s*(protein|enzyme|hormone)\b", r"\1\2"),
        ]

        # Unicode normalization patterns
        self.unicode_patterns = [
            # Greek letters
            (r"[αΑ]", "alpha"),
            (r"[βΒ]", "beta"),
            (r"[γΓ]", "gamma"),
            (r"[δΔ]", "delta"),
            (r"[εΕ]", "epsilon"),
            (r"[ζΖ]", "zeta"),
            (r"[ηΗ]", "eta"),
            (r"[θΘ]", "theta"),
            (r"[ιΙ]", "iota"),
            (r"[κΚ]", "kappa"),
            (r"[λΛ]", "lambda"),
            (r"[μΜ]", "mu"),
            (r"[νΝ]", "nu"),
            (r"[ξΞ]", "xi"),
            (r"[οΟ]", "omicron"),
            (r"[πΠ]", "pi"),
            (r"[ρΡ]", "rho"),
            (r"[σΣ]", "sigma"),
            (r"[τΤ]", "tau"),
            (r"[υΥ]", "upsilon"),
            (r"[φΦ]", "phi"),
            (r"[χΧ]", "chi"),
            (r"[ψΨ]", "psi"),
            (r"[ωΩ]", "omega"),
            # Special characters
            (r"[µμ]", "micro"),  # Micro sign
            (r"[°]", "degree"),
            (r"[±]", "plus-minus"),
            (r"[×]", "times"),
            (r"[÷]", "divided-by"),
        ]

        # Unit preservation patterns
        self.unit_patterns = [
            # Common medical units
            (r"\b(\d+(?:\.\d+)?)\s*mg\b", r"\1 mg"),
            (r"\b(\d+(?:\.\d+)?)\s*g\b", r"\1 g"),
            (r"\b(\d+(?:\.\d+)?)\s*kg\b", r"\1 kg"),
            (r"\b(\d+(?:\.\d+)?)\s*ml\b", r"\1 ml"),
            (r"\b(\d+(?:\.\d+)?)\s*l\b", r"\1 l"),
            (r"\b(\d+(?:\.\d+)?)\s*mcg\b", r"\1 mcg"),
            (r"\b(\d+(?:\.\d+)?)\s*IU\b", r"\1 IU"),
            (r"\b(\d+(?:\.\d+)?)\s*U\b", r"\1 U"),
            (r"\b(\d+(?:\.\d+)?)\s*mEq\b", r"\1 mEq"),
            (r"\b(\d+(?:\.\d+)?)\s*mol\b", r"\1 mol"),
            (r"\b(\d+(?:\.\d+)?)\s*mmol\b", r"\1 mmol"),
            (r"\b(\d+(?:\.\d+)?)\s*μmol\b", r"\1 μmol"),
            (r"\b(\d+(?:\.\d+)?)\s*nmol\b", r"\1 nmol"),
            (r"\b(\d+(?:\.\d+)?)\s*pmol\b", r"\1 pmol"),
            # Dosage patterns
            (r"\b(\d+(?:\.\d+)?)\s*mg/kg\b", r"\1 mg/kg"),
            (r"\b(\d+(?:\.\d+)?)\s*mg/m²\b", r"\1 mg/m²"),
            (r"\b(\d+(?:\.\d+)?)\s*IU/kg\b", r"\1 IU/kg"),
            (r"\b(\d+(?:\.\d+)?)\s*IU/m²\b", r"\1 IU/m²"),
        ]

    def _init_replacement_maps(self) -> None:
        """Initialize replacement maps for normalization."""
        # Medical term replacements
        self.medical_replacements = {
            # Common medical abbreviations
            "vs": "versus",
            "w/": "with",
            "w/o": "without",
            "b/c": "because",
            "b/w": "between",
            "w/i": "within",
            "w/o": "without",
            # Medical units
            "mcg": "microgram",
            "mg": "milligram",
            "g": "gram",
            "kg": "kilogram",
            "ml": "milliliter",
            "l": "liter",
            "IU": "international unit",
            "U": "unit",
            "mEq": "milliequivalent",
            "mol": "mole",
            "mmol": "millimole",
            "μmol": "micromole",
            "nmol": "nanomole",
            "pmol": "picomole",
        }

        # Unit standardization map
        self.unit_standardization = {
            "mcg": "μg",
            "microgram": "μg",
            "milligram": "mg",
            "gram": "g",
            "kilogram": "kg",
            "milliliter": "ml",
            "liter": "l",
            "international unit": "IU",
            "unit": "U",
            "milliequivalent": "mEq",
            "mole": "mol",
            "millimole": "mmol",
            "micromole": "μmol",
            "nanomole": "nmol",
            "picomole": "pmol",
        }

    def normalize_text(self, text: str) -> NormalizationResult:
        """Normalize medical text according to configured level.

        Args:
        ----
            text: Input text to normalize

        Returns:
        -------
            NormalizationResult with original, normalized, and machine text

        """
        if not text or not text.strip():
            return NormalizationResult(
                original_text=text,
                normalized_text=text,
                machine_text=text,
                normalization_applied=[],
                confidence_score=1.0,
            )

        original_text = text
        normalized_text = text
        machine_text = text
        normalization_applied = []

        try:
            # Step 1: Fix hyphenation at line breaks
            if self.level in [NormalizationLevel.STANDARD, NormalizationLevel.AGGRESSIVE]:
                normalized_text, applied = self._fix_hyphenation(normalized_text)
                normalization_applied.extend(applied)

            # Step 2: Unicode harmonization
            if self.level in [NormalizationLevel.STANDARD, NormalizationLevel.AGGRESSIVE]:
                normalized_text, applied = self._harmonize_unicode(normalized_text)
                normalization_applied.extend(applied)

            # Step 3: Unit preservation
            normalized_text, applied = self._preserve_units(normalized_text)
            normalization_applied.extend(applied)

            # Step 4: Create machine-only text with unit standardization
            if self.level == NormalizationLevel.AGGRESSIVE:
                machine_text, applied = self._create_machine_text(normalized_text)
                normalization_applied.extend(applied)
            else:
                machine_text = normalized_text

            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                original_text, normalized_text, normalization_applied
            )

            return NormalizationResult(
                original_text=original_text,
                normalized_text=normalized_text,
                machine_text=machine_text,
                normalization_applied=normalization_applied,
                confidence_score=confidence_score,
            )

        except Exception as e:
            logger.error(f"Error normalizing text: {e}")
            return NormalizationResult(
                original_text=original_text,
                normalized_text=original_text,
                machine_text=original_text,
                normalization_applied=["error"],
                confidence_score=0.0,
            )

    def _fix_hyphenation(self, text: str) -> tuple[str, list[str]]:
        """Fix hyphenation at line breaks in medical terms."""
        normalized_text = text
        applied = []

        for pattern, replacement in self.hyphenation_patterns:
            if re.search(pattern, normalized_text, re.MULTILINE):
                normalized_text = re.sub(pattern, replacement, normalized_text, flags=re.MULTILINE)
                applied.append(f"hyphenation_fix: {pattern}")

        return normalized_text, applied

    def _harmonize_unicode(self, text: str) -> tuple[str, list[str]]:
        """Harmonize Unicode characters to standard forms."""
        normalized_text = text
        applied = []

        # Apply Unicode normalization
        normalized_text = unicodedata.normalize("NFKC", normalized_text)

        # Apply specific Unicode replacements
        for pattern, replacement in self.unicode_patterns:
            if re.search(pattern, normalized_text):
                normalized_text = re.sub(pattern, replacement, normalized_text)
                applied.append(f"unicode_harmonization: {pattern} -> {replacement}")

        return normalized_text, applied

    def _preserve_units(self, text: str) -> tuple[str, list[str]]:
        """Preserve units and dosages exactly."""
        normalized_text = text
        applied = []

        for pattern, replacement in self.unit_patterns:
            if re.search(pattern, normalized_text):
                normalized_text = re.sub(pattern, replacement, normalized_text)
                applied.append(f"unit_preservation: {pattern}")

        return normalized_text, applied

    def _create_machine_text(self, text: str) -> tuple[str, list[str]]:
        """Create machine-only text with unit standardization."""
        machine_text = text
        applied = []

        # Apply unit standardization
        for original, standardized in self.unit_standardization.items():
            if original in machine_text:
                machine_text = machine_text.replace(original, standardized)
                applied.append(f"unit_standardization: {original} -> {standardized}")

        return machine_text, applied

    def _calculate_confidence_score(
        self, original: str, normalized: str, applied: list[str]
    ) -> float:
        """Calculate confidence score for normalization."""
        if not applied or "error" in applied:
            return 0.0

        # Base confidence
        confidence = 1.0

        # Reduce confidence for aggressive changes
        aggressive_changes = len([a for a in applied if "unicode_harmonization" in a])
        confidence -= aggressive_changes * 0.1

        # Reduce confidence for unit standardization
        unit_changes = len([a for a in applied if "unit_standardization" in a])
        confidence -= unit_changes * 0.05

        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))

    def validate_normalization(self, result: NormalizationResult) -> bool:
        """Validate normalization result quality.

        Args:
        ----
            result: NormalizationResult to validate

        Returns:
        -------
            True if validation passes, False otherwise

        """
        if result.confidence_score < 0.5:
            return False

        # Check for critical medical terms preservation
        critical_terms = ["mg", "g", "kg", "ml", "l", "IU", "U", "mEq", "mol", "mmol"]
        for term in critical_terms:
            if term in result.original_text and term not in result.normalized_text:
                return False

        # Check for reasonable text length preservation
        length_ratio = len(result.normalized_text) / len(result.original_text)
        if length_ratio < 0.8 or length_ratio > 1.2:
            return False

        return True

    def get_normalization_stats(self) -> dict[str, any]:
        """Get normalization statistics."""
        return {
            "level": self.level.value,
            "hyphenation_patterns": len(self.hyphenation_patterns),
            "unicode_patterns": len(self.unicode_patterns),
            "unit_patterns": len(self.unit_patterns),
            "medical_replacements": len(self.medical_replacements),
            "unit_standardization": len(self.unit_standardization),
        }


def normalize_medical_text(
    text: str, level: NormalizationLevel = NormalizationLevel.STANDARD
) -> NormalizationResult:
    """Convenience function for medical text normalization.

    Args:
    ----
        text: Text to normalize
        level: Normalization level

    Returns:
    -------
        NormalizationResult

    """
    normalizer = MedicalNormalizer(level)
    return normalizer.normalize_text(text)
