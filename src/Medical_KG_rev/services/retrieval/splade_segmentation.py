"""SPLADE chunk segmentation utilities.

This module provides utilities for segmenting chunks into SPLADE-compatible
segments with proper tokenization alignment and boundary handling.
"""

import logging

from Medical_KG_rev.services.retrieval.splade_service import SPLADESegment

logger = logging.getLogger(__name__)


class SPLADESegmenter:
    """Segmenter for creating SPLADE-compatible chunks.

    This class handles the segmentation of text chunks into segments that
    are compatible with SPLADE tokenization and processing requirements.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        min_segment_tokens: int = 10,
        preserve_sentence_boundaries: bool = True,
    ):
        """Initialize SPLADE segmenter.

        Args:
            max_tokens: Maximum tokens per segment
            overlap_tokens: Number of tokens to overlap between segments
            min_segment_tokens: Minimum tokens required for a valid segment
            preserve_sentence_boundaries: Whether to preserve sentence boundaries

        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_segment_tokens = min_segment_tokens
        self.preserve_sentence_boundaries = preserve_sentence_boundaries

        logger.info(
            "Initialized SPLADE segmenter",
            extra={
                "max_tokens": self.max_tokens,
                "overlap_tokens": self.overlap_tokens,
                "min_segment_tokens": self.min_segment_tokens,
                "preserve_sentence_boundaries": self.preserve_sentence_boundaries,
            },
        )

    def _count_tokens_fallback(self, text: str) -> int:
        """Fallback token counting using word-based approximation.

        Args:
            text: Text to count tokens for

        Returns:
            Approximate token count

        """
        # Simple word-based approximation (roughly 1.3 tokens per word)
        words = len(text.split())
        return int(words * 1.3)

    def _split_text_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences while preserving medical terminology.

        Args:
            text: Text to split

        Returns:
            List of sentences

        """
        import re

        # Medical-specific sentence splitting patterns
        # Preserve abbreviations like "Dr.", "Prof.", "vs.", "e.g.", "i.e."
        medical_abbreviations = [
            r"\bDr\.",
            r"\bProf\.",
            r"\bvs\.",
            r"\be\.g\.",
            r"\bi\.e\.",
            r"\betc\.",
            r"\bFig\.",
            r"\bTable\.",
            r"\bRef\.",
            r"\bEq\.",
            r"\bmg\.",
            r"\bml\.",
            r"\bkg\.",
            r"\bcm\.",
            r"\bmm\.",
            r"\bμL\.",
            r"\bμM\.",
            r"\bnM\.",
            r"\bpM\.",
            r"\bfM\.",
        ]

        # Create pattern to preserve medical abbreviations
        preserve_pattern = "|".join(medical_abbreviations)

        # Split on sentence boundaries but preserve medical abbreviations
        sentences = re.split(r"(?<!\b(?:" + preserve_pattern + r"))[.!?]+\s+", text)

        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Add back punctuation if it was removed
                if not sentence.endswith((".", "!", "?")):
                    sentence += "."
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    def _split_text_into_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs.

        Args:
            text: Text to split

        Returns:
            List of paragraphs

        """
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if p.strip()]

    def _create_segment(
        self,
        text: str,
        start_char: int,
        end_char: int,
        chunk_id: str,
        segment_index: int,
        token_count: int | None = None,
    ) -> SPLADESegment:
        """Create a SPLADE segment.

        Args:
            text: Segment text content
            start_char: Start character position
            end_char: End character position
            chunk_id: Parent chunk identifier
            segment_index: Index of this segment within the chunk
            token_count: Number of tokens (calculated if not provided)

        Returns:
            SPLADE segment

        """
        if token_count is None:
            token_count = self._count_tokens_fallback(text)

        return SPLADESegment(
            text=text,
            start_char=start_char,
            end_char=end_char,
            token_count=token_count,
            segment_id=f"{chunk_id}_seg_{segment_index}",
        )

    def segment_chunk(
        self,
        chunk_text: str,
        chunk_id: str,
        tokenizer=None,
    ) -> list[SPLADESegment]:
        """Segment a chunk into SPLADE-compatible segments.

        Args:
            chunk_text: Text content of the chunk
            chunk_id: Unique identifier for the chunk
            tokenizer: Optional tokenizer for accurate token counting

        Returns:
            List of SPLADE segments

        """
        segments = []
        char_position = 0

        # Determine token counting method
        if tokenizer is not None:

            def count_tokens(text: str) -> int:
                try:
                    tokens = tokenizer.encode(text, add_special_tokens=True)
                    return len(tokens)
                except Exception as e:
                    logger.warning(
                        "Failed to count tokens with provided tokenizer, using fallback",
                        extra={"error": str(e)},
                    )
                    return self._count_tokens_fallback(text)

        else:
            count_tokens = self._count_tokens_fallback

        # First, try to split by paragraphs if they're not too long
        paragraphs = self._split_text_into_paragraphs(chunk_text)

        for para_idx, paragraph in enumerate(paragraphs):
            para_tokens = count_tokens(paragraph)

            if para_tokens <= self.max_tokens:
                # Paragraph fits in one segment
                segment = self._create_segment(
                    text=paragraph,
                    start_char=char_position,
                    end_char=char_position + len(paragraph),
                    chunk_id=chunk_id,
                    segment_index=len(segments),
                    token_count=para_tokens,
                )
                segments.append(segment)
                char_position += len(paragraph) + 2  # +2 for paragraph separator
            else:
                # Paragraph is too long, split by sentences
                sentences = self._split_text_into_sentences(paragraph)
                current_segment_text = ""
                current_tokens = 0
                segment_start_char = char_position

                for sentence in sentences:
                    sentence_tokens = count_tokens(sentence)

                    # Check if adding this sentence would exceed max_tokens
                    if current_tokens + sentence_tokens > self.max_tokens and current_segment_text:
                        # Create segment from current content
                        segment = self._create_segment(
                            text=current_segment_text.strip(),
                            start_char=segment_start_char,
                            end_char=char_position,
                            chunk_id=chunk_id,
                            segment_index=len(segments),
                            token_count=current_tokens,
                        )
                        segments.append(segment)

                        # Start new segment with overlap if configured
                        if self.overlap_tokens > 0 and current_segment_text:
                            # Find overlap text (last few words)
                            words = current_segment_text.split()
                            overlap_words = (
                                words[-self.overlap_tokens :]
                                if len(words) >= self.overlap_tokens
                                else words
                            )
                            overlap_text = " ".join(overlap_words)

                            current_segment_text = overlap_text + " " + sentence
                            current_tokens = count_tokens(current_segment_text)
                            segment_start_char = char_position - len(overlap_text)
                        else:
                            current_segment_text = sentence
                            current_tokens = sentence_tokens
                            segment_start_char = char_position
                    else:
                        # Add sentence to current segment
                        if current_segment_text:
                            current_segment_text += " " + sentence
                        else:
                            current_segment_text = sentence
                        current_tokens += sentence_tokens

                    char_position += len(sentence) + 1  # +1 for space

                # Add final segment if there's remaining content
                if current_segment_text.strip():
                    segment = self._create_segment(
                        text=current_segment_text.strip(),
                        start_char=segment_start_char,
                        end_char=char_position,
                        chunk_id=chunk_id,
                        segment_index=len(segments),
                        token_count=current_tokens,
                    )
                    segments.append(segment)

                char_position += 2  # +2 for paragraph separator

        # Filter out segments that are too short
        filtered_segments = [seg for seg in segments if seg.token_count >= self.min_segment_tokens]

        logger.info(
            "Chunk segmented for SPLADE",
            extra={
                "chunk_id": chunk_id,
                "original_length": len(chunk_text),
                "segments_count": len(filtered_segments),
                "filtered_out": len(segments) - len(filtered_segments),
            },
        )

        return filtered_segments

    def validate_segments(self, segments: list[SPLADESegment]) -> list[str]:
        """Validate segments for SPLADE compatibility.

        Args:
            segments: List of segments to validate

        Returns:
            List of validation error messages

        """
        errors = []

        for segment in segments:
            # Check token count
            if segment.token_count > self.max_tokens:
                errors.append(
                    f"Segment {segment.segment_id} exceeds max tokens: "
                    f"{segment.token_count} > {self.max_tokens}"
                )

            if segment.token_count < self.min_segment_tokens:
                errors.append(
                    f"Segment {segment.segment_id} below min tokens: "
                    f"{segment.token_count} < {self.min_segment_tokens}"
                )

            # Check text content
            if not segment.text.strip():
                errors.append(f"Segment {segment.segment_id} has empty text")

            # Check character boundaries
            if segment.start_char >= segment.end_char:
                errors.append(
                    f"Segment {segment.segment_id} has invalid boundaries: "
                    f"{segment.start_char} >= {segment.end_char}"
                )

        # Check for overlapping segments
        for i, seg1 in enumerate(segments):
            for j, seg2 in enumerate(segments[i + 1 :], i + 1):
                if seg1.start_char < seg2.end_char and seg2.start_char < seg1.end_char:
                    errors.append(f"Segments {seg1.segment_id} and {seg2.segment_id} overlap")

        return errors

    def get_segmentation_stats(self, segments: list[SPLADESegment]) -> dict:
        """Get statistics about segmentation results.

        Args:
            segments: List of segments to analyze

        Returns:
            Dictionary with segmentation statistics

        """
        if not segments:
            return {
                "total_segments": 0,
                "total_tokens": 0,
                "total_characters": 0,
                "avg_tokens_per_segment": 0,
                "avg_characters_per_segment": 0,
                "min_tokens": 0,
                "max_tokens": 0,
            }

        total_tokens = sum(seg.token_count for seg in segments)
        total_characters = sum(len(seg.text) for seg in segments)

        token_counts = [seg.token_count for seg in segments]

        return {
            "total_segments": len(segments),
            "total_tokens": total_tokens,
            "total_characters": total_characters,
            "avg_tokens_per_segment": total_tokens / len(segments),
            "avg_characters_per_segment": total_characters / len(segments),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "tokens_distribution": {
                "under_100": len([t for t in token_counts if t < 100]),
                "100_300": len([t for t in token_counts if 100 <= t < 300]),
                "300_500": len([t for t in token_counts if 300 <= t < 500]),
                "over_500": len([t for t in token_counts if t >= 500]),
            },
        }
