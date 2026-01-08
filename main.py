"""
Evo2 Variant Pathogenicity Prediction API

This module provides a production-grade API for predicting the pathogenicity
of genetic variants using the Evo2 deep learning model. Supports SNVs,
deletions, and insertions across multiple genome assemblies.
"""

import os
from enum import Enum
from typing import Optional

import modal
import requests
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

# ============================================================================
# Modal Image Configuration
# ============================================================================

evo2_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install([
        "build-essential", "cmake", "ninja-build",
        "libcudnn8", "libcudnn8-dev", "git", "gcc", "g++"
    ])
    .env({
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
    })
    .run_commands("pip install wheel setuptools packaging")
    .run_commands("pip install evo2")
    .run_commands("pip install 'transformer_engine[pytorch]' --no-build-isolation")
    .run_commands("pip install flash-attn --no-build-isolation")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("evo2/notebooks", "/evo2/notebooks")
)

# ============================================================================
# Modal App Configuration
# ============================================================================

app = modal.App("extended-evo2-snv-pathogenicity", image=evo2_image)

# Persistent volume for HuggingFace model cache
volume = modal.Volume.from_name("hf_cache", create_if_missing=True)
MOUNT_PATH = "/root/.cache/huggingface"

# ============================================================================
# Data Models
# ============================================================================

class MutationType(str, Enum):
    """Supported mutation types for variant analysis."""
    SNV = "SNV"  # Single Nucleotide Variant
    DELETION = "DELETION"  # Deletion
    INSERTION = "INSERTION"  # Insertion


class VariantRequest(BaseModel):
    """Request model for variant analysis endpoint."""
    variant_position: int  # 1-based genomic position
    alternative: str  # Alternative allele (nucleotide(s), "-", or empty string for deletion)
    genome: str  # Genome assembly (e.g., "hg38", "hg19")
    chromosome: str  # Chromosome (e.g., "chr17", "chr1")
    mutation_type: MutationType = MutationType.SNV  # Type of mutation
    reference: Optional[str] = None  # Optional reference allele

    class Config:
        """Pydantic configuration for VariantRequest model."""

        # Allow empty strings for alternative field (needed for DELETION mutations)
        # Pydantic by default accepts empty strings, but this makes it explicit
        pass

# ============================================================================
# Helper Functions
# ============================================================================

def get_genome_sequence(
    position: int, genome: str, chromosome: str, window_size: int = 8192
) -> tuple[str, int]:
    """
    Fetch genome sequence window from UCSC Genome Browser API.

    Args:
        position: 1-based genomic position
        genome: Genome assembly identifier (e.g., "hg38")
        chromosome: Chromosome identifier (e.g., "chr17")
        window_size: Size of sequence window to fetch (default: 8192 bp)

    Returns:
        Tuple of (sequence, start_position) where start_position is 0-based

    Raises:
        HTTPException: If API request fails or returns invalid data
    """
    half_window = window_size // 2
    start = max(0, position - 1 - half_window)  # Convert to 0-based
    end = position - 1 + half_window + 1

    print(
        f"Fetching {window_size}bp window around position {position} "
        f"from UCSC API..."
    )
    print(f"Coordinates: {chromosome}:{start}-{end} ({genome})")

    api_url = (
        f"https://api.genome.ucsc.edu/getData/sequence?"
        f"genome={genome};chrom={chromosome};start={start};end={end}"
    )
    response = requests.get(api_url)

    if response.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to fetch genome sequence from UCSC API: "
                f"{response.status_code}"
            ),
        )

    genome_data = response.json()

    if "dna" not in genome_data:
        error = genome_data.get("error", "Unknown error")
        raise HTTPException(
            status_code=500,
            detail=f"UCSC API error: {error}"
        )

    sequence = genome_data.get("dna", "").upper()
    expected_length = end - start

    if len(sequence) != expected_length:
        print(
            f"Warning: received sequence length ({len(sequence)}) "
            f"differs from expected ({expected_length})"
        )

    print(
        f"Loaded reference genome sequence window "
        f"(length: {len(sequence)} bases)"
    )

    return sequence, start


def analyze_variant(
    relative_pos_in_window: int,
    reference: str,
    alternative: str,
    window_seq: str,
    model,
    mutation_type: MutationType = MutationType.SNV,
) -> dict:
    """
    Analyze a variant by comparing reference and variant sequence scores.

    Constructs the variant sequence based on mutation type, scores both reference
    and variant sequences using the Evo2 model, and calculates delta score and
    pathogenicity prediction.

    Args:
        relative_pos_in_window: Position of variant within the window (0-indexed)
        reference: Reference nucleotide(s) at the position
        alternative: Alternative nucleotide(s) or empty string for deletion
        window_seq: The reference sequence window (8192 bp)
        model: Evo2 model instance
        mutation_type: Type of mutation (SNV, DELETION, INSERTION)

    Returns:
        Dictionary containing:
            - reference: Reference allele
            - alternative: Alternative allele
            - delta_score: Difference in log-likelihood scores (variant - ref)
            - prediction: "Likely pathogenic" or "Likely benign"
            - classification_confidence: Confidence score (0.0 to 1.0)
            - mutation_type: Type of mutation analyzed

    Raises:
        ValueError: If mutation type is unsupported
    """
    # Build variant sequence based on mutation type
    if mutation_type == MutationType.SNV:
        # Single nucleotide substitution: replace one base
        var_seq = (
            window_seq[:relative_pos_in_window]
            + alternative
            + window_seq[relative_pos_in_window + 1:]
        )

    elif mutation_type == MutationType.DELETION:
        # Deletion: remove reference nucleotide(s)
        del_length = len(reference) if reference else 1
        var_seq = (
            window_seq[:relative_pos_in_window]
            + window_seq[relative_pos_in_window + del_length:]
        )

    elif mutation_type == MutationType.INSERTION:
        # Insertion: insert alternative sequence after reference position
        var_seq = (
            window_seq[:relative_pos_in_window + 1]
            + alternative
            + window_seq[relative_pos_in_window + 1:]
        )

    else:
        raise ValueError(f"Unsupported mutation type: {mutation_type}")

    # Score both reference and variant sequences
    ref_score = model.score_sequences([window_seq])[0]
    var_score = model.score_sequences([var_seq])[0]

    # Calculate delta score (negative = loss of function)
    delta_score = var_score - ref_score

    # Classification threshold and standard deviations from BRCA1 training data
    # These values were determined by running brca1_example.remote()
    threshold = -0.0009178519
    lof_std = 0.0015140239  # Loss of function standard deviation
    func_std = 0.0009016589  # Functional standard deviation

    # Classify variant based on delta score
    if delta_score < threshold:
        prediction = "Likely pathogenic"
        confidence = min(1.0, abs(delta_score - threshold) / lof_std)
    else:
        prediction = "Likely benign"
        confidence = min(1.0, abs(delta_score - threshold) / func_std)

    return {
        "reference": reference,
        "alternative": alternative,
        "delta_score": float(delta_score),
        "prediction": prediction,
        "classification_confidence": float(confidence),
        "mutation_type": mutation_type.value,
    }

# ============================================================================
# Authentication
# ============================================================================

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(
    api_key: Optional[str] = Security(API_KEY_HEADER),
) -> bool:
    """
    Verify API key from request header against environment variable.

    If MODAL_API_KEY environment variable is set, requires valid API key in request.
    If not set, allows requests without authentication (development mode).

    Args:
        api_key: API key from X-API-Key header (optional)

    Returns:
        True if authentication passes

    Raises:
        HTTPException: 401 if API key required but missing
        HTTPException: 403 if API key is invalid
    """
    expected_key = os.environ.get("MODAL_API_KEY")

    # Development mode: allow requests if no API key configured
    if not expected_key:
        return True

    # Production mode: require valid API key
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Please provide X-API-Key header.",
        )

    if api_key != expected_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )

    return True

# ============================================================================
# Main API Class
# ============================================================================

@app.cls(
    gpu="H100",
    volumes={MOUNT_PATH: volume},
    max_containers=3,
    retries=2,
    scaledown_window=120,
)
class Evo2Model:
    """
    Modal class for Evo2 variant analysis API.

    Loads the Evo2 model once on container startup and provides
    a FastAPI endpoint for variant analysis.
    """

    def __init__(self):
        """Initialize Evo2Model instance."""
        self.model = None

    @modal.enter()
    def load_evo2_model(self):
        """Load Evo2 model into memory when container starts."""
        from evo2 import Evo2  # noqa: E402

        print("Loading evo2 model...")
        self.model = Evo2('evo2_7b')
        print("Evo2 model loaded")
    
    @modal.fastapi_endpoint(
        method="POST",
    )
    def analyze_single_variant(
        self,
        request: VariantRequest,
        _api_key: bool = Depends(verify_api_key),  # noqa: ARG002
    ):
        """
        Analyze a single variant for pathogenicity prediction.

        Fetches genome sequence, constructs variant sequence based on mutation type,
        scores both sequences with Evo2 model, and returns pathogenicity prediction.

        Args:
            request: VariantRequest containing variant details
            _api_key: Verified API key (from dependency injection, unused)

        Returns:
            Dictionary containing:
                - position: Genomic position
                - chromosome: Chromosome identifier
                - genome: Genome assembly
                - reference: Reference allele
                - alternative: Alternative allele
                - delta_score: Delta log-likelihood score
                - prediction: Pathogenicity prediction
                - classification_confidence: Confidence score
                - mutation_type: Type of mutation

        Raises:
            HTTPException: 400 for invalid request parameters
            HTTPException: 401 for missing API key (if required)
            HTTPException: 403 for invalid API key
            HTTPException: 500 for server errors
        """
        # Extract request parameters
        variant_position = request.variant_position
        alternative = request.alternative
        genome = request.genome
        chromosome = request.chromosome
        mutation_type = request.mutation_type
        provided_reference = request.reference

        print(f"Analyzing variant: {chromosome}:{variant_position} {genome}")
        print(f"Mutation type: {mutation_type}, Alternative: {alternative}")

        window_size = 8192  # Evo2 model window size

        # Fetch genome sequence window from UCSC API
        window_seq, seq_start = get_genome_sequence(
            position=variant_position,
            genome=genome,
            chromosome=chromosome,
            window_size=window_size,
        )

        print(
            f"Fetched genome sequence window, "
            f"first 100 bases: {window_seq[:100]}"
        )

        # Calculate relative position within fetched window (0-indexed)
        relative_pos = variant_position - 1 - seq_start
        print(f"Relative position within window: {relative_pos}")

        # Validate position is within fetched window
        if relative_pos < 0 or relative_pos >= len(window_seq):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Variant position {variant_position} is outside the "
                    f"fetched window (start={seq_start+1}, "
                    f"end={seq_start+len(window_seq)})"
                ),
            )

        # Determine reference sequence
        if provided_reference:
            reference = provided_reference.upper()
            # Validate reference matches genome sequence for SNV
            if mutation_type == MutationType.SNV:
                if window_seq[relative_pos] != reference:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Provided reference '{reference}' does not match "
                            f"genome sequence '{window_seq[relative_pos]}' "
                            f"at position {variant_position}"
                        ),
                    )
        else:
            # Auto-detect reference from genome sequence
            reference = window_seq[relative_pos]

        print(f"Reference allele: {reference}")

        # Validate and normalize alternative based on mutation type
        if mutation_type == MutationType.SNV:
            if len(alternative) != 1 or alternative.upper() not in "ACGT":
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "For SNV, alternative must be a single nucleotide "
                        "(A, C, G, or T)"
                    ),
                )
            alternative = alternative.upper()

        elif mutation_type == MutationType.DELETION:
            # For deletion, alternative should be "-" or empty
            if alternative and alternative.upper() not in ["-", ""]:
                raise HTTPException(
                    status_code=400,
                    detail="For deletion, alternative should be '-' or empty string",
                )
            alternative = ""  # Normalize to empty string

        elif mutation_type == MutationType.INSERTION:
            # For insertion, alternative must be non-empty nucleotide sequence
            if not alternative or not all(
                c.upper() in "ACGT" for c in alternative
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "For insertion, alternative must be a non-empty sequence "
                        "of nucleotides (A, C, G, T)"
                    ),
                )
            alternative = alternative.upper()

        # Analyze the variant
        result = analyze_variant(
            relative_pos_in_window=relative_pos,
            reference=reference,
            alternative=alternative,
            window_seq=window_seq,
            model=self.model,
            mutation_type=mutation_type,
        )

        # Add metadata to result
        result["position"] = variant_position
        result["chromosome"] = chromosome
        result["genome"] = genome

        return result
