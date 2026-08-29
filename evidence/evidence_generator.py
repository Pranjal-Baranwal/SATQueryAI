"""
Evidence Generator
==================

Collects outputs from the SATQueryAI analysis modules and
converts them into structured evidence.

Evidence sources:
    - Single Image Analysis (SIA)
    - Bi-Temporal Analysis (BTA)
    - Cross-Modal Analysis
    - Optical Processing
    - SAR Processing
    - Geospatial Alignment

This module does NOT calculate the final confidence score.
That responsibility belongs to confidence.py.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional


# ============================================================
# EVIDENCE STRENGTH
# ============================================================

EVIDENCE_STRENGTH = {
    "strong": 1.0,
    "moderate": 0.7,
    "weak": 0.4,
    "unknown": 0.0,
}


# ============================================================
# EVIDENCE GENERATOR
# ============================================================

class EvidenceGenerator:
    """
    Converts model and geospatial outputs into structured
    evidence for downstream confidence estimation.
    """

    def __init__(self):

        self.evidence = []

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.evidence = []

    # ========================================================
    # ADD EVIDENCE
    # ========================================================

    def add_evidence(
        self,
        source: str,
        observation: str,
        strength: str = "moderate",
        modality: Optional[str] = None,
        category: Optional[str] = None,
        supporting_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Add one structured evidence item.
        """

        strength = strength.lower().strip()

        if strength not in EVIDENCE_STRENGTH:

            strength = "unknown"

        item = {
            "id": len(self.evidence) + 1,
            "source": source,
            "observation": observation,
            "strength": strength,
            "strength_score": EVIDENCE_STRENGTH[
                strength
            ],
            "modality": modality,
            "category": category,
            "supporting_data": (
                supporting_data
                if supporting_data is not None
                else {}
            ),
        }

        self.evidence.append(
            item
        )

        return item

    # ========================================================
    # TEXT EXTRACTION
    # ========================================================

    @staticmethod
    def extract_statements(
        text: Any
    ) -> List[str]:
        """
        Convert model output into individual evidence
        statements.

        Handles:
            - plain strings
            - lists
            - dictionaries
            - None
        """

        if text is None:

            return []

        if isinstance(
            text,
            dict
        ):

            # Prefer answer fields

            for key in [
                "answer",
                "response",
                "analysis",
                "text",
                "result",
            ]:

                if key in text:

                    return EvidenceGenerator.extract_statements(
                        text[key]
                    )

            # Otherwise convert dictionary values

            statements = []

            for value in text.values():

                statements.extend(
                    EvidenceGenerator.extract_statements(
                        value
                    )
                )

            return statements

        if isinstance(
            text,
            list
        ):

            statements = []

            for item in text:

                statements.extend(
                    EvidenceGenerator.extract_statements(
                        item
                    )
                )

            return statements

        if not isinstance(
            text,
            str
        ):

            return []

        text = text.strip()

        if not text:

            return []

        # ----------------------------------------------------
        # Split common bullet/number formats
        # ----------------------------------------------------

        lines = text.splitlines()

        statements = []

        for line in lines:

            line = line.strip()

            if not line:

                continue

            # Remove common bullet prefixes

            prefixes = [
                "- ",
                "* ",
                "• ",
            ]

            for prefix in prefixes:

                if line.startswith(prefix):

                    line = line[
                        len(prefix):
                    ]

                    break

            # Remove numbered prefixes

            if (
                len(line) >= 3
                and line[0].isdigit()
                and line[1] in [".", ")"]
            ):

                line = line[2:].strip()

            elif (
                len(line) >= 4
                and line[0].isdigit()
                and line[1].isdigit()
                and line[2] in [".", ")"]
            ):

                line = line[3:].strip()

            if line:

                statements.append(
                    line
                )

        # If there were no line breaks, use the entire answer

        if not statements:

            statements = [text]

        return statements

    # ========================================================
    # CLASSIFY CATEGORY
    # ========================================================

    @staticmethod
    def classify_category(
        statement: str
    ) -> str:
        """
        Estimate an evidence category from the statement.
        """

        text = statement.lower()

        categories = {
            "vegetation": [
                "vegetation",
                "tree",
                "trees",
                "forest",
                "crop",
                "agriculture",
                "green",
                "plant",
            ],

            "building": [
                "building",
                "buildings",
                "structure",
                "construction",
                "roof",
                "house",
                "campus",
            ],

            "road": [
                "road",
                "roads",
                "highway",
                "street",
                "path",
                "infrastructure",
            ],

            "water": [
                "water",
                "river",
                "lake",
                "pond",
                "reservoir",
                "coast",
            ],

            "land_cover": [
                "land cover",
                "land use",
                "terrain",
                "surface",
                "bare land",
                "soil",
            ],

            "urban": [
                "urban",
                "city",
                "settlement",
                "built-up",
                "built up",
            ],

            "change": [
                "change",
                "changed",
                "increase",
                "decrease",
                "new",
                "removed",
                "cleared",
                "expanded",
                "loss",
                "growth",
            ],

            "environment": [
                "environment",
                "deforestation",
                "erosion",
                "flood",
                "damage",
                "disturbance",
            ],
        }

        for category, keywords in categories.items():

            for keyword in keywords:

                if keyword in text:

                    return category

        return "general"

    # ========================================================
    # ADD MODEL OUTPUT
    # ========================================================

    def add_model_output(
        self,
        source: str,
        output: Any,
        modality: Optional[str] = None,
        strength: str = "moderate",
    ):
        """
        Convert a model output into structured evidence.
        """

        statements = (
            self.extract_statements(
                output
            )
        )

        added = []

        for statement in statements:

            category = (
                self.classify_category(
                    statement
                )
            )

            item = self.add_evidence(
                source=source,
                observation=statement,
                strength=strength,
                modality=modality,
                category=category,
            )

            added.append(
                item
            )

        return added

    # ========================================================
    # ADD SIA
    # ========================================================

    def add_sia(
        self,
        output
    ):

        return self.add_model_output(
            source="SIA",
            output=output,
            modality="optical",
            strength="moderate",
        )

    # ========================================================
    # ADD BTA
    # ========================================================

    def add_bta(
        self,
        output
    ):

        return self.add_model_output(
            source="BTA",
            output=output,
            modality="temporal",
            strength="moderate",
        )

    # ========================================================
    # ADD CROSS-MODAL
    # ========================================================

    def add_cross_modal(
        self,
        output
    ):

        return self.add_model_output(
            source="Cross-Modal",
            output=output,
            modality="optical+sar",
            strength="strong",
        )

    # ========================================================
    # ADD OPTICAL INFORMATION
    # ========================================================

    def add_optical(
        self,
        output
    ):

        return self.add_model_output(
            source="Optical Processing",
            output=output,
            modality="optical",
            strength="moderate",
        )

    # ========================================================
    # ADD SAR INFORMATION
    # ========================================================

    def add_sar(
        self,
        output
    ):

        return self.add_model_output(
            source="SAR Processing",
            output=output,
            modality="sar",
            strength="moderate",
        )

    # ========================================================
    # ADD GEOSPATIAL INFORMATION
    # ========================================================

    def add_geospatial(
        self,
        output
    ):
        """
        Add geospatial validation information.

        Geospatial alignment is given strong evidence strength
        when the source and reference are actually aligned.
        """

        if output is None:

            return []

        # ----------------------------------------------------
        # Handle alignment dictionary
        # ----------------------------------------------------

        if isinstance(
            output,
            dict
        ):

            aligned = output.get(
                "already_aligned",
                output.get(
                    "aligned",
                    None
                )
            )

            if aligned is True:

                return [
                    self.add_evidence(
                        source="Geospatial Alignment",
                        observation=(
                            "Source and reference rasters "
                            "are spatially aligned."
                        ),
                        strength="strong",
                        modality="geospatial",
                        category="alignment",
                        supporting_data=output,
                    )
                ]

            # ------------------------------------------------
            # Generic geospatial result
            # ------------------------------------------------

            return [
                self.add_evidence(
                    source="Geospatial Alignment",
                    observation=(
                        "Geospatial metadata and alignment "
                        "information are available."
                    ),
                    strength="moderate",
                    modality="geospatial",
                    category="alignment",
                    supporting_data=output,
                )
            ]

        return self.add_model_output(
            source="Geospatial Alignment",
            output=output,
            modality="geospatial",
            strength="moderate",
        )

    # ========================================================
    # SOURCE SUMMARY
    # ========================================================

    def source_summary(self):

        summary = {}

        for item in self.evidence:

            source = item[
                "source"
            ]

            if source not in summary:

                summary[source] = 0

            summary[source] += 1

        return summary

    # ========================================================
    # CATEGORY SUMMARY
    # ========================================================

    def category_summary(self):

        summary = {}

        for item in self.evidence:

            category = item[
                "category"
            ]

            if category not in summary:

                summary[category] = 0

            summary[category] += 1

        return summary

    # ========================================================
    # MODALITY SUMMARY
    # ========================================================

    def modality_summary(self):

        summary = {}

        for item in self.evidence:

            modality = item[
                "modality"
            ]

            if modality is None:

                modality = "unknown"

            if modality not in summary:

                summary[modality] = 0

            summary[modality] += 1

        return summary

    # ========================================================
    # CROSS-MODAL AGREEMENT
    # ========================================================

    def calculate_agreement(
        self
    ):
        """
        Estimate whether multiple sources support the same
        categories.

        This is deliberately simple and transparent.
        """

        category_sources = {}

        for item in self.evidence:

            category = item[
                "category"
            ]

            source = item[
                "source"
            ]

            if category not in category_sources:

                category_sources[
                    category
                ] = set()

            category_sources[
                category
            ].add(
                source
            )

        agreement = {}

        for category, sources in (
            category_sources.items()
        ):

            agreement[
                category
            ] = {
                "source_count": len(
                    sources
                ),
                "sources": sorted(
                    sources
                ),
                "multi_source": (
                    len(sources) >= 2
                ),
            }

        return agreement

    # ========================================================
    # BUILD RESULT
    # ========================================================

    def build(
        self,
        query: Optional[str] = None
    ):
        """
        Build the complete structured evidence package.
        """

        agreement = (
            self.calculate_agreement()
        )

        result = {
            "query": query,
            "generated_at": (
                datetime.utcnow()
                .isoformat()
            ),
            "evidence_count": len(
                self.evidence
            ),
            "evidence": self.evidence,
            "source_summary": (
                self.source_summary()
            ),
            "category_summary": (
                self.category_summary()
            ),
            "modality_summary": (
                self.modality_summary()
            ),
            "cross_source_agreement": agreement,
        }

        return result

    # ========================================================
    # EXPORT
    # ========================================================

    def to_dict(
        self,
        query: Optional[str] = None
    ):

        return self.build(
            query=query
        )


# ============================================================
# GLOBAL ENGINE
# ============================================================

evidence_generator = (
    EvidenceGenerator()
)


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def generate_evidence(
    query=None,
    sia=None,
    bta=None,
    cross_modal=None,
    optical=None,
    sar=None,
    geospatial=None,
):
    """
    Generate a complete evidence package.

    All inputs are optional.
    """

    generator = EvidenceGenerator()

    # --------------------------------------------------------
    # Model outputs
    # --------------------------------------------------------

    if sia is not None:

        generator.add_sia(
            sia
        )

    if bta is not None:

        generator.add_bta(
            bta
        )

    if cross_modal is not None:

        generator.add_cross_modal(
            cross_modal
        )

    # --------------------------------------------------------
    # Processing outputs
    # --------------------------------------------------------

    if optical is not None:

        generator.add_optical(
            optical
        )

    if sar is not None:

        generator.add_sar(
            sar
        )

    if geospatial is not None:

        generator.add_geospatial(
            geospatial
        )

    return generator.build(
        query=query
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n========== EVIDENCE GENERATOR TEST =========="
    )

    # --------------------------------------------------------
    # Example model outputs
    # --------------------------------------------------------

    sia_output = (
        "The image contains roads, buildings, "
        "and vegetation."
    )

    bta_output = (
        "A new building appears in the second image. "
        "Vegetation has decreased near the construction area."
    )

    cross_modal_output = (
        "Optical and SAR imagery both indicate "
        "built-up structures and roads."
    )

    # --------------------------------------------------------
    # Example geospatial information
    # --------------------------------------------------------

    geospatial_output = {
        "already_aligned": True,
        "same_crs": True,
        "same_resolution": True,
        "same_dimensions": True,
        "same_transform": True,
        "same_bounds": True,
    }

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    result = generate_evidence(
        query=(
            "What significant changes occurred "
            "in the area?"
        ),
        sia=sia_output,
        bta=bta_output,
        cross_modal=cross_modal_output,
        geospatial=geospatial_output,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        f"\nEvidence count: "
        f"{result['evidence_count']}"
    )

    print(
        "\nSource Summary:"
    )

    for key, value in (
        result["source_summary"].items()
    ):

        print(
            f"{key}: {value}"
        )

    print(
        "\nCategory Summary:"
    )

    for key, value in (
        result["category_summary"].items()
    ):

        print(
            f"{key}: {value}"
        )

    print(
        "\nCross-Source Agreement:"
    )

    for key, value in (
        result[
            "cross_source_agreement"
        ].items()
    ):

        print(
            f"{key}: {value}"
        )

    print(
        "\nEvidence:"
    )

    for item in result[
        "evidence"
    ]:

        print(
            f"\n[{item['id']}] "
            f"{item['source']}"
        )

        print(
            f"Observation: "
            f"{item['observation']}"
        )

        print(
            f"Strength: "
            f"{item['strength']}"
        )

    print(
        "\n=============================================="
    )