"""
SATQueryAI Intent Classifier
============================

Classifies a natural-language satellite imagery query into
an intent that can be handled by the SATQueryAI Router.

Supported intents:

    single_image_analysis
    bi_temporal_change
    cross_modal_analysis
    sar_analysis
    optical_sar_analysis
    image_comparison
    complex_analysis

This is a lightweight rule-based classifier.
It does not load any AI model.
"""

from typing import Dict, List, Tuple, Optional

from routing.state import (
    RoutingState,
    update_state,
)


# ============================================================
# INTENT DEFINITIONS
# ============================================================

SINGLE_IMAGE = "single_image_analysis"

BI_TEMPORAL = "bi_temporal_change"

CROSS_MODAL = "cross_modal_analysis"

SAR_ANALYSIS = "sar_analysis"

OPTICAL_SAR = "optical_sar_analysis"

IMAGE_COMPARISON = "image_comparison"

COMPLEX = "complex_analysis"


# ============================================================
# KEYWORD GROUPS
# ============================================================

CHANGE_KEYWORDS = [
    "change",
    "changed",
    "changes",
    "difference",
    "differences",
    "compare",
    "comparison",
    "before and after",
    "over time",
    "temporal",
    "time series",
    "development",
    "growth",
    "increase",
    "decrease",
    "expansion",
    "expanded",
    "construction",
    "demolition",
    "new building",
    "removed",
    "loss",
]

T1_T2_KEYWORDS = [
    "t1",
    "t2",
    "image 1",
    "image 2",
    "first image",
    "second image",
    "two images",
    "both images",
]

SAR_KEYWORDS = [
    "sar",
    "synthetic aperture radar",
    "radar",
    "microwave",
    "backscatter",
]

OPTICAL_KEYWORDS = [
    "optical",
    "multispectral",
    "rgb",
    "visible spectrum",
    "satellite image",
]

OBJECT_KEYWORDS = [
    "object",
    "objects",
    "building",
    "buildings",
    "road",
    "roads",
    "vehicle",
    "vehicles",
    "tree",
    "trees",
    "vegetation",
    "water",
    "river",
    "lake",
    "pond",
    "structure",
]

SCENE_KEYWORDS = [
    "what is visible",
    "what do you see",
    "what can you see",
    "describe",
    "description",
    "scene",
    "landscape",
    "land cover",
    "land use",
    "area",
    "region",
]

COMPLEX_KEYWORDS = [
    "and",
    "also",
    "together",
    "both",
    "combined",
    "simultaneously",
    "using sar and optical",
    "using optical and sar",
]


# ============================================================
# INTENT CLASSIFIER
# ============================================================

class IntentClassifier:
    """
    Lightweight rule-based intent classifier.
    """

    def __init__(self):

        self.intent_names = [
            SINGLE_IMAGE,
            BI_TEMPORAL,
            CROSS_MODAL,
            SAR_ANALYSIS,
            OPTICAL_SAR,
            IMAGE_COMPARISON,
            COMPLEX,
        ]

    # ========================================================
    # NORMALIZE QUERY
    # ========================================================

    @staticmethod
    def normalize_query(
        query: str
    ) -> str:
        """
        Normalize the user query for keyword matching.
        """

        if not isinstance(
            query,
            str
        ):

            return ""

        return (
            query
            .lower()
            .strip()
        )

    # ========================================================
    # KEYWORD MATCHING
    # ========================================================

    @staticmethod
    def find_matches(
        query: str,
        keywords: List[str]
    ) -> List[str]:
        """
        Return all keywords found in the query.
        """

        matches = []

        for keyword in keywords:

            if keyword in query:

                matches.append(
                    keyword
                )

        return matches

    # ========================================================
    # CLASSIFY
    # ========================================================

    def classify(
        self,
        query: str
    ) -> Dict:
        """
        Classify a natural-language query.

        Returns:
            Dictionary containing:

                intent
                confidence
                reason
                matched_keywords
        """

        query = self.normalize_query(
            query
        )

        if not query:

            return {
                "intent": SINGLE_IMAGE,
                "confidence": 0.20,
                "reason": (
                    "No query was provided. "
                    "Defaulting to single-image analysis."
                ),
                "matched_keywords": [],
            }

        # ----------------------------------------------------
        # Find keyword matches
        # ----------------------------------------------------

        change_matches = self.find_matches(
            query,
            CHANGE_KEYWORDS
        )

        temporal_matches = self.find_matches(
            query,
            T1_T2_KEYWORDS
        )

        sar_matches = self.find_matches(
            query,
            SAR_KEYWORDS
        )

        optical_matches = self.find_matches(
            query,
            OPTICAL_KEYWORDS
        )

        object_matches = self.find_matches(
            query,
            OBJECT_KEYWORDS
        )

        scene_matches = self.find_matches(
            query,
            SCENE_KEYWORDS
        )

        complex_matches = self.find_matches(
            query,
            COMPLEX_KEYWORDS
        )

        # ----------------------------------------------------
        # Flags
        # ----------------------------------------------------

        has_change = bool(
            change_matches
        )

        has_temporal = bool(
            temporal_matches
        )

        has_sar = bool(
            sar_matches
        )

        has_optical = bool(
            optical_matches
        )

        has_objects = bool(
            object_matches
        )

        has_scene = bool(
            scene_matches
        )

        # ====================================================
        # RULE 1
        # TEMPORAL + SAR/OPTICAL
        # ====================================================

        if (
            (has_change or has_temporal)
            and has_sar
            and has_optical
        ):

            return self._result(
                intent=COMPLEX,
                confidence=0.97,
                reason=(
                    "The query combines temporal/change "
                    "analysis with both optical and SAR "
                    "imagery."
                ),
                matches=(
                    change_matches
                    + temporal_matches
                    + sar_matches
                    + optical_matches
                ),
            )

        # ====================================================
        # RULE 2
        # CHANGE + SAR
        # ====================================================

        if (
            (has_change or has_temporal)
            and has_sar
        ):

            return self._result(
                intent=COMPLEX,
                confidence=0.94,
                reason=(
                    "The query requests temporal/change "
                    "analysis together with SAR information."
                ),
                matches=(
                    change_matches
                    + temporal_matches
                    + sar_matches
                ),
            )

        # ====================================================
        # RULE 3
        # CHANGE + OPTICAL
        # ====================================================

        if (
            (has_change or has_temporal)
            and has_optical
        ):

            return self._result(
                intent=BI_TEMPORAL,
                confidence=0.94,
                reason=(
                    "The query requests temporal/change "
                    "analysis using optical imagery."
                ),
                matches=(
                    change_matches
                    + temporal_matches
                    + optical_matches
                ),
            )

        # ====================================================
        # RULE 4
        # EXPLICIT T1/T2 CHANGE
        # ====================================================

        if (
            has_change
            and has_temporal
        ):

            return self._result(
                intent=BI_TEMPORAL,
                confidence=0.98,
                reason=(
                    "The query explicitly asks about "
                    "changes between temporal images."
                ),
                matches=(
                    change_matches
                    + temporal_matches
                ),
            )

        # ====================================================
        # RULE 5
        # TWO IMAGES / COMPARISON
        # ====================================================

        if (
            has_temporal
            or "compare" in query
            or "comparison" in query
            or "difference" in query
        ):

            return self._result(
                intent=IMAGE_COMPARISON,
                confidence=0.90,
                reason=(
                    "The query indicates comparison "
                    "between multiple images."
                ),
                matches=(
                    change_matches
                    + temporal_matches
                ),
            )

        # ====================================================
        # RULE 6
        # OPTICAL + SAR
        # ====================================================

        if (
            has_sar
            and has_optical
        ):

            return self._result(
                intent=OPTICAL_SAR,
                confidence=0.98,
                reason=(
                    "The query explicitly references "
                    "both optical and SAR imagery."
                ),
                matches=(
                    sar_matches
                    + optical_matches
                ),
            )

        # ====================================================
        # RULE 7
        # SAR ONLY
        # ====================================================

        if has_sar:

            return self._result(
                intent=SAR_ANALYSIS,
                confidence=0.95,
                reason=(
                    "The query explicitly requests "
                    "SAR/radar analysis."
                ),
                matches=sar_matches,
            )

        # ====================================================
        # RULE 8
        # OBJECT / SCENE ANALYSIS
        # ====================================================

        if (
            has_objects
            or has_scene
        ):

            return self._result(
                intent=SINGLE_IMAGE,
                confidence=0.90,
                reason=(
                    "The query asks for analysis of "
                    "objects, scene, or land cover "
                    "in a satellite image."
                ),
                matches=(
                    object_matches
                    + scene_matches
                ),
            )

        # ====================================================
        # RULE 9
        # DEFAULT
        # ====================================================

        return self._result(
            intent=SINGLE_IMAGE,
            confidence=0.55,
            reason=(
                "The query does not clearly indicate "
                "temporal or multi-modal analysis, "
                "so it defaults to single-image analysis."
            ),
            matches=[],
        )

    # ========================================================
    # RESULT BUILDER
    # ========================================================

    @staticmethod
    def _result(
        intent: str,
        confidence: float,
        reason: str,
        matches: List[str]
    ) -> Dict:

        return {
            "intent": intent,
            "confidence": round(
                confidence,
                4
            ),
            "reason": reason,
            "matched_keywords": list(
                dict.fromkeys(
                    matches
                )
            ),
        }

    # ========================================================
    # CLASSIFY STATE
    # ========================================================

    def classify_state(
        self,
        state: RoutingState
    ) -> RoutingState:
        """
        Classify the query contained in RoutingState and
        update the state.
        """

        query = state.get(
            "query",
            ""
        )

        result = self.classify(
            query
        )

        return update_state(

            state,

            intent=result[
                "intent"
            ],

            intent_confidence=result[
                "confidence"
            ],

            intent_reason=result[
                "reason"
            ],

            metadata={
                **state.get(
                    "metadata",
                    {}
                ),

                "intent_matches":
                    result[
                        "matched_keywords"
                    ],
            },
        )

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        query: str
    ) -> str:
        """
        Return only the predicted intent.
        """

        return self.classify(
            query
        )[
            "intent"
        ]


# ============================================================
# GLOBAL CLASSIFIER
# ============================================================

intent_classifier = (
    IntentClassifier()
)


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def classify_intent(
    query: str
) -> Dict:

    return intent_classifier.classify(
        query
    )


def classify_state(
    state: RoutingState
) -> RoutingState:

    return intent_classifier.classify_state(
        state
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n========== INTENT CLASSIFIER TEST =========="
    )

    test_queries = [

        # ----------------------------------------------------
        # SIA
        # ----------------------------------------------------

        "What objects are visible in this satellite image?",

        "Describe the land cover in this image.",

        # ----------------------------------------------------
        # BTA
        # ----------------------------------------------------

        "What changed between T1 and T2?",

        "Compare the two satellite images.",

        "Has construction increased between the two images?",

        # ----------------------------------------------------
        # SAR
        # ----------------------------------------------------

        "What does the SAR image show?",

        # ----------------------------------------------------
        # Optical + SAR
        # ----------------------------------------------------

        "Analyze the optical and SAR imagery together.",

        "What information can we get by combining optical and radar data?",

        # ----------------------------------------------------
        # Complex
        # ----------------------------------------------------

        "What changed between T1 and T2 and what do the "
        "optical and SAR images indicate?",
    ]

    for index, query in enumerate(
        test_queries,
        start=1
    ):

        result = classify_intent(
            query
        )

        print(
            f"\n--- Test {index} ---"
        )

        print(
            f"Query: {query}"
        )

        print(
            f"Intent: "
            f"{result['intent']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence']}"
        )

        print(
            f"Reason: "
            f"{result['reason']}"
        )

        print(
            f"Matched keywords: "
            f"{result['matched_keywords']}"
        )

    print(
        "\n============================================="
    )