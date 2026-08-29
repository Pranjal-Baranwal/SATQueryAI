"""
SATQueryAI Router
=================

Determines which SATQueryAI model(s) should handle a request
based on the intent produced by intent_classifier.py.

Routes:

    SIA
        Single Image Analysis

    BTA
        Bi-Temporal Analysis

    Cross-Modal
        Optical + SAR Analysis

    BTA + Cross-Modal
        Complex temporal + multi-modal analysis

The router does NOT load or execute models.
It only determines the execution route.
"""

from typing import Any, Dict, List

from routing.state import (
    RoutingState,
    update_state,
)


# ============================================================
# ROUTE DEFINITIONS
# ============================================================

SIA = "SIA"
BTA = "BTA"
CROSS_MODAL = "Cross-Modal"


# ============================================================
# INTENT → ROUTE MAPPING
# ============================================================

INTENT_ROUTES = {

    # --------------------------------------------------------
    # Single image analysis
    # --------------------------------------------------------

    "single_image_analysis": [
        SIA
    ],

    "image_analysis": [
        SIA
    ],

    "object_detection": [
        SIA
    ],

    "scene_understanding": [
        SIA
    ],

    "land_cover_analysis": [
        SIA
    ],

    # --------------------------------------------------------
    # Bi-temporal analysis
    # --------------------------------------------------------

    "bi_temporal_change": [
        BTA
    ],

    "change_detection": [
        BTA
    ],

    "temporal_analysis": [
        BTA
    ],

    "image_comparison": [
        BTA
    ],

    # --------------------------------------------------------
    # Cross-modal analysis
    # --------------------------------------------------------

    "cross_modal_analysis": [
        CROSS_MODAL
    ],

    "sar_analysis": [
        CROSS_MODAL
    ],

    "optical_sar_analysis": [
        CROSS_MODAL
    ],

    "multi_modal_analysis": [
        CROSS_MODAL
    ],

    # --------------------------------------------------------
    # Complex analysis
    # --------------------------------------------------------

    "complex_analysis": [
        BTA,
        CROSS_MODAL
    ],
}


# ============================================================
# ROUTER
# ============================================================

class Router:
    """
    SATQueryAI routing engine.

    Takes a classified intent and determines the appropriate
    model execution route.
    """

    def __init__(self):

        self.intent_routes = (
            INTENT_ROUTES.copy()
        )

    # ========================================================
    # GET ROUTE
    # ========================================================

    def get_route(
        self,
        intent: str
    ) -> List[str]:
        """
        Return the model route associated with an intent.

        Unknown intents default to SIA.
        """

        if not intent:

            return [SIA]

        normalized_intent = (
            str(intent)
            .strip()
            .lower()
        )

        route = self.intent_routes.get(
            normalized_intent
        )

        if route is None:

            return [SIA]

        return list(route)

    # ========================================================
    # ROUTING CONFIDENCE
    # ========================================================

    def calculate_routing_confidence(
        self,
        state: RoutingState,
        routes: List[str]
    ) -> float:
        """
        Calculate routing confidence.

        The intent classifier's confidence is used as the
        primary signal.

        Additional validation is performed using the available
        input files.
        """

        intent_confidence = state.get(
            "intent_confidence"
        )

        if intent_confidence is None:

            intent_confidence = 0.5

        try:

            intent_confidence = float(
                intent_confidence
            )

        except (
            TypeError,
            ValueError
        ):

            intent_confidence = 0.5

        # ----------------------------------------------------
        # Clamp confidence
        # ----------------------------------------------------

        intent_confidence = max(
            0.0,
            min(
                1.0,
                intent_confidence
            )
        )

        # ----------------------------------------------------
        # Input validation
        # ----------------------------------------------------

        input_bonus = 0.0

        # BTA requires two temporal images

        if BTA in routes:

            t1_exists = bool(
                state.get(
                    "t1_path"
                )
            )

            t2_exists = bool(
                state.get(
                    "t2_path"
                )
            )

            if t1_exists and t2_exists:

                input_bonus = 0.05

            else:

                input_bonus = -0.15

        # Cross-modal requires optical + SAR

        elif CROSS_MODAL in routes:

            optical_exists = bool(
                state.get(
                    "optical_path"
                )
            )

            sar_exists = bool(
                state.get(
                    "sar_path"
                )
            )

            if optical_exists and sar_exists:

                input_bonus = 0.05

            else:

                input_bonus = -0.15

        # SIA requires at least one image

        elif SIA in routes:

            image_exists = any(
                [
                    state.get(
                        "image_path"
                    ),

                    state.get(
                        "optical_path"
                    ),

                    state.get(
                        "t1_path"
                    ),
                ]
            )

            if image_exists:

                input_bonus = 0.05

            else:

                input_bonus = -0.10

        confidence = (
            intent_confidence
            + input_bonus
        )

        return round(
            max(
                0.0,
                min(
                    1.0,
                    confidence
                )
            ),
            4
        )

    # ========================================================
    # ROUTING REASON
    # ========================================================

    def generate_reason(
        self,
        state: RoutingState,
        routes: List[str]
    ) -> str:
        """
        Generate a human-readable explanation for the route.
        """

        intent = state.get(
            "intent",
            "unknown"
        )

        if routes == [SIA]:

            return (
                f"Intent '{intent}' requires "
                "single-image analysis, so the request "
                "was routed to SIA."
            )

        if routes == [BTA]:

            return (
                f"Intent '{intent}' requires "
                "bi-temporal analysis, so the request "
                "was routed to BTA."
            )

        if routes == [CROSS_MODAL]:

            return (
                f"Intent '{intent}' requires "
                "optical and SAR analysis, so the request "
                "was routed to Cross-Modal."
            )

        if (
            BTA in routes
            and CROSS_MODAL in routes
        ):

            return (
                f"Intent '{intent}' requires both "
                "temporal and multi-modal analysis, "
                "so the request was routed to BTA "
                "and Cross-Modal."
            )

        return (
            f"Intent '{intent}' was routed to: "
            f"{', '.join(routes)}."
        )

    # ========================================================
    # ROUTE STATE
    # ========================================================

    def route_state(
        self,
        state: RoutingState
    ) -> RoutingState:
        """
        Determine the route and update RoutingState.
        """

        intent = state.get(
            "intent"
        )

        # ----------------------------------------------------
        # Determine route
        # ----------------------------------------------------

        routes = self.get_route(
            intent
        )

        # ----------------------------------------------------
        # Calculate confidence
        # ----------------------------------------------------

        routing_confidence = (
            self.calculate_routing_confidence(
                state,
                routes
            )
        )

        # ----------------------------------------------------
        # Generate reason
        # ----------------------------------------------------

        routing_reason = (
            self.generate_reason(
                state,
                routes
            )
        )

        # ----------------------------------------------------
        # Primary route
        # ----------------------------------------------------

        primary_route = routes[0]

        # ----------------------------------------------------
        # Update state
        # ----------------------------------------------------

        return update_state(
            state,

            route=primary_route,

            routes=routes,

            routing_confidence=(
                routing_confidence
            ),

            routing_reason=(
                routing_reason
            ),
        )

    # ========================================================
    # ROUTE QUERY
    # ========================================================

    def route_query(
        self,
        state: RoutingState
    ) -> RoutingState:
        """
        Alias for route_state().
        """

        return self.route_state(
            state
        )


# ============================================================
# GLOBAL ROUTER
# ============================================================

router = Router()


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def route_request(
    state: RoutingState
) -> RoutingState:
    """
    Route a SATQueryAI request.
    """

    return router.route_state(
        state
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n========== ROUTER TEST =========="
    )

    # --------------------------------------------------------
    # Test 1: Single image
    # --------------------------------------------------------

    print(
        "\n--- Test 1: SIA ---"
    )

    sia_state = RoutingState(

        query=(
            "What objects are visible "
            "in this satellite image?"
        ),

        intent=(
            "single_image_analysis"
        ),

        intent_confidence=0.95,

        image_path=(
            "data/input/T1.png"
        ),

        routes=[],

        executed_models=[],

        errors=[],

        metadata={},
    )

    sia_state = route_request(
        sia_state
    )

    print(
        "Route:",
        sia_state["route"]
    )

    print(
        "Routes:",
        sia_state["routes"]
    )

    print(
        "Routing confidence:",
        sia_state[
            "routing_confidence"
        ]
    )

    print(
        "Reason:",
        sia_state[
            "routing_reason"
        ]
    )

    # --------------------------------------------------------
    # Test 2: BTA
    # --------------------------------------------------------

    print(
        "\n--- Test 2: BTA ---"
    )

    bta_state = RoutingState(

        query=(
            "What changed between "
            "T1 and T2?"
        ),

        intent=(
            "bi_temporal_change"
        ),

        intent_confidence=0.96,

        t1_path=(
            "data/input/T1.png"
        ),

        t2_path=(
            "data/input/T2.png"
        ),

        routes=[],

        executed_models=[],

        errors=[],

        metadata={},
    )

    bta_state = route_request(
        bta_state
    )

    print(
        "Route:",
        bta_state["route"]
    )

    print(
        "Routes:",
        bta_state["routes"]
    )

    print(
        "Routing confidence:",
        bta_state[
            "routing_confidence"
        ]
    )

    print(
        "Reason:",
        bta_state[
            "routing_reason"
        ]
    )

    # --------------------------------------------------------
    # Test 3: Cross-Modal
    # --------------------------------------------------------

    print(
        "\n--- Test 3: Cross-Modal ---"
    )

    cross_modal_state = RoutingState(

        query=(
            "Analyze the optical and "
            "SAR imagery together."
        ),

        intent=(
            "cross_modal_analysis"
        ),

        intent_confidence=0.94,

        optical_path=(
            "data/input/T1.png"
        ),

        sar_path=(
            "data/input/Sample.tif"
        ),

        routes=[],

        executed_models=[],

        errors=[],

        metadata={},
    )

    cross_modal_state = route_request(
        cross_modal_state
    )

    print(
        "Route:",
        cross_modal_state["route"]
    )

    print(
        "Routes:",
        cross_modal_state["routes"]
    )

    print(
        "Routing confidence:",
        cross_modal_state[
            "routing_confidence"
        ]
    )

    print(
        "Reason:",
        cross_modal_state[
            "routing_reason"
        ]
    )

    # --------------------------------------------------------
    # Test 4: Complex
    # --------------------------------------------------------

    print(
        "\n--- Test 4: Complex ---"
    )

    complex_state = RoutingState(

        query=(
            "What changed between T1 and T2 "
            "and what do the SAR and optical "
            "images indicate?"
        ),

        intent=(
            "complex_analysis"
        ),

        intent_confidence=0.91,

        t1_path=(
            "data/input/T1.png"
        ),

        t2_path=(
            "data/input/T2.png"
        ),

        optical_path=(
            "data/input/T1.png"
        ),

        sar_path=(
            "data/input/Sample.tif"
        ),

        routes=[],

        executed_models=[],

        errors=[],

        metadata={},
    )

    complex_state = route_request(
        complex_state
    )

    print(
        "Primary route:",
        complex_state["route"]
    )

    print(
        "Routes:",
        complex_state["routes"]
    )

    print(
        "Routing confidence:",
        complex_state[
            "routing_confidence"
        ]
    )

    print(
        "Reason:",
        complex_state[
            "routing_reason"
        ]
    )

    print(
        "\n================================="
    )