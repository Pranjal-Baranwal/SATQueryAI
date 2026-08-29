"""
SATQueryAI Routing State
========================

Defines the shared state passed between nodes of the
SATQueryAI routing pipeline.

Pipeline:

    User Query
        ↓
    Intent Classifier
        ↓
    Router
        ↓
    Model Execution
        ↓
    Evidence Generator
        ↓
    Confidence Estimator
        ↓
    Final Answer
"""


from typing import Any, Dict, List, Optional, TypedDict


# ============================================================
# ROUTING STATE
# ============================================================

class RoutingState(TypedDict, total=False):
    """
    Shared state for the SATQueryAI routing pipeline.

    All fields are optional so individual nodes can gradually
    populate the state.
    """

    # --------------------------------------------------------
    # USER INPUT
    # --------------------------------------------------------

    query: str

    # --------------------------------------------------------
    # INPUT DATA
    # --------------------------------------------------------

    optical_path: Optional[str]

    sar_path: Optional[str]

    image_path: Optional[str]

    t1_path: Optional[str]

    t2_path: Optional[str]

    # --------------------------------------------------------
    # INTENT
    # --------------------------------------------------------

    intent: Optional[str]

    intent_confidence: Optional[float]

    intent_reason: Optional[str]

    # --------------------------------------------------------
    # ROUTING
    # --------------------------------------------------------

    route: Optional[str]

    routes: List[str]

    routing_confidence: Optional[float]

    routing_reason: Optional[str]

    # --------------------------------------------------------
    # MODEL OUTPUTS
    # --------------------------------------------------------

    sia_result: Optional[Dict[str, Any]]

    sia_answer: Optional[str]

    bta_result: Optional[Dict[str, Any]]

    bta_answer: Optional[str]

    cross_modal_result: Optional[Dict[str, Any]]

    cross_modal_answer: Optional[str]

    # --------------------------------------------------------
    # EVIDENCE
    # --------------------------------------------------------

    evidence: Optional[Dict[str, Any]]

    evidence_items: List[Dict[str, Any]]

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence: Optional[Dict[str, Any]]

    confidence_score: Optional[float]

    confidence_level: Optional[str]

    # --------------------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------------------

    final_answer: Optional[str]

    final_response: Optional[Dict[str, Any]]

    # --------------------------------------------------------
    # EXECUTION
    # --------------------------------------------------------

    executed_models: List[str]

    execution_status: Optional[str]

    execution_time: Optional[float]

    # --------------------------------------------------------
    # ERROR HANDLING
    # --------------------------------------------------------

    error: Optional[str]

    errors: List[str]

    # --------------------------------------------------------
    # DEBUGGING / METADATA
    # --------------------------------------------------------

    metadata: Dict[str, Any]


# ============================================================
# STATE CREATION
# ============================================================

def create_initial_state(
    query: str,
    image_path: Optional[str] = None,
    optical_path: Optional[str] = None,
    sar_path: Optional[str] = None,
    t1_path: Optional[str] = None,
    t2_path: Optional[str] = None,
) -> RoutingState:
    """
    Create the initial state for a SATQueryAI request.

    Parameters
    ----------
    query:
        User's natural-language query.

    image_path:
        Optional single satellite image.

    optical_path:
        Optional Optical image.

    sar_path:
        Optional SAR image.

    t1_path:
        Optional first temporal image.

    t2_path:
        Optional second temporal image.
    """

    return RoutingState(

        # ----------------------------------------------------
        # User input
        # ----------------------------------------------------

        query=query,

        # ----------------------------------------------------
        # Input data
        # ----------------------------------------------------

        image_path=image_path,

        optical_path=optical_path,

        sar_path=sar_path,

        t1_path=t1_path,

        t2_path=t2_path,

        # ----------------------------------------------------
        # Intent
        # ----------------------------------------------------

        intent=None,

        intent_confidence=None,

        intent_reason=None,

        # ----------------------------------------------------
        # Routing
        # ----------------------------------------------------

        route=None,

        routes=[],

        routing_confidence=None,

        routing_reason=None,

        # ----------------------------------------------------
        # Model outputs
        # ----------------------------------------------------

        sia_result=None,

        sia_answer=None,

        bta_result=None,

        bta_answer=None,

        cross_modal_result=None,

        cross_modal_answer=None,

        # ----------------------------------------------------
        # Evidence
        # ----------------------------------------------------

        evidence=None,

        evidence_items=[],

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence=None,

        confidence_score=None,

        confidence_level=None,

        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

        final_answer=None,

        final_response=None,

        # ----------------------------------------------------
        # Execution
        # ----------------------------------------------------

        executed_models=[],

        execution_status="initialized",

        execution_time=None,

        # ----------------------------------------------------
        # Errors
        # ----------------------------------------------------

        error=None,

        errors=[],

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata={},
    )


# ============================================================
# STATE UPDATE HELPERS
# ============================================================

def update_state(
    state: RoutingState,
    **updates
) -> RoutingState:
    """
    Update selected state fields.

    Returns a new state dictionary rather than modifying
    the original object in-place.
    """

    new_state = dict(state)

    new_state.update(
        updates
    )

    return RoutingState(
        **new_state
    )


# ============================================================
# ADD EXECUTED MODEL
# ============================================================

def add_executed_model(
    state: RoutingState,
    model_name: str
) -> RoutingState:
    """
    Add a model to the executed-model list.
    """

    new_state = dict(state)

    executed_models = list(
        state.get(
            "executed_models",
            []
        )
    )

    if model_name not in executed_models:

        executed_models.append(
            model_name
        )

    new_state[
        "executed_models"
    ] = executed_models

    return RoutingState(
        **new_state
    )


# ============================================================
# ADD ERROR
# ============================================================

def add_error(
    state: RoutingState,
    error_message: str
) -> RoutingState:
    """
    Add an error to the state while preserving previous errors.
    """

    new_state = dict(state)

    errors = list(
        state.get(
            "errors",
            []
        )
    )

    errors.append(
        error_message
    )

    new_state[
        "errors"
    ] = errors

    new_state[
        "error"
    ] = error_message

    new_state[
        "execution_status"
    ] = "error"

    return RoutingState(
        **new_state
    )


# ============================================================
# MARK COMPLETE
# ============================================================

def mark_complete(
    state: RoutingState
) -> RoutingState:
    """
    Mark the routing pipeline as successfully completed.
    """

    new_state = dict(state)

    new_state[
        "execution_status"
    ] = "completed"

    return RoutingState(
        **new_state
    )


# ============================================================
# MARK RUNNING
# ============================================================

def mark_running(
    state: RoutingState
) -> RoutingState:
    """
    Mark the routing pipeline as running.
    """

    new_state = dict(state)

    new_state[
        "execution_status"
    ] = "running"

    return RoutingState(
        **new_state
    )


# ============================================================
# STATE VALIDATION
# ============================================================

def validate_state(
    state: RoutingState
) -> List[str]:
    """
    Validate the minimum information required to execute
    the routing pipeline.

    Returns:
        List of validation errors.

    An empty list means the state is valid.
    """

    errors = []

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    query = state.get(
        "query"
    )

    if not query:

        errors.append(
            "User query is missing."
        )

    elif not isinstance(
        query,
        str
    ):

        errors.append(
            "User query must be a string."
        )

    elif not query.strip():

        errors.append(
            "User query cannot be empty."
        )

    # --------------------------------------------------------
    # Image availability
    # --------------------------------------------------------

    image_available = any(
        [
            state.get("image_path"),
            state.get("optical_path"),
            state.get("sar_path"),
            state.get("t1_path"),
            state.get("t2_path"),
        ]
    )

    if not image_available:

        errors.append(
            "No satellite image input was provided."
        )

    return errors


# ============================================================
# STATE SUMMARY
# ============================================================

def state_summary(
    state: RoutingState
) -> Dict[str, Any]:
    """
    Return a compact state summary useful for debugging.
    """

    return {

        "query": state.get(
            "query"
        ),

        "intent": state.get(
            "intent"
        ),

        "intent_confidence": state.get(
            "intent_confidence"
        ),

        "route": state.get(
            "route"
        ),

        "routes": state.get(
            "routes",
            []
        ),

        "routing_confidence": state.get(
            "routing_confidence"
        ),

        "executed_models": state.get(
            "executed_models",
            []
        ),

        "confidence_score": state.get(
            "confidence_score"
        ),

        "confidence_level": state.get(
            "confidence_level"
        ),

        "execution_status": state.get(
            "execution_status"
        ),

        "error": state.get(
            "error"
        ),
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n========== ROUTING STATE TEST =========="
    )

    # --------------------------------------------------------
    # Create state
    # --------------------------------------------------------

    state = create_initial_state(

        query=(
            "What changed between the "
            "two satellite images?"
        ),

        t1_path=(
            "data/input/T1.png"
        ),

        t2_path=(
            "data/input/T2.png"
        ),
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validation_errors = (
        validate_state(
            state
        )
    )

    if validation_errors:

        print(
            "\nValidation errors:"
        )

        for error in validation_errors:

            print(
                f"- {error}"
            )

    else:

        print(
            "\nState validation: PASSED"
        )

    # --------------------------------------------------------
    # Simulate routing
    # --------------------------------------------------------

    state = update_state(
        state,
        intent="bi_temporal_change",
        intent_confidence=0.96,
        intent_reason=(
            "The query asks what changed "
            "between two images."
        ),
        route="BTA",
        routes=["BTA"],
        routing_confidence=0.96,
        routing_reason=(
            "Two temporal images were provided "
            "and the query explicitly asks about change."
        ),
    )

    # --------------------------------------------------------
    # Simulate execution
    # --------------------------------------------------------

    state = mark_running(
        state
    )

    state = add_executed_model(
        state,
        "BTA"
    )

    # --------------------------------------------------------
    # Simulate result
    # --------------------------------------------------------

    state = update_state(
        state,
        bta_answer=(
            "A new building and road development "
            "are visible in T2."
        ),
        confidence_score=0.82,
        confidence_level="High",
    )

    state = mark_complete(
        state
    )

    # --------------------------------------------------------
    # Display summary
    # --------------------------------------------------------

    summary = state_summary(
        state
    )

    print(
        "\nState Summary:"
    )

    for key, value in summary.items():

        print(
            f"{key}: {value}"
        )

    print(
        "\n========================================"
    )