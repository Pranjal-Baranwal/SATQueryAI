
import time
from typing import Any, Dict, List, Optional

from routing.state import (
    RoutingState,
    create_initial_state,
    validate_state,
    mark_running,
    mark_complete,
    add_error,
    update_state,
    add_executed_model,
)

from routing.intent_classifier import (
    IntentClassifier,
)

from routing.router import (
    Router,
    SIA,
    BTA,
    CROSS_MODAL,
)



class SATQueryGraph:
    """
    Main SATQueryAI orchestration graph.
    """

    def __init__(self):

        self.intent_classifier = (
            IntentClassifier()
        )

        self.router = Router()


    def classify_intent(
        self,
        state: RoutingState
    ) -> RoutingState:
        """
        Classify the user's query.
        """

        try:

            result = (
                self.intent_classifier.classify(
                    state.get(
                        "query",
                        ""
                    )
                )
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

        except Exception as exc:

            return add_error(
                state,
                f"Intent classification failed: {exc}"
            )

    def route(
        self,
        state: RoutingState
    ) -> RoutingState:
        """
        Determine which model(s) should execute.
        """

        try:

            return self.router.route_state(
                state
            )

        except Exception as exc:

            return add_error(
                state,
                f"Routing failed: {exc}"
            )


    def run_sia(
        self,
        state: RoutingState
    ) -> RoutingState:
        """
        Execute Single Image Analysis.

        Import is performed inside the function so the model
        is not loaded unless SIA is actually selected.
        """

        try:

            from models.sia.inference import (
                run_sia,
            )

        except ImportError as exc:

            return add_error(
                state,
                f"Could not import SIA inference: {exc}"
            )

        try:

            image_path = (
                state.get(
                    "image_path"
                )
                or
                state.get(
                    "optical_path"
                )
                or
                state.get(
                    "t1_path"
                )
            )

            if not image_path:

                return add_error(
                    state,
                    "SIA requires an image input."
                )

            result = run_sia(
                image_path=image_path,
                query=state.get(
                    "query",
                    ""
                ),
            )


            answer = self.extract_answer(
                result
            )

            state = update_state(

                state,

                sia_result=result,

                sia_answer=answer,
            )

            state = add_executed_model(
                state,
                SIA
            )

            return state

        except Exception as exc:

            return add_error(
                state,
                f"SIA execution failed: {exc}"
            )

    def run_bta(
        self,
        state: RoutingState
    ) -> RoutingState:
        """
        Execute Bi-Temporal Analysis.
        """

        try:

            from models.bta.inference import (
                run_bta,
            )

        except ImportError as exc:

            return add_error(
                state,
                f"Could not import BTA inference: {exc}"
            )

        try:

            t1_path = state.get(
                "t1_path"
            )

            t2_path = state.get(
                "t2_path"
            )

            if not t1_path:

                return add_error(
                    state,
                    "BTA requires T1 image."
                )

            if not t2_path:

                return add_error(
                    state,
                    "BTA requires T2 image."
                )

            result = run_bta(

                t1_path=t1_path,

                t2_path=t2_path,

                query=state.get(
                    "query",
                    ""
                ),
            )


            answer = self.extract_answer(
                result
            )

            state = update_state(

                state,

                bta_result=result,

                bta_answer=answer,
            )

            state = add_executed_model(
                state,
                BTA
            )

            return state

        except Exception as exc:

            return add_error(
                state,
                f"BTA execution failed: {exc}"
            )

    def run_cross_modal(
        self,
        state: RoutingState
    ) -> RoutingState:
        """
        Execute Cross-Modal Optical + SAR Analysis.
        """

        try:

            from models.cross_modal.inference import (
                run_cross_modal,
            )

        except ImportError as exc:

            return add_error(
                state,
                f"Could not import Cross-Modal inference: {exc}"
            )

        try:

            optical_path = state.get(
                "optical_path"
            )

            sar_path = state.get(
                "sar_path"
            )

            if not optical_path:

                return add_error(
                    state,
                    "Cross-Modal requires an optical image."
                )

            if not sar_path:

                return add_error(
                    state,
                    "Cross-Modal requires a SAR image."
                )

            result = run_cross_modal(

                optical_path=optical_path,

                sar_path=sar_path,

                query=state.get(
                    "query",
                    ""
                ),
            )


            answer = self.extract_answer(
                result
            )

            state = update_state(

                state,

                cross_modal_result=result,

                cross_modal_answer=answer,
            )

            state = add_executed_model(
                state,
                CROSS_MODAL
            )

            return state

        except Exception as exc:

            return add_error(
                state,
                f"Cross-Modal execution failed: {exc}"
            )

    def execute_models(
        self,
        state: RoutingState
    ) -> RoutingState:
        """
        Execute every model selected by the router.

        Models execute sequentially to avoid exceeding the
        RTX 5050's available VRAM.
        """

        routes = state.get(
            "routes",
            []
        )

        if not routes:

            route = state.get(
                "route"
            )

            if route:

                routes = [
                    route
                ]


        for route in routes:


            if route == SIA:

                state = self.run_sia(
                    state
                )


            elif route == BTA:

                state = self.run_bta(
                    state
                )


            elif route == CROSS_MODAL:

                state = self.run_cross_modal(
                    state
                )

        return state

    def generate_evidence(
        self,
        state: RoutingState
    ) -> RoutingState:
        """
        Generate structured evidence from model outputs.
        """

        try:

            from evidence.evidence_generator import (
                generate_evidence,
            )

        except ImportError as exc:

            return add_error(
                state,
                f"Could not import evidence generator: {exc}"
            )

        try:

            evidence = generate_evidence(

                query=state.get(
                    "query"
                ),

                sia=state.get(
                    "sia_answer"
                ),

                bta=state.get(
                    "bta_answer"
                ),

                cross_modal=state.get(
                    "cross_modal_answer"
                ),
            )

            return update_state(

                state,

                evidence=evidence,

                evidence_items=evidence.get(
                    "evidence",
                    []
                ),
            )

        except Exception as exc:

            return add_error(
                state,
                f"Evidence generation failed: {exc}"
            )

    def calculate_confidence(
        self,
        state: RoutingState
    ) -> RoutingState:
        """
        Calculate final confidence.
        """

        try:

            from evidence.confidence import (
                calculate_confidence,
            )

        except ImportError as exc:

            return add_error(
                state,
                f"Could not import confidence estimator: {exc}"
            )

        try:

            evidence = state.get(
                "evidence"
            )

            if evidence is None:

                evidence = {
                    "evidence": state.get(
                        "evidence_items",
                        []
                    )
                }

            confidence = calculate_confidence(
                evidence
            )

            return update_state(

                state,

                confidence=confidence,

                confidence_score=confidence.get(
                    "confidence_score"
                ),

                confidence_level=confidence.get(
                    "confidence_level"
                ),
            )

        except Exception as exc:

            return add_error(
                state,
                f"Confidence calculation failed: {exc}"
            )


    def generate_final_answer(
        self,
        state: RoutingState
    ) -> RoutingState:
        """
        Combine model outputs into the final answer.

        For a single model, return that model's answer.

        For multiple models, combine their answers into a
        structured response.
        """

        executed_models = state.get(
            "executed_models",
            []
        )

        answers = []

        if state.get(
            "sia_answer"
        ):

            answers.append(
                (
                    "SIA",
                    state[
                        "sia_answer"
                    ]
                )
            )

        if state.get(
            "bta_answer"
        ):

            answers.append(
                (
                    "BTA",
                    state[
                        "bta_answer"
                    ]
                )
            )


        if state.get(
            "cross_modal_answer"
        ):

            answers.append(
                (
                    "Cross-Modal",
                    state[
                        "cross_modal_answer"
                    ]
                )
            )

        if not answers:

            final_answer = (
                "No model was able to produce "
                "an answer."
            )


        elif len(
            answers
        ) == 1:

            final_answer = answers[0][1]

        else:

            sections = []

            for model_name, answer in answers:

                sections.append(

                    f"{model_name} Analysis:\n"
                    f"{answer}"
                )

            final_answer = (
                "\n\n".join(
                    sections
                )
            )

        final_response = {

            "answer":
                final_answer,

            "models":
                executed_models,

            "confidence":
                state.get(
                    "confidence_score"
                ),

            "confidence_level":
                state.get(
                    "confidence_level"
                ),

            "intent":
                state.get(
                    "intent"
                ),

            "route":
                state.get(
                    "routes",
                    []
                ),
        }

        return update_state(

            state,

            final_answer=final_answer,

            final_response=final_response,
        )


    def run(
        self,
        state: RoutingState
    ) -> RoutingState:
        """
        Execute the complete SATQueryAI pipeline.
        """

        start_time = time.perf_counter()


        validation_errors = validate_state(
            state
        )

        if validation_errors:

            for error in validation_errors:

                state = add_error(
                    state,
                    error
                )

            return state


        state = mark_running(
            state
        )

        state = self.classify_intent(
            state
        )

        if state.get(
            "error"
        ):

            return self.finish(
                state,
                start_time
            )


        state = self.route(
            state
        )

        if state.get(
            "error"
        ):

            return self.finish(
                state,
                start_time
            )


        state = self.execute_models(
            state
        )

        state = self.generate_evidence(
            state
        )

        state = self.calculate_confidence(
            state
        )


        state = self.generate_final_answer(
            state
        )


        return self.finish(
            state,
            start_time
        )


    def finish(
        self,
        state: RoutingState,
        start_time: float
    ) -> RoutingState:
        """
        Finish graph execution and record timing.
        """

        elapsed = (
            time.perf_counter()
            - start_time
        )

        state = update_state(

            state,

            execution_time=round(
                elapsed,
                3
            ),
        )

        if state.get(
            "error"
        ):

            return state

        return mark_complete(
            state
        )


    @staticmethod
    def extract_answer(
        result: Any
    ) -> str:
        """
        Extract a textual answer from different possible
        inference return formats.
        """

        if result is None:

            return ""


        if isinstance(
            result,
            str
        ):

            return result

        if isinstance(
            result,
            dict
        ):

            for key in [
                "answer",
                "response",
                "text",
                "result",
                "output",
            ]:

                value = result.get(
                    key
                )

                if isinstance(
                    value,
                    str
                ):

                    return value

            return str(
                result
            )

        return str(
            result
        )


graph = SATQueryGraph()


def run_pipeline(
    query: str,
    image_path: Optional[str] = None,
    optical_path: Optional[str] = None,
    sar_path: Optional[str] = None,
    t1_path: Optional[str] = None,
    t2_path: Optional[str] = None,
) -> RoutingState:
    """
    Run the complete SATQueryAI pipeline.

    Example:

        result = run_pipeline(
            query="What changed between T1 and T2?",
            t1_path="data/input/T1.png",
            t2_path="data/input/T2.png"
        )
    """

    state = create_initial_state(

        query=query,

        image_path=image_path,

        optical_path=optical_path,

        sar_path=sar_path,

        t1_path=t1_path,

        t2_path=t2_path,
    )

    return graph.run(
        state
    )



if __name__ == "__main__":

    print(
        "\n========== SATQUERYAI GRAPH TEST =========="
    )

    test_query = (
        "What changed between T1 and T2?"
    )

    state = create_initial_state(

        query=test_query,

        t1_path=(
            "data/input/T1.png"
        ),

        t2_path=(
            "data/input/T2.png"
        ),
    )

    print(
        "\nQuery:"
    )

    print(
        state["query"]
    )


    validation_errors = (
        validate_state(
            state
        )
    )

    if validation_errors:

        print(
            "\nValidation FAILED:"
        )

        for error in validation_errors:

            print(
                f"- {error}"
            )

    else:

        print(
            "\nValidation: PASSED"
        )

    state = graph.classify_intent(
        state
    )

    print(
        "\nIntent:"
    )

    print(
        state.get(
            "intent"
        )
    )

    print(
        "Intent confidence:",
        state.get(
            "intent_confidence"
        )
    )


    state = graph.route(
        state
    )

    print(
        "\nRoute:"
    )

    print(
        state.get(
            "route"
        )
    )

    print(
        "Routes:",
        state.get(
            "routes"
        )
    )

    print(
        "Routing confidence:",
        state.get(
            "routing_confidence"
        )
    )

    print(
        "Routing reason:"
    )

    print(
        state.get(
            "routing_reason"
        )
    )

    print(
        "\n==========================================="
    )