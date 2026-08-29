"""
SATQueryAI - Main Application
=============================

Main backend entry point for SATQueryAI.

Pipeline:

    User Query
         |
         v
    Intent Classifier
         |
         v
       Router
         |
         +-------------------+
         |                   |
         v                   v
        SIA                 BTA
         |                   |
         +---------+---------+
                   |
                   v
             Cross-Modal
                   |
                   v
              Geospatial
                   |
                   v
           Evidence Generator
                   |
                   v
              Confidence
                   |
                   v
             Final Response
"""

import os
import time
from typing import Optional, Dict, Any

from routing.graph import run_pipeline


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_T1_PATH = "data/input/T1.png"
DEFAULT_T2_PATH = "data/input/T2.png"
DEFAULT_OPTICAL_PATH = "data/input/T1.png"
DEFAULT_SAR_PATH = "data/input/Sample.tif"


# ============================================================
# SATQUERYAI APPLICATION
# ============================================================

class SATQueryAI:
    """
    Main SATQueryAI application.

    The frontend should communicate with this class rather
    than directly communicating with individual models.
    """

    def __init__(self):

        print(
            "SATQueryAI initialized."
        )

    # ========================================================
    # PROCESS QUERY
    # ========================================================

    def process_query(
        self,
        query: str,
        image_path: Optional[str] = None,
        optical_path: Optional[str] = None,
        sar_path: Optional[str] = None,
        t1_path: Optional[str] = None,
        t2_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a user query through the complete SATQueryAI
        routing pipeline.
        """

        # ----------------------------------------------------
        # Validate query
        # ----------------------------------------------------

        if not query:

            return {
                "success": False,
                "error": "Query cannot be empty.",
            }

        query = query.strip()

        if not query:

            return {
                "success": False,
                "error": "Query cannot be empty.",
            }

        # ----------------------------------------------------
        # Execute pipeline
        # ----------------------------------------------------

        try:

            state = run_pipeline(

                query=query,

                image_path=image_path,

                optical_path=optical_path,

                sar_path=sar_path,

                t1_path=t1_path,

                t2_path=t2_path,
            )

            return self.format_response(
                state
            )

        except Exception as exc:

            return {

                "success": False,

                "error": str(exc),

                "error_type": type(
                    exc
                ).__name__,
            }

    # ========================================================
    # FORMAT RESPONSE
    # ========================================================

    @staticmethod
    def format_response(
        state
    ) -> Dict[str, Any]:
        """
        Convert internal RoutingState into a frontend-friendly
        response.
        """

        errors = state.get(
            "errors",
            []
        )

        # ----------------------------------------------------
        # Pipeline error
        # ----------------------------------------------------

        if state.get(
            "error"
        ):

            return {

                "success": False,

                "error": state.get(
                    "error"
                ),

                "errors": errors,

                "query": state.get(
                    "query"
                ),

                "intent": state.get(
                    "intent"
                ),

                "route": state.get(
                    "route"
                ),

                "routes": state.get(
                    "routes",
                    []
                ),

                "execution_status": state.get(
                    "execution_status"
                ),

                "state": state,
            }

        # ----------------------------------------------------
        # Successful response
        # ----------------------------------------------------

        return {

            "success": True,

            # =================================================
            # QUERY
            # =================================================

            "query": state.get(
                "query"
            ),

            # =================================================
            # INTENT
            # =================================================

            "intent": state.get(
                "intent"
            ),

            "intent_confidence": state.get(
                "intent_confidence"
            ),

            "intent_reason": state.get(
                "intent_reason"
            ),

            # =================================================
            # ROUTING
            # =================================================

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

            "routing_reason": state.get(
                "routing_reason"
            ),

            # =================================================
            # MODEL EXECUTION
            # =================================================

            "executed_models": state.get(
                "executed_models",
                []
            ),

            # =================================================
            # SIA
            # =================================================

            "sia_answer": state.get(
                "sia_answer"
            ),

            "sia_result": state.get(
                "sia_result"
            ),

            # =================================================
            # BTA
            # =================================================

            "bta_answer": state.get(
                "bta_answer"
            ),

            "bta_result": state.get(
                "bta_result"
            ),

            # =================================================
            # CROSS-MODAL
            # =================================================

            "cross_modal_answer": state.get(
                "cross_modal_answer"
            ),

            "cross_modal_result": state.get(
                "cross_modal_result"
            ),

            # =================================================
            # GEOSPATIAL
            # =================================================

            "geospatial_result": state.get(
                "geospatial_result"
            ),

            "geospatial_answer": state.get(
                "geospatial_answer"
            ),

            # =================================================
            # EVIDENCE
            # =================================================

            "evidence": state.get(
                "evidence"
            ),

            "evidence_items": state.get(
                "evidence_items",
                []
            ),

            # =================================================
            # CONFIDENCE
            # =================================================

            "confidence": state.get(
                "confidence"
            ),

            "confidence_score": state.get(
                "confidence_score"
            ),

            "confidence_level": state.get(
                "confidence_level"
            ),

            # =================================================
            # FINAL ANSWER
            # =================================================

            "answer": state.get(
                "final_answer"
            ),

            "final_response": state.get(
                "final_response"
            ),

            # =================================================
            # EXECUTION
            # =================================================

            "execution_status": state.get(
                "execution_status"
            ),

            "execution_time": state.get(
                "execution_time"
            ),

            "errors": errors,

            # =================================================
            # COMPLETE STATE
            # =================================================

            "state": state,
        }


# ============================================================
# GLOBAL APPLICATION INSTANCE
# ============================================================

app = SATQueryAI()


# ============================================================
# FRONTEND INTERFACE
# ============================================================

def process_query(
    query: str,
    image_path: Optional[str] = None,
    optical_path: Optional[str] = None,
    sar_path: Optional[str] = None,
    t1_path: Optional[str] = None,
    t2_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main function that should be called by the frontend.
    """

    return app.process_query(

        query=query,

        image_path=image_path,

        optical_path=optical_path,

        sar_path=sar_path,

        t1_path=t1_path,

        t2_path=t2_path,
    )


# ============================================================
# HEALTH CHECK
# ============================================================

def health_check() -> Dict[str, Any]:
    """
    Basic application health check.
    """

    return {

        "status": "ok",

        "application": "SATQueryAI",

        "routing": "available",

        "models": [
            "SIA",
            "BTA",
            "Cross-Modal",
        ],

        "evidence": "available",

        "confidence": "available",
    }


# ============================================================
# TERMINAL DISPLAY
# ============================================================

def print_result(
    result: Dict[str, Any]
):
    """
    Pretty-print the SATQueryAI result.
    """

    print(
        "\n========================================"
    )

    print(
        "             SATQUERYAI RESULT"
    )

    print(
        "========================================"
    )

    if not result.get(
        "success"
    ):

        print(
            "\nERROR:"
        )

        print(
            result.get(
                "error"
            )
        )

        if result.get(
            "errors"
        ):

            print(
                "\nErrors:"
            )

            for error in result[
                "errors"
            ]:

                print(
                    f"- {error}"
                )

        print(
            "\n========================================"
        )

        return

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    print(
        "\nQuery:"
    )

    print(
        result.get(
            "query"
        )
    )

    # --------------------------------------------------------
    # Intent
    # --------------------------------------------------------

    print(
        "\nIntent:"
    )

    print(
        result.get(
            "intent"
        )
    )

    print(
        "Intent confidence:",
        result.get(
            "intent_confidence"
        )
    )

    # --------------------------------------------------------
    # Routing
    # --------------------------------------------------------

    print(
        "\nRoute:"
    )

    print(
        result.get(
            "route"
        )
    )

    print(
        "Routes:",
        result.get(
            "routes"
        )
    )

    print(
        "Routing confidence:",
        result.get(
            "routing_confidence"
        )
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    print(
        "\nExecuted models:"
    )

    print(
        result.get(
            "executed_models"
        )
    )

    # --------------------------------------------------------
    # Final answer
    # --------------------------------------------------------

    print(
        "\nFinal Answer:"
    )

    print(
        result.get(
            "answer"
        )
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    print(
        "\nConfidence score:"
    )

    print(
        result.get(
            "confidence_score"
        )
    )

    print(
        "Confidence level:",
        result.get(
            "confidence_level"
        )
    )

    # --------------------------------------------------------
    # Execution
    # --------------------------------------------------------

    print(
        "\nExecution status:"
    )

    print(
        result.get(
            "execution_status"
        )
    )

    print(
        "Execution time:",
        result.get(
            "execution_time"
        ),
        "seconds"
    )

    print(
        "\n========================================"
    )


# ============================================================
# APPLICATION TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "        SATQUERYAI APPLICATION TEST"
    )

    print(
        "========================================"
    )

    print(
        "\nHealth check:"
    )

    print(
        health_check()
    )

    print(
        "\n----------------------------------------"
    )

    print(
        "Testing routing pipeline"
    )

    print(
        "----------------------------------------"
    )

    print(
        "\nQuery:"
    )

    print(
        "What changed between T1 and T2?"
    )

    print(
        "\nT1:",
        DEFAULT_T1_PATH
    )

    print(
        "T2:",
        DEFAULT_T2_PATH
    )

    print(
        "\nStarting pipeline..."
    )

    start_time = time.time() if "time" in globals() else None

    result = process_query(

        query=(
            "What changed between T1 and T2?"
        ),

        t1_path=DEFAULT_T1_PATH,

        t2_path=DEFAULT_T2_PATH,
    )

    print_result(
        result
    )