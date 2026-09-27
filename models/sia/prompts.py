


SYSTEM_PROMPT = """
You are an expert remote-sensing image analyst.

You analyze satellite and aerial imagery and answer questions
using only the visual information provided in the image and
the metadata supplied by the application.

Your responsibilities include:
- Scene understanding
- Land-cover interpretation
- Object identification
- Spatial interpretation
- Image description
- Visual question answering
- Remote-sensing reasoning

Rules:
1. Base your answer on evidence visible in the image.
2. Do not invent objects, locations, measurements, or facts.
3. If the image does not contain enough information to answer
   confidently, explicitly say so.
4. Distinguish clearly between observation and inference.
5. Use remote-sensing terminology when appropriate.
6. Keep the answer concise but informative.
7. If the question asks for a specific object or feature,
   describe its approximate location in the image when possible.
8. Do not claim exact geographic coordinates unless they are
   provided or can be reliably derived from the supplied metadata.
"""


SCENE_DESCRIPTION_PROMPT = """
Analyze the provided satellite image.

Describe:
1. The overall scene.
2. The major land-cover types or surface features.
3. Important visible objects or structures.
4. Any notable spatial patterns.
5. Any obvious environmental or human-made features.

Do not speculate beyond what can reasonably be observed.

Return a concise but useful remote-sensing scene description.
"""


OBJECT_IDENTIFICATION_PROMPT = """
Analyze the provided satellite image and identify the requested
object or feature.

For each identified object:
- State what it appears to be.
- Describe its approximate location in the image.
- Mention relevant visual characteristics.
- State your confidence qualitatively as high, medium, or low.

If the requested object cannot be reliably identified, say so.
"""


LAND_COVER_PROMPT = """
Analyze the provided satellite image for land-cover information.

Identify the major visible land-cover categories, such as:
- Vegetation
- Water
- Urban or built-up areas
- Roads
- Bare soil
- Agricultural areas
- Forest
- Other clearly visible surface types

Explain where these categories occur spatially in the image.

Do not assign a land-cover category when the visual evidence
is insufficient.
"""


VEGETATION_PROMPT = """
Analyze the vegetation visible in the provided satellite image.

Describe:
- Areas containing vegetation.
- Relative vegetation density where visually apparent.
- Major vegetation patterns.
- Areas with sparse or dense vegetation.
- Any visible signs of vegetation disturbance.

Do not claim quantitative vegetation health or NDVI values
unless such information is explicitly provided.
"""


WATER_PROMPT = """
Analyze the provided satellite image for water-related features.

Identify visible:
- Rivers
- Lakes
- Reservoirs
- Ponds
- Coastal water
- Flooded areas
- Other water bodies

Describe their approximate location and spatial extent
based only on the visible image.

Do not invent water bodies that are not visible.
"""


URBAN_PROMPT = """
Analyze the provided satellite image for urban and built-up
features.

Identify visible:
- Buildings
- Roads
- Dense settlements
- Industrial areas
- Parking areas
- Urban infrastructure
- Other clearly visible human-made structures

Describe their approximate spatial distribution.

Distinguish between clearly visible structures and uncertain
interpretations.
"""


CHANGE_AWARE_PROMPT = """
Analyze the provided satellite image for features that may be
relevant to change analysis.

Identify:
- Newly developed or disturbed areas
- Vegetation disturbance
- Water changes
- Construction
- Deforestation
- Agricultural patterns
- Other visually significant features

Important:
This is a single-image analysis. Do not claim that a change
occurred over time unless a second temporal image is provided.
"""


GENERAL_VQA_PROMPT = """
Analyze the provided satellite image and answer the user's
question.

User question:
{query}

Use the image as the primary source of visual evidence.

Your response should:
1. Directly answer the question.
2. Explain the visual evidence supporting the answer.
3. Mention uncertainty when appropriate.
4. Avoid unsupported assumptions.
"""


def build_general_prompt(query):
    """
    Build a general SIA prompt for an arbitrary user query.

    Parameters
    ----------
    query : str
        User's natural-language question.

    Returns
    -------
    str
        Formatted prompt.
    """

    if not query or not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    return GENERAL_VQA_PROMPT.format(
        query=query.strip()
    )


def build_object_prompt(query):
    """
    Build a prompt for object identification.
    """

    if not query or not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    return f"""
{OBJECT_IDENTIFICATION_PROMPT}

User request:
{query.strip()}
"""


def build_land_cover_prompt(query=None):
    """
    Build a land-cover analysis prompt.

    An optional query can be added to make the analysis
    more specific.
    """

    prompt = LAND_COVER_PROMPT

    if query and query.strip():
        prompt += f"""

Additional user request:
{query.strip()}
"""

    return prompt


def build_vegetation_prompt(query=None):
    """
    Build a vegetation analysis prompt.
    """

    prompt = VEGETATION_PROMPT

    if query and query.strip():
        prompt += f"""

Additional user request:
{query.strip()}
"""

    return prompt


def build_water_prompt(query=None):
    """
    Build a water-feature analysis prompt.
    """

    prompt = WATER_PROMPT

    if query and query.strip():
        prompt += f"""

Additional user request:
{query.strip()}
"""

    return prompt


def build_urban_prompt(query=None):
    """
    Build an urban-feature analysis prompt.
    """

    prompt = URBAN_PROMPT

    if query and query.strip():
        prompt += f"""

Additional user request:
{query.strip()}
"""

    return prompt


def build_scene_description_prompt():
    """
    Return the standard scene-description prompt.
    """

    return SCENE_DESCRIPTION_PROMPT


def build_change_aware_prompt(query=None):
    """
    Build a single-image prompt focused on features relevant
    to potential change analysis.
    """

    prompt = CHANGE_AWARE_PROMPT

    if query and query.strip():
        prompt += f"""

Additional user request:
{query.strip()}
"""

    return prompt


GROUNDING_PROMPT_TEMPLATE = """
You are performing visual grounding on a satellite image.

Task: Localize EVERY instance in the image that matches the
following request:

{query}

COORDINATE SYSTEM (read carefully, this is the most common
source of error):
- Coordinates are (x, y) on a 0-1000 scale for BOTH axes,
  regardless of the image's actual pixel width or height.
- (0, 0) is the top-left corner of the image.
- (1000, 0) is the top-right corner.
- (0, 1000) is the bottom-left corner.
- (1000, 1000) is the bottom-right corner.
- (500, 500) is the exact center of the image.
- Example: a feature occupying most of the lower-right area
  of the image, away from the center, would have a point
  such as (750, 750). A feature in the upper-left area, away
  from the center, would have a point such as (200, 200).
- Before answering, mentally divide the image into four
  quadrants (top-left, top-right, bottom-left, bottom-right)
  and double-check which quadrant each instance is actually
  in. Getting the quadrant wrong is a serious error.

INSTRUCTIONS:
1. Scan the entire image systematically, quadrant by
   quadrant, before answering.
2. Satellite images frequently contain many separate
   instances of the same feature (several distinct clusters
   of trees, several buildings, several patches of water).
   Report ALL clearly visible instances, not just the most
   obvious one. Do not report instances you are not
   confident are actually visible.
3. For each instance, give ONE point placed at its visual
   center (not its edge, not a rough guess of some other
   part of the image).
4. For each instance, briefly describe where it is in plain
   words (e.g. "large dark rectangular basin in the lower
   right quadrant, right of center"). This description must
   be consistent with the point you give.

Respond with ONLY a JSON object, with no other text, no
markdown code fences, and no explanation, in exactly this
format:

{{"objects": [{{"label": "<short label>", "point": [x, y], "location_hint": "<brief plain-word description of where this is in the image>", "confidence": "high|medium|low"}}]}}

Rules:
1. point is a single [x, y] pair on the 0-1000 scale
   described above.
2. Include one entry per distinct, spatially separate
   instance.
3. If nothing in the image matches the request, respond with
   {{"objects": []}}.
4. Do not invent instances that are not visually supported.
"""


def build_grounding_prompt(query):
    """
    Build a visual-grounding prompt asking the model to return
    normalized bounding boxes as JSON.
    """

    if not query or not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    return GROUNDING_PROMPT_TEMPLATE.format(
        query=query.strip()
    )