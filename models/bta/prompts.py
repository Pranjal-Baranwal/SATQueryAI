"""
Prompt templates for Bi-Temporal Analysis (BTA).

BTA compares two satellite images of the same area captured
at different points in time.

Image 1 = T1 (earlier)
Image 2 = T2 (later)
"""


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an expert remote-sensing analyst specializing in
bi-temporal satellite image analysis and change detection.

You are given two satellite images of the same geographic
area captured at different points in time.

Image T1 represents the earlier observation.
Image T2 represents the later observation.

Your task is to compare T1 and T2 and identify meaningful
changes between them.

Follow these rules:

1. Always distinguish T1 from T2.
2. Only report changes supported by visual evidence.
3. Do not describe a feature as "changed" merely because
   it appears different due to shadows, image quality,
   illumination, season, or viewing conditions.
4. Separate observed changes from possible interpretations.
5. Do not invent objects, structures, roads, vegetation,
   water bodies, or other features.
6. If no meaningful change is visible, explicitly state that.
7. If the evidence is uncertain, clearly state the uncertainty.
8. Describe the approximate spatial location of changes
   within the image when possible.
9. Focus on meaningful land-cover, infrastructure,
   environmental, and human-induced changes.
10. Do not claim exact geographic coordinates unless they are
    provided separately by the application.
"""


# ============================================================
# GENERAL BTA
# ============================================================

GENERAL_BTA_PROMPT = """
Compare the two satellite images provided.

Image T1:
Earlier observation.

Image T2:
Later observation.

Identify the meaningful visual changes between T1 and T2.

For each change, describe:

- What changed.
- Where the change appears.
- Whether the feature increased, decreased, appeared,
  disappeared, or was modified.
- The likely type of change.
- Your confidence: High, Medium, or Low.

Distinguish clearly between direct visual observations
and interpretations.

If there is no clearly visible meaningful change, state so.

User question:
{query}
"""


# ============================================================
# CHANGE DETECTION
# ============================================================

CHANGE_DETECTION_PROMPT = """
Perform a detailed visual change-detection analysis between
satellite images T1 and T2.

Compare the images systematically.

Look for changes involving:

- Buildings
- Roads
- Infrastructure
- Vegetation
- Forest
- Agricultural land
- Water bodies
- Rivers
- Lakes
- Flooded areas
- Bare soil
- Construction
- Excavation
- Urban expansion
- Land-cover conversion
- Other clearly visible surface changes

For every significant change:

1. Identify the feature.
2. Describe its approximate location.
3. Describe its condition in T1.
4. Describe its condition in T2.
5. Describe the observed change.
6. Give a confidence level.

Do not treat differences caused only by lighting, shadows,
seasonality, sensor differences, or image quality as definite
land-cover changes.
"""


# ============================================================
# URBAN CHANGE
# ============================================================

URBAN_CHANGE_PROMPT = """
Compare T1 and T2 specifically for urban and built-environment
changes.

Look for:

- New buildings
- Demolished buildings
- Building expansion
- New roads
- Road expansion
- New infrastructure
- Construction activity
- Urban expansion
- Changes in developed land
- Industrial development
- Parking or paved-area expansion
- Changes in settlement density

For each detected change, report:

Location:
T1 condition:
T2 condition:
Observed change:
Confidence:

Only report changes supported by visible evidence.
"""


# ============================================================
# VEGETATION CHANGE
# ============================================================

VEGETATION_CHANGE_PROMPT = """
Compare T1 and T2 for vegetation-related changes.

Look for:

- Deforestation
- Reforestation
- Vegetation loss
- Vegetation growth
- Agricultural expansion
- Agricultural abandonment
- Vegetation disturbance
- Clearing
- Newly vegetated areas
- Changes in forest extent

For each detected change, describe:

- Approximate location
- Vegetation condition in T1
- Vegetation condition in T2
- Observed change
- Confidence

Do not interpret seasonal differences as definite
vegetation loss or gain unless the evidence strongly
supports a real spatial change.
"""


# ============================================================
# WATER CHANGE
# ============================================================

WATER_CHANGE_PROMPT = """
Compare T1 and T2 for changes involving water.

Look for:

- Flooding
- Flood recession
- Expansion of water bodies
- Reduction of water bodies
- Changes in rivers
- Changes in lakes
- Reservoir changes
- Newly appearing water
- Disappearing water
- Changes in shoreline or water extent

For each detected change, report:

- Approximate location
- Water extent in T1
- Water extent in T2
- Observed change
- Confidence

Do not confuse shadows, dark surfaces, or image artifacts
with water.
"""


# ============================================================
# INFRASTRUCTURE CHANGE
# ============================================================

INFRASTRUCTURE_CHANGE_PROMPT = """
Compare T1 and T2 for infrastructure-related changes.

Focus on:

- Roads
- Bridges
- Buildings
- Communication infrastructure
- Industrial structures
- Transportation infrastructure
- Construction sites
- New paved surfaces
- Demolition
- Infrastructure expansion

For every significant change:

- Identify the feature.
- Describe its location.
- Describe its appearance in T1.
- Describe its appearance in T2.
- Explain the observed modification.
- Assign High, Medium, or Low confidence.

Do not report uncertain visual differences as confirmed
infrastructure changes.
"""


# ============================================================
# ENVIRONMENTAL CHANGE
# ============================================================

ENVIRONMENTAL_CHANGE_PROMPT = """
Compare T1 and T2 for environmentally significant changes.

Look for:

- Deforestation
- Vegetation degradation
- Soil disturbance
- Flooding
- Water-body changes
- Coastal changes
- Land degradation
- Agricultural changes
- Fire scars or burned areas
- Habitat disturbance
- Other visible environmental changes

Clearly separate:

Observed visual change

from

Possible environmental interpretation.

Do not make claims that cannot be supported by the imagery.
"""


# ============================================================
# CHANGE SUMMARY
# ============================================================

CHANGE_SUMMARY_PROMPT = """
Compare T1 and T2 and provide a concise summary of the
most important changes.

Rank the detected changes from most significant to least
significant.

For each significant change provide:

1. Feature
2. Approximate location
3. T1 condition
4. T2 condition
5. Change description
6. Confidence

End with an overall assessment of whether the area shows:

- Major change
- Moderate change
- Minor change
- No clearly detectable change

Do not exaggerate uncertain differences.
"""


# ============================================================
# NO-CHANGE VERIFICATION
# ============================================================

NO_CHANGE_VERIFICATION_PROMPT = """
Compare T1 and T2 carefully and determine whether there are
any meaningful visible changes.

Pay particular attention to:

- Buildings
- Roads
- Vegetation
- Water
- Agricultural areas
- Bare land
- Infrastructure

Do not report changes caused only by:

- Shadows
- Lighting
- Seasonal appearance
- Minor image alignment differences
- Atmospheric effects
- Sensor differences
- Compression artifacts

If no meaningful change is clearly visible, state:

"No significant visible change detected."

If changes are visible, describe them and provide confidence.
"""


# ============================================================
# GENERAL PROMPT BUILDER
# ============================================================

def build_general_prompt(query):
    """
    Build a general BTA prompt.
    """

    if not query or not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    return GENERAL_BTA_PROMPT.format(
        query=query.strip()
    )


# ============================================================
# CHANGE DETECTION PROMPT BUILDER
# ============================================================

def build_change_detection_prompt(query=None):
    """
    Build a detailed change-detection prompt.
    """

    prompt = CHANGE_DETECTION_PROMPT

    if query and query.strip():

        prompt += f"""

Additional user question:
{query.strip()}
"""

    return prompt


# ============================================================
# URBAN PROMPT BUILDER
# ============================================================

def build_urban_change_prompt(query=None):
    """
    Build an urban change analysis prompt.
    """

    prompt = URBAN_CHANGE_PROMPT

    if query and query.strip():

        prompt += f"""

Additional user question:
{query.strip()}
"""

    return prompt


# ============================================================
# VEGETATION PROMPT BUILDER
# ============================================================

def build_vegetation_change_prompt(query=None):
    """
    Build a vegetation change analysis prompt.
    """

    prompt = VEGETATION_CHANGE_PROMPT

    if query and query.strip():

        prompt += f"""

Additional user question:
{query.strip()}
"""

    return prompt


# ============================================================
# WATER PROMPT BUILDER
# ============================================================

def build_water_change_prompt(query=None):
    """
    Build a water change analysis prompt.
    """

    prompt = WATER_CHANGE_PROMPT

    if query and query.strip():

        prompt += f"""

Additional user question:
{query.strip()}
"""

    return prompt


# ============================================================
# INFRASTRUCTURE PROMPT BUILDER
# ============================================================

def build_infrastructure_change_prompt(query=None):
    """
    Build an infrastructure change analysis prompt.
    """

    prompt = INFRASTRUCTURE_CHANGE_PROMPT

    if query and query.strip():

        prompt += f"""

Additional user question:
{query.strip()}
"""

    return prompt


# ============================================================
# ENVIRONMENTAL PROMPT BUILDER
# ============================================================

def build_environmental_change_prompt(query=None):
    """
    Build an environmental change analysis prompt.
    """

    prompt = ENVIRONMENTAL_CHANGE_PROMPT

    if query and query.strip():

        prompt += f"""

Additional user question:
{query.strip()}
"""

    return prompt


# ============================================================
# SUMMARY PROMPT BUILDER
# ============================================================

def build_change_summary_prompt():
    """
    Return the change-summary prompt.
    """

    return CHANGE_SUMMARY_PROMPT


# ============================================================
# NO-CHANGE PROMPT BUILDER
# ============================================================

def build_no_change_verification_prompt():
    """
    Return the no-change verification prompt.
    """

    return NO_CHANGE_VERIFICATION_PROMPT