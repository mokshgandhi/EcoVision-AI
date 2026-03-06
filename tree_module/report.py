import os
from openai import OpenAI


def generate_forest_report(metrics):

    api_key = "YOUR_OPENAI_API_KEY"

    # If no key is provided, skip report generation
    if not api_key:
        return None

    client = OpenAI(api_key=api_key)

    prompt = f"""
You are a forest ecology expert analyzing forest monitoring data.

Forest Metrics:
Tree Count: {metrics["tree_count"]}
Tree Density: {metrics["tree_density"]} trees/km²
Average Tree Spacing: {metrics["avg_tree_spacing"]} meters
Forest Health Score: {metrics["forest_health_score"]}
Mean NDVI: {metrics["ndvi_mean"]}

Provide a concise report including:

1. Forest Health Assessment
2. Key Observations
3. Potential Environmental Risks
4. Recommended Forest Management Actions

Keep the report under 200 words.
"""

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an environmental scientist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:

        print("AI report generation failed:", e)
        return None