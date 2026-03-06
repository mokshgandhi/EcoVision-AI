import os
from openai import OpenAI

client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

def generate_forest_report(metrics):

    prompt = f"""
You are an environmental scientist analyzing forest monitoring data.

Given the following forest metrics, generate a concise analytical report
with insights and actionable forest management recommendations.

Metrics:
Tree Count: {metrics["tree_count"]}
Tree Density: {metrics["tree_density"]} trees/km²
Average Tree Spacing: {metrics["avg_tree_spacing"]} m
Forest Health Score: {metrics["forest_health_score"]}
NDVI Vegetation Index: {metrics.get("ndvi_mean","unknown")}

Your report must contain:

1. Forest Health Assessment
2. Key Observations
3. Potential Risks
4. Recommended Actions for Forest Management

Keep the report under 200 words and make it professional.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a forest ecology expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    report = response.choices[0].message.content

    return report