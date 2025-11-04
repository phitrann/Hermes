Overview
In this assessment, you are asked to design and prototype a lightweight AI Logistics Assistant named Hermes that helps operations managers analyze shipment data and answer natural language questions.

The goal is to demonstrate your ability to apply data analytics, and basic NLP - optionally with simple machine learning - to a practical logistics scenario.

Scenario

Your company manages shipments across different routes and warehouses.

Managers often ask questions like:
-	“Which route had the most delays last week?”
-	“Show the top 3 warehouses with the highest processing time.”
-	“Predict average delivery delay next week.”

You will build a small prototype that can:
-	Understand logistics-related queries (chatbot style).
-	Search and summarize shipment information.
-	Optionally make a simple prediction or recommendation based on the data.

Your Task
Implement a simple interactive app (Streamlit, Gradio, or Jupyter Notebook) with the following components:
- Chat-based Query Interface
    - Accept user input in natural language.
    - Supports at least 3 example query types (e.g., delay statistics, warehouse ranking, route performance, etc.).
    - Returns textual or visual answers from your data.
- Data Search & Analytics
    - Parse mock shipment data (shipments.csv).
    - Support queries like filtering by route, warehouse, or delaying reason.
    - Display relevant summaries or charts.
- Bonus (Optional)
    - Add a simple model (e.g., linear regression) to predict next week’s average delay.
    - Or create a simple rule-based recommendation for warehouse optimization.
________________________________________
Key Functional Goals
Your AI system should be able to:
1.	Data understanding

Read and process structured shipment datasets (mock CSV/JSON you create).
Mock Data Example
```
shipments.csv
id, route, warehouse, delivery_time, delay_minutes, delay_reason, date
1, Route A, WH1,5.2,30, Weather,2024-10-10
2, Route B, WH2,4.8,0, None,2024-10-11
...
```

2.	Query understanding & generation

Compute and present meaningful answers from the dataset: aggregations, top-k lists, time-series summaries, and filters by reason/warehouse/route.

Example queries Hermes should handle
- “Which route had the most delays last week?”
- “Show total delayed shipments by delay reason.”
- “List warehouses with average delivery time above 5 days.”
- “What was the average delay in October?”
- “Predict the delay rate for next week.” (optional / bonus)
3.	Simplicity
- Keep your code modular, readable, and easy to run locally.
4.	Creativity & UX
- Add small but valuable touches: visualization of trends (charts), caching for faster responses, clear UI/UX for chat & result display.
________________________________________

Deliverables

Submit the following (all in a Microsoft Form response or as links):

1.	Documentation (short report)
- Describe your data structure, system architecture, and workflow.
- Explain how your system performs query understanding and data summarization.
- Include screenshots or examples of queries and expected outputs.

2.	Prototype / Code
- Provide an interactive prototype (e.g. Jupyter Notebook, Web-app) as a zipped folder or GitHub repo
- You may use mock or synthetic data to simulate shipment reports.

3.	Evaluation Design
- Propose how you would evaluate your AI system’s performance:
    - Accuracy of query results - Are responses correct and relevant?
    - Explainability - Can the system justify or show data sources for its answers?
    - Response time - Is the system efficient and responsive?

