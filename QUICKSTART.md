# Hermes AI - Quick Start Guide

## 🚀 Getting Started with the New Conversational Interface

### Overview
Hermes AI now features a modern, conversational chatbot interface similar to Claude, ChatGPT, and Grok. This guide will help you get started quickly.

---

## 🎯 Quick Start (3 Steps)

### 1. Start the Application

```bash
# Option 1: Using the console script
hermes

# Option 2: Using Python module
python -m hermes.app

# Option 3: Development mode (auto-reload)
python dev.py
```

The app will launch at `http://localhost:7860`

### 2. Load Your Data

In the sidebar:
1. Choose **"Select Existing"** or **"Upload New"**
2. Select your CSV file (e.g., `data/shipments.csv`)
3. Click **"📥 Load Data"**
4. Wait for the ✅ success message

### 3. Start Chatting!

Type your question in the chat box or:
- Click a **suggestion** button below the input
- Use a **sample question** from the dropdown
- Click a **quick action** button (Predict, Recommend, Statistics)

---

## 💬 Example Conversations

### Getting Started
```
You: Show me summary statistics
AI: 📊 Statistics Summary
    Total Shipments: 1,000
    Delayed Shipments: 234
    Average Delay: 45.3 minutes
    ...
```

### Visualizations
```
You: Visualize shipment trends over time
AI: 📊 Here's a chart showing shipment trends...
    [Chart displays inline in conversation]
```

### Predictions
```
You: Predict delays for next week
AI: 🔮 Delay Prediction Forecast
    Model Performance:
    - R² Score: 0.847
    - RMSE: 12.34 minutes
    
    Next Week Forecast:
    - Average Delay: 42.5 minutes
    - Median Delay: 38.2 minutes
    ...
```

### Recommendations
```
You: Give me recommendations
AI: 💡 Recommendations to Improve Your Logistics
    1. Best Practice 🟢
       Finding: Warehouse A has best delivery time
       Action: Document and replicate processes
    ...
```

---

## 🎨 Interface Overview

### Main Chat Area
- **Message Bubbles**: Your questions on the right, AI responses on the left
- **Inline Charts**: Visualizations appear directly in conversation
- **Typing Indicators**: See when AI is processing your request

### Sidebar Sections

#### 📁 Data Source
- **Mode Selection**: Choose existing or upload new CSV
- **File Selector**: Pick your data file
- **Load Button**: Explicitly load the selected data
- **Status Indicator**: Shows if data is loaded

#### ⚡ Quick Actions
- **🔮 Predict Delays**: Get delay forecasts
- **💡 Get Recommendations**: Receive improvement suggestions
- **📊 Show Statistics**: Display data summary

#### 💡 Sample Questions
- **Dropdown**: Pre-loaded example questions
- **Use Button**: Add question to chat

#### 🛠️ Tools
- **💾 Export History**: Download chat as JSON
- **🗑️ Clear Chat**: Start fresh conversation

### Input Area
- **Text Box**: Type your question (supports multi-line)
- **Send Button**: Submit your query
- **Suggestions**: Quick-click buttons for common queries

---

## 🔧 Common Tasks

### Analyzing Delays
```
Examples:
- "Which routes have the most delays?"
- "Compare delay patterns across warehouses"
- "Show me the trend of delays over the last month"
```

### Creating Visualizations
```
Examples:
- "Visualize shipment volume by warehouse"
- "Create a chart showing on-time delivery rates"
- "Plot delay distribution"
```

### Getting Insights
```
Examples:
- "What's causing most delays?"
- "Which warehouse performs best?"
- "Show me peak shipping hours"
```

### Making Predictions
```
Examples:
- "Predict next week's delays"
- "Forecast delivery times for next month"
- "What delays should I expect?"
```

---

## 💡 Pro Tips

### 1. Be Specific
❌ "Show me data"
✅ "Show me the top 10 routes with highest delays"

### 2. Ask Follow-up Questions
The chat preserves context, so you can:
```
You: Show me warehouse performance
AI: [Shows stats]
You: Now visualize that as a bar chart
AI: [Creates chart]
```

### 3. Use Natural Language
You don't need technical jargon:
- "What's going on with delays?" ✅
- "SELECT * FROM shipments WHERE delay > 0" ❌

### 4. Request Different Views
```
- "Show this as a table"
- "Create a chart for that"
- "Give me the numbers"
- "Visualize this data"
```

### 5. Combine Requests
```
"Analyze delay patterns by warehouse and create a visualization"
```

---

## 🐛 Troubleshooting

### Data Not Loading?
1. Check CSV format (must have proper headers)
2. Ensure required columns: `date`, `delay_minutes`, etc.
3. Look for status messages in sidebar
4. Check console logs for detailed errors

### No Chart Generated?
- LLM might return text instead of chart
- Try being more explicit: "create a bar chart of..."
- Check `charts/` folder for saved files
- Ensure `matplotlib` is installed

### Predictions Not Working?
Required columns:
- `date` (datetime)
- `delay_minutes` (numeric)
- `route` (categorical)
- `warehouse` (categorical)

### LLM Errors?
1. Ensure LLM endpoint is running (default: `http://localhost:8001/v1`)
2. Check `src/hermes/config.py` for correct URL
3. Verify API key if required
4. For testing, consider mocking `SmartDataframe.chat()`

---

## 🎯 Sample Workflow

### Complete Analysis Session

1. **Load Data**
   ```
   Click "Load Data" → Select shipments.csv → ✅ Success
   ```

2. **Get Overview**
   ```
   You: Show me summary statistics
   ```

3. **Identify Issues**
   ```
   You: Which warehouses have the most delays?
   ```

4. **Visualize**
   ```
   You: Create a chart comparing warehouse performance
   ```

5. **Get Recommendations**
   ```
   Click "Get Recommendations" button
   ```

6. **Predict Future**
   ```
   Click "Predict Delays" button
   ```

7. **Export Results**
   ```
   Click "Export History" to save conversation
   ```

---

## 📊 Data Format

Your CSV should have columns like:
```csv
id,date,warehouse,route,delay_minutes,delivery_time,on_time,cost
1,2025-01-01,Warehouse A,Route 101,0,2.5,1,125.50
2,2025-01-02,Warehouse B,Route 102,45,3.2,0,145.75
...
```

**Recommended Columns:**
- `id`: Unique identifier
- `date`: Shipment date (YYYY-MM-DD)
- `warehouse`: Warehouse name/ID
- `route`: Route identifier
- `delay_minutes`: Delay in minutes (0 = on time)
- `delivery_time`: Time to deliver (days)
- `on_time`: Binary flag (1 = on time, 0 = delayed)
- `cost`: Shipment cost (optional)
- `delay_reason`: Reason for delay (optional)

---

## 🔗 Additional Resources

- **README.md**: Project overview and setup
- **CHANGELOG.md**: Detailed changes in v2.0
- **TUTORIAL.md**: In-depth tutorials
- **.github/copilot-instructions.md**: Development guide

---

## 🆘 Getting Help

1. **Check Logs**: Look in `logs/` directory
2. **Enable Verbose**: Set `verbose=True` in `config.py`
3. **Review Examples**: See `data/shipment_questions_500.csv`
4. **GitHub Issues**: Report bugs or request features

---

## 🎉 Enjoy Your New Conversational AI Assistant!

The new interface makes logistics analytics as easy as having a conversation. Just ask what you want to know, and Hermes AI will help you discover insights from your data.

Happy analyzing! 🚀
