# Basis - Financial Analysis Workspace

A financial analysis IDE that functions like "Cursor-for-Finance". Ask natural language questions about markets, and the system generates and executes Python code to answer them.

## Project Structure

```
Basis/
├── backend/          # FastAPI server
│   ├── main.py      # Main API server with code generation and execution
│   └── requirements.txt
└── frontend/        # Next.js application
    ├── app/         # Next.js app router pages
    ├── components/  # React components
    └── lib/         # Utility functions
```

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the `backend/` directory:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

5. Start the FastAPI server:
```bash
uvicorn main:app --reload --port 8000
```

The backend will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Features

- **Natural Language Queries**: Ask questions about financial markets in plain English
- **AI Code Generation**: GPT-4o generates Python code using yfinance and plotly
- **Live Code Execution**: Backend safely executes generated code
- **Interactive Charts**: Plotly visualizations rendered in real-time
- **Code Editor**: Monaco Editor (VS Code-like) for viewing and editing generated code
- **Chat History**: Sidebar with previous queries and results
- **Dark Mode UI**: Bloomberg Terminal-inspired dark theme

## Example Queries

- "Show me AAPL stock price over the last 6 months"
- "Compare the correlation between AAPL and MSFT over the past year"
- "Calculate the drawdown for TSLA stock"
- "Show me the volume profile for NVDA"

## Tech Stack

### Backend
- FastAPI
- OpenAI API (GPT-4o)
- yfinance
- pandas
- numpy
- plotly

### Frontend
- Next.js 14 (App Router)
- TypeScript
- Tailwind CSS
- Monaco Editor
- react-plotly.js
- Shadcn/UI components
- Lucide React icons

## API Endpoints

### POST `/api/generate`

Generate and execute Python code from a natural language prompt.

**Request:**
```json
{
  "prompt": "Show me AAPL stock price over the last 6 months"
}
```

**Response:**
```json
{
  "generated_code": "import yfinance as yf\n...",
  "output_type": "plot",
  "output_data": "{...plotly json...}",
  "stdout": ""
}
```

### GET `/health`

Health check endpoint.

## Development Notes

- The backend uses `exec()` to run generated code. In production, consider using sandboxed execution environments.
- The system prompt is designed to prevent hallucination of unsupported libraries (e.g., matplotlib).
- All plots must be saved to `fig_json` variable using `fig.to_json()`.

## License

MIT

