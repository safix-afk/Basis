"""
Basis Backend - FastAPI server for financial analysis code generation and execution
"""
import os
import sys
import io
import traceback
import warnings
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional, Dict, Any
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

app = FastAPI(title="Basis API", version="1.0.0")

# CORS middleware
# Allow localhost for development and Vercel for production
cors_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
# Add Vercel URL from environment if provided
vercel_url = os.getenv("VERCEL_URL")
if vercel_url:
    cors_origins.append(f"https://{vercel_url}")
    cors_origins.append(f"https://*.vercel.app")  # Allow all Vercel previews

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class GenerateRequest(BaseModel):
    prompt: str


class GenerateResponse(BaseModel):
    generated_code: str
    output_type: str  # 'plot' | 'text'
    output_data: str
    stdout: str = ""


class CodeExecutor:
    """Safely executes Python code and captures outputs"""
    
    def __init__(self):
        # Import required libraries in the execution context
        self.exec_globals = {
            '__builtins__': __builtins__,
            'yf': None,  # Will be imported
            'pd': None,  # Will be imported
            'np': None,  # Will be imported
            'go': None,  # Will be imported
            'fig_json': None,
        }
        
    def execute(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code and capture:
        - fig_json (Plotly figure as JSON string)
        - stdout (print statements)
        - stderr (errors)
        """
        # Import libraries in the execution context
        import yfinance as yf
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        import plotly.utils
        import json
        
        self.exec_globals.update({
            'yf': yf,
            'pd': pd,
            'np': np,
            'go': go,
            'plotly': __import__('plotly'),
            'json': json,
            'fig_json': None,
        })
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Suppress warnings to prevent them from being treated as errors
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            
            try:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(code, self.exec_globals)
                
                stdout_text = stdout_capture.getvalue()
                stderr_text = stderr_capture.getvalue()
                
                # Filter out warnings and progress bars from stderr - only treat actual errors as failures
                # yfinance outputs progress bars and warnings to stderr, which should not cause failure
                error_lines = []
                warning_lines = []
                
                if stderr_text:
                    for line in stderr_text.split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                        # Ignore yfinance progress bars - they output to stderr but are not errors
                        # Pattern: "[*********************100%***********************] 1 of 1 completed"
                        if any(pattern in line.lower() for pattern in ['[***', '100%', 'completed', 'downloading', ' of ']):
                            continue
                        # Also ignore lines that are just progress bars (contain brackets with asterisks and percentages)
                        if '[' in line and '*' in line and ('%' in line or 'completed' in line.lower()):
                            continue
                        # Ignore lines that match the yfinance progress pattern exactly
                        if line.count('*') > 10 and '%' in line:  # Progress bars have many asterisks
                            continue
                        # Check if it's a warning (FutureWarning, DeprecationWarning, etc.)
                        if any(warn_type in line for warn_type in ['Warning:', 'FutureWarning', 'DeprecationWarning', 'UserWarning']):
                            warning_lines.append(line)
                        # Ignore empty lines and progress indicators
                        elif line.startswith('[') and ('%' in line or '*' in line):
                            continue
                        else:
                            # Only treat non-warning, non-progress stderr as actual errors
                            error_lines.append(line)
                
                # Get fig_json if it exists and convert to a format react-plotly.js can handle
                fig_json = self.exec_globals.get('fig_json')
                if fig_json:
                    try:
                        # If it's a string (JSON), parse and convert binary data
                        fig_dict = json.loads(fig_json) if isinstance(fig_json, str) else fig_json
                        # Convert binary-encoded numpy arrays to regular lists and handle dates
                        if 'data' in fig_dict:
                            import base64
                            import numpy as np
                            import pandas as pd
                            for trace in fig_dict['data']:
                                # Handle y-axis data
                                if 'y' in trace:
                                    if isinstance(trace['y'], dict) and 'bdata' in trace['y']:
                                        # Binary encoded - decode
                                        try:
                                            bdata_str = trace['y']['bdata']
                                            # Handle unicode escapes
                                            if '\\u' in bdata_str:
                                                bdata_str = bdata_str.encode('utf-8').decode('unicode_escape').encode('latin1')
                                            else:
                                                bdata_str = bdata_str.encode('latin1')
                                            bdata = base64.b64decode(bdata_str)
                                            dtype_str = trace['y'].get('dtype', 'f8')
                                            dtype_map = {'f8': 'float64', 'i8': 'int64', 'f4': 'float32', 'i4': 'int32'}
                                            dtype = dtype_map.get(dtype_str, 'float64')
                                            arr = np.frombuffer(bdata, dtype=dtype)
                                            if 'shape' in trace['y']:
                                                shape = tuple(map(int, trace['y']['shape'].split(', ')))
                                                arr = arr.reshape(shape)
                                            # Flatten the array if it's 2D (e.g., shape like (127, 1))
                                            if arr.ndim > 1:
                                                arr = arr.flatten()
                                            # Convert to list and replace NaN/Inf with None
                                            import math
                                            y_list = arr.tolist() if hasattr(arr, 'tolist') else list(arr)
                                            trace['y'] = [None if (isinstance(y, float) and (math.isnan(y) or math.isinf(y))) else y for y in y_list]
                                        except Exception:
                                            pass  # Keep original if conversion fails
                                    elif isinstance(trace['y'], list):
                                        # Already a list, ensure it's clean and flattened
                                        # Check if it's nested (list of lists)
                                        if len(trace['y']) > 0 and isinstance(trace['y'][0], (list, np.ndarray)):
                                            # Flatten nested lists
                                            trace['y'] = [float(item) if isinstance(item, (int, float, np.number)) else (float(item[0]) if isinstance(item, (list, np.ndarray)) and len(item) > 0 else item) for item in trace['y']]
                                        else:
                                            # Flat list, just ensure types and replace NaN with None
                                            import math
                                            trace['y'] = [None if (isinstance(y, float) and (math.isnan(y) or math.isinf(y))) else (float(y) if isinstance(y, (int, float, np.number)) else y) for y in trace['y']]
                                
                                # Handle x-axis data (dates)
                                if 'x' in trace:
                                    if isinstance(trace.get('x'), dict) and 'bdata' in trace.get('x', {}):
                                        # Binary encoded - decode
                                        try:
                                            bdata_str = trace['x']['bdata']
                                            if '\\u' in bdata_str:
                                                bdata_str = bdata_str.encode('utf-8').decode('unicode_escape').encode('latin1')
                                            else:
                                                bdata_str = bdata_str.encode('latin1')
                                            bdata = base64.b64decode(bdata_str)
                                            dtype_str = trace['x'].get('dtype', 'f8')
                                            dtype_map = {'f8': 'float64', 'i8': 'int64', 'f4': 'float32', 'i4': 'int32', 'M8[ns]': 'datetime64[ns]'}
                                            dtype = dtype_map.get(dtype_str, 'float64')
                                            arr = np.frombuffer(bdata, dtype=dtype)
                                            if 'shape' in trace['x']:
                                                shape = tuple(map(int, trace['x']['shape'].split(', ')))
                                                arr = arr.reshape(shape)
                                            # Convert to date strings if datetime
                                            if 'datetime' in str(dtype) or dtype == 'datetime64[ns]':
                                                trace['x'] = [pd.Timestamp(ts).isoformat() for ts in arr]
                                            else:
                                                trace['x'] = arr.tolist() if hasattr(arr, 'tolist') else list(arr)
                                        except Exception:
                                            pass  # Keep original if conversion fails
                                    elif isinstance(trace.get('x'), list):
                                        # Convert numeric or string timestamps to ISO date strings if they look like timestamps
                                        x_data = trace['x']
                                        if len(x_data) > 0:
                                            first_val = x_data[0]
                                            first_val_float = None
                                            is_date_string = False
                                            
                                            # Check if values are numeric timestamps or string representations of large numbers
                                            if isinstance(first_val, (int, float, np.number, np.integer, np.floating)):
                                                first_val_float = float(first_val)
                                            elif isinstance(first_val, str):
                                                # Check if it's already an ISO date string
                                                if 'T' in first_val or (len(first_val) > 10 and first_val[:4].isdigit() and first_val[4] == '-'):
                                                    is_date_string = True
                                                else:
                                                    # Try to parse as number - might be string representation of nanosecond timestamp
                                                    try:
                                                        first_val_float = float(first_val)
                                                    except (ValueError, TypeError):
                                                        pass  # Not a number, keep as string
                                            
                                            # Only convert if not already a date string
                                            if not is_date_string and first_val_float is not None:
                                                # Pandas timestamps when converted to numpy become nanosecond integers
                                                if first_val_float > 1e15:  # Likely nanoseconds timestamp (pandas default)
                                                    try:
                                                        # Convert nanoseconds to ISO date strings
                                                        trace['x'] = [pd.Timestamp(int(float(ts)), unit='ns').strftime('%Y-%m-%dT%H:%M:%S') for ts in x_data]
                                                    except Exception as e:
                                                        try:
                                                            trace['x'] = [pd.Timestamp(int(float(ts)), unit='ns').isoformat() for ts in x_data]
                                                        except:
                                                            trace['x'] = [str(x) for x in x_data]
                                                elif first_val_float > 1e9:  # Likely seconds timestamp
                                                    try:
                                                        trace['x'] = [pd.Timestamp(int(float(ts)), unit='s').strftime('%Y-%m-%dT%H:%M:%S') for ts in x_data]
                                                    except:
                                                        try:
                                                            trace['x'] = [pd.Timestamp(int(float(ts)), unit='s').isoformat() for ts in x_data]
                                                        except:
                                                            trace['x'] = [str(x) for x in x_data]
                                                else:
                                                    trace['x'] = [float(x) if isinstance(x, (int, float, np.number)) else str(x) for x in x_data]
                                            elif is_date_string:
                                                # Already date strings, ensure all are strings
                                                trace['x'] = [str(x) for x in x_data]
                                            else:
                                                # Other types, ensure they're strings
                                                trace['x'] = [str(x) for x in x_data]
                        # Convert NaN, Infinity values to null for JSON compatibility
                        import math
                        def clean_for_json(obj):
                            if isinstance(obj, dict):
                                return {k: clean_for_json(v) for k, v in obj.items()}
                            elif isinstance(obj, list):
                                return [clean_for_json(item) for item in obj]
                            elif isinstance(obj, float):
                                if math.isnan(obj) or math.isinf(obj):
                                    return None
                                return obj
                            elif isinstance(obj, np.floating):
                                if math.isnan(float(obj)) or math.isinf(float(obj)):
                                    return None
                                return float(obj)
                            elif isinstance(obj, np.integer):
                                return int(obj)
                            return obj
                        
                        fig_dict = clean_for_json(fig_dict)
                        fig_json = json.dumps(fig_dict)
                    except Exception:
                        pass  # Use original fig_json if conversion fails
                
                # If we have fig_json or stdout, the code executed successfully
                # Only treat as error if we have no successful output AND there are actual error lines
                if error_lines and not fig_json and not stdout_text:
                    return {
                        'success': False,
                        'error': '\n'.join(error_lines),
                        'stdout': stdout_text,
                        'fig_json': None
                    }
                # If we have fig_json or stdout, it's a success (stderr was just noise like progress bars)
                
                # Success - warnings are included in stdout for visibility
                if warning_lines:
                    stdout_text = stdout_text + '\n' + '\n'.join(warning_lines) if stdout_text else '\n'.join(warning_lines)
                
                return {
                    'success': True,
                    'fig_json': fig_json,
                    'stdout': stdout_text,
                    'error': None
                }
                
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                return {
                    'success': False,
                    'error': error_msg,
                    'stdout': stdout_capture.getvalue(),
                    'fig_json': None
                }


def generate_code_with_llm(prompt: str) -> str:
    """
    Use OpenAI GPT-4o to generate Python code for financial analysis
    """
    system_prompt = """You are a financial data analyst expert. Generate Python code to answer financial questions.

CRITICAL RULES:
1. ALWAYS use `yfinance` imported as `yf` for stock data
2. ALWAYS use `plotly.graph_objects` imported as `go` for plotting (NEVER matplotlib)
3. ALWAYS save the Plotly figure as JSON to a variable named `fig_json` using: `import json; import pandas as pd; fig_dict = fig.to_dict(); [trace.update({'x': [str(pd.Timestamp(ts)) if isinstance(ts, (pd.Timestamp, type(pd.Timestamp.now()))) or (isinstance(ts, str) and 'T' in ts) else str(ts) for ts in (trace['x'].tolist() if hasattr(trace.get('x'), 'tolist') else trace.get('x', []))], 'y': trace['y'].tolist() if hasattr(trace.get('y'), 'tolist') else trace.get('y')}) for trace in fig_dict['data']]; fig_json = json.dumps(fig_dict)`
4. If the user asks for text output, use `print()` statements
5. Use pandas (imported as `pd`) and numpy (imported as `np`) for data manipulation
6. When using `yf.download()`, include `auto_adjust=True` parameter to avoid warnings: `yf.download('TICKER', period='6mo', auto_adjust=True)`
7. Suppress warnings at the start of code: `import warnings; warnings.filterwarnings('ignore')`

Example patterns:
- For single stock: `data = yf.download('AAPL', period='6mo', auto_adjust=True)`
- For multiple stocks comparison: `data = yf.download(['AAPL', 'MSFT'], period='1y', auto_adjust=True); close_prices = data['Close']`
- For correlation between two stocks: `data = yf.download(['AAPL', 'MSFT'], period='1y', auto_adjust=True); close_prices = data['Close']; rolling_corr = close_prices['AAPL'].rolling(window=20).corr(close_prices['MSFT']); rolling_corr = rolling_corr.fillna(0)` (fill NaN with 0 or dropna() before plotting)
- For drawdown: `drawdown = (price / price.cummax()) - 1`
- For plotting: Always create a `go.Figure()` and save to `fig_json` using the conversion pattern in rule 3
- When downloading multiple tickers, use a list: `yf.download(['TICKER1', 'TICKER2'], period='1y')` which returns a MultiIndex DataFrame
- To access Close prices for multiple stocks: `data = yf.download(['AAPL', 'MSFT'], period='1y', auto_adjust=True); close_prices = data['Close']` gives a DataFrame with columns 'AAPL' and 'MSFT'

Return ONLY the Python code, no explanations, no markdown formatting."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        code = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()
        
        return code
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


executor = CodeExecutor()


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_and_execute(request: GenerateRequest):
    """
    Main endpoint: Generate code from prompt, execute it, and return results
    """
    try:
        # Step A: Generate code with LLM
        generated_code = generate_code_with_llm(request.prompt)
        
        # Step B: Execute the code
        execution_result = executor.execute(generated_code)
        
        if not execution_result['success']:
            # Return error but still show the code
            return GenerateResponse(
                generated_code=generated_code,
                output_type="text",
                output_data=f"Error: {execution_result['error']}",
                stdout=execution_result['stdout']
            )
        
        # Step C: Determine output type and format response
        fig_json = execution_result['fig_json']
        stdout = execution_result['stdout']
        
        if fig_json:
            # We have a Plotly chart (already converted from binary format in executor)
            return GenerateResponse(
                generated_code=generated_code,
                output_type="plot",
                output_data=fig_json,
                stdout=stdout
            )
        elif stdout:
            # We have text output
            return GenerateResponse(
                generated_code=generated_code,
                output_type="text",
                output_data=stdout,
                stdout=stdout
            )
        else:
            # No output, but execution succeeded
            return GenerateResponse(
                generated_code=generated_code,
                output_type="text",
                output_data="Code executed successfully (no output)",
                stdout=""
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

