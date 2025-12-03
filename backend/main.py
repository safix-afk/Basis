"""
Basis Backend - FastAPI server for financial analysis code generation and execution
"""
import os
import sys
import io
import re
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

# For production, allow all Vercel domains using regex pattern
# FastAPI's CORSMiddleware supports allow_origin_regex for pattern matching
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=r"https://.*\.vercel\.(app|dev)",
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
                        # If it's a Plotly figure object, convert to dict first
                        if hasattr(fig_json, 'to_dict'):
                            fig_dict = fig_json.to_dict()
                        # If it's a string (JSON), parse and convert binary data
                        elif isinstance(fig_json, str):
                            fig_dict = json.loads(fig_json)
                        else:
                            fig_dict = fig_json
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
                        # Also handle numpy arrays recursively
                        import math
                        def clean_for_json(obj):
                            if isinstance(obj, dict):
                                return {k: clean_for_json(v) for k, v in obj.items()}
                            elif isinstance(obj, list):
                                return [clean_for_json(item) for item in obj]
                            elif isinstance(obj, np.ndarray):
                                # Convert numpy array to list recursively
                                return clean_for_json(obj.tolist())
                            elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                                return int(obj)
                            elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
                                if math.isnan(float(obj)) or math.isinf(float(obj)):
                                    return None
                                return float(obj)
                            elif isinstance(obj, (pd.Timestamp, pd.DatetimeTZDtype)):
                                return obj.isoformat()
                            elif isinstance(obj, float):
                                if math.isnan(obj) or math.isinf(obj):
                                    return None
                                return obj
                            elif hasattr(obj, 'tolist'):
                                # Handle other array-like objects (e.g., pandas Index)
                                return clean_for_json(obj.tolist())
                            return obj
                        
                        fig_dict = clean_for_json(fig_dict)
                        fig_json = json.dumps(fig_dict)
                    except Exception as e:
                        # If conversion fails, try to convert Figure object to dict and then JSON
                        try:
                            if hasattr(fig_json, 'to_dict'):
                                fig_dict = fig_json.to_dict()
                                fig_dict = clean_for_json(fig_dict)
                                fig_json = json.dumps(fig_dict)
                            elif isinstance(fig_json, dict):
                                fig_dict = clean_for_json(fig_json)
                                fig_json = json.dumps(fig_dict)
                            else:
                                # Last resort: convert to string
                                fig_json = json.dumps(str(fig_json))
                        except Exception:
                            # If all else fails, convert to string representation
                            fig_json = json.dumps({"error": "Failed to serialize figure", "data": str(fig_json)})
                
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


def sanitize_generated_code(code: str) -> str:
    """
    Fix common syntax errors in LLM-generated code
    Uses multiple passes to ensure all errors are caught
    """
    # Pass 1: Fix basic spacing and assignment issues
    lines = code.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        original_line = line
        stripped = line.strip()
        
        # Skip comments
        if stripped.startswith('#'):
            fixed_lines.append(line)
            continue
        
        # Fix 1: Missing assignment operator - catch all variations
        # Pattern: "data yf.download (" or "data yf.download(" -> "data = yf.download("
        # This must come BEFORE removing spaces before parens
        # Match: variable_name whitespace library.method optional_space optional_paren
        line = re.sub(
            r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s+(yf|pd|np|go|fig)\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
            r'\1 = \2.\3(',
            line
        )
        
        # Fix 1b: Missing assignment for go.Figure with space after dot
        # Pattern: "fig go. Figure(" -> "fig = go.Figure("
        line = re.sub(
            r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s+go\.\s+Figure\s*\(',
            r'\1 = go.Figure(',
            line
        )
        
        # Fix 4: Spaces after dots: "go. Figure" -> "go.Figure"
        line = re.sub(r'go\.\s+Figure', 'go.Figure', line)
        line = re.sub(r'pd\.\s+Timestamp', 'pd.Timestamp', line)
        line = re.sub(r'np\.\s+', 'np.', line)
        # General pattern for any module. method
        line = re.sub(r'(\w+)\.\s+(\w+)', r'\1.\2', line)
        
        # Fix 5: Spaces before opening parentheses
        # Pattern: "filterwarnings (" -> "filterwarnings("
        # Be careful - only fix if it's clearly a function call
        # Match: word boundary, word chars, space, opening paren
        line = re.sub(r'\b(\w+)\s+\(', r'\1(', line)
        
        # Fix 6: Spaces in square brackets: "trace ['x']" -> "trace['x']"
        line = re.sub(r'(\w+)\s+\[', r'\1[', line)
        
        # Fix 7: Unterminated strings in function calls
        # Pattern: "('ignore)" -> "('ignore')"
        # Match: opening paren, optional space, quote, text (no quote), closing paren
        line = re.sub(r"\(\s*'([^')]+)\)", r"('\1')", line)
        
        # Fix 7b: Unterminated strings before closing parens (e.g., "yaxis_title='text)" -> "yaxis_title='text')")
        # Pattern: quote, text, closing paren without closing quote
        line = re.sub(r"='([^']+)\)", r"='\1')", line)
        line = re.sub(r":\s*'([^']+)\)", r": '\1')", line)
        line = re.sub(r"\s+'([^']+)\)", r" '\1')", line)
        
        # Fix 8: Check for unterminated strings
        single_quotes = len([m for m in re.finditer(r"(?<!\\)'", line)])
        if single_quotes % 2 == 1:
            # Odd number of quotes = unterminated string
            if re.search(r"\(\s*'[^']*$", line):
                # Function call with unterminated string at end
                open_parens = line.count('(')
                close_parens = line.count(')')
                if open_parens > close_parens:
                    line = line.rstrip() + "')"
                elif not stripped.endswith("')"):
                    line = line.rstrip() + "'"
        
        # Fix 9: Extra quotes before closing parens
        line = re.sub(r"True'\)", "True)", line)
        line = re.sub(r"False'\)", "False)", line)
        line = re.sub(r"=True'\)", "=True)", line)
        line = re.sub(r"=False'\)", "=False)", line)
        if re.search(r"'\)\s*$", line):
            line = re.sub(r"'\)\s*$", ")", line)
        
        # Fix 10: Handle incomplete lines that might cause syntax errors
        # If a line ends with an incomplete function call like "type(", try to complete it
        # But be conservative - only fix obvious cases
        if stripped.endswith('type(') or (stripped.endswith('(') and 'type(' in stripped and stripped.count('(') > stripped.count(')')):
            # This is likely an incomplete line - we can't safely complete it
            # But we can at least ensure it doesn't break the rest of the code
            # If it's clearly incomplete, we might need to comment it out or add a pass
            # For now, just ensure proper formatting
            pass
        
        fixed_lines.append(line)
    
    code = '\n'.join(fixed_lines)
    
    # Pass 2: Try to compile and fix any remaining syntax errors
    max_passes = 3
    for pass_num in range(max_passes):
        try:
            compile(code, '<string>', 'exec')
            break  # Success, no more fixes needed
        except SyntaxError as e:
            if pass_num == max_passes - 1:
                # Last pass, give up
                break
            
            # Try to fix the error line
            lines = code.split('\n')
            error_line_num = (e.lineno or 1) - 1  # Convert to 0-based index
            if 0 <= error_line_num < len(lines):
                error_line = lines[error_line_num]
                fixed_line = error_line
                
                # Apply aggressive fixes
                # Fix missing assignment
                fixed_line = re.sub(r'\b(\w+)\s+(yf|pd|np|go|fig)\.', r'\1 = \2.', fixed_line)
                # Fix spaces before parens
                fixed_line = re.sub(r'(\w+)\s+\(', r'\1(', fixed_line)
                # Fix spaces after dots
                fixed_line = re.sub(r'(\w+)\.\s+(\w+)', r'\1.\2', fixed_line)
                # Fix unterminated strings
                fixed_line = re.sub(r"\(\s*'([^')]+)\)", r"('\1')", fixed_line)
                # Fix spaces in brackets
                fixed_line = re.sub(r'(\w+)\s+\[', r'\1[', fixed_line)
                
                # Handle incomplete lines - if line ends with incomplete function call
                stripped_fixed = fixed_line.strip()
                if stripped_fixed.endswith('(') and stripped_fixed.count('(') > stripped_fixed.count(')'):
                    # Incomplete function call - try to complete common patterns
                    if 'type(' in stripped_fixed and stripped_fixed.endswith('type('):
                        # Complete: type(pd.Timestamp.now())
                        fixed_line = fixed_line.rstrip() + "pd.Timestamp.now()))"
                    elif 'isinstance' in stripped_fixed:
                        # Complete isinstance call
                        fixed_line = fixed_line.rstrip() + "pd.Timestamp))"
                    else:
                        # Generic completion - add closing parens
                        missing_parens = stripped_fixed.count('(') - stripped_fixed.count(')')
                        fixed_line = fixed_line.rstrip() + ')' * missing_parens
                
                # Fix unterminated strings at end
                if "'" in fixed_line:
                    single_quotes = len([m for m in re.finditer(r"(?<!\\)'", fixed_line)])
                    if single_quotes % 2 == 1:
                        if re.search(r"\(\s*'[^']*$", fixed_line):
                            open_parens = fixed_line.count('(')
                            close_parens = fixed_line.count(')')
                            if open_parens > close_parens:
                                fixed_line = fixed_line.rstrip() + "')"
                            else:
                                fixed_line = fixed_line.rstrip() + "'"
                
                if fixed_line != error_line:
                    lines[error_line_num] = fixed_line
                    code = '\n'.join(lines)
    
    return code


def generate_code_with_llm(prompt: str) -> str:
    """
    Use OpenAI GPT-4o to generate Python code for financial analysis
    """
    system_prompt = """You are a financial data analyst expert. Generate Python code to answer financial questions.

CRITICAL SYNTAX RULES - FOLLOW EXACTLY:
1. ALWAYS use assignment operator: `data = yf.download(...)` NOT `data yf.download(...)`
2. ALWAYS use correct quotes: `yf.download('TICKER', period='6mo', auto_adjust=True)` NOT `yf.download('TICKER', period='6mo', auto_adjust=True')`
3. NO spaces in module access: `go.Figure()` NOT `go. Figure()`
4. ALWAYS close all parentheses and quotes properly

CRITICAL FUNCTIONAL RULES:
1. ALWAYS use `yfinance` imported as `yf` for stock data
2. ALWAYS use `plotly.graph_objects` imported as `go` for plotting (NEVER matplotlib)
3. For plotting, ALWAYS save the Plotly figure to a variable named `fig_json`. You can either:
   - Option A (Simplest - Recommended): Just assign the figure object directly: `fig_json = fig`
   - Option B: Convert to dict: `fig_json = fig.to_dict()`
   DO NOT try to convert to JSON string yourself - the backend will handle all numpy array and date conversions automatically.
   Example:
   ```python
   fig = go.Figure(data=[go.Scatter(x=data.index, y=data['Close'])])
   fig.update_layout(title='Stock Price')
   fig_json = fig  # Just assign the figure object - backend handles conversion
   ```
4. If the user asks for text output, use `print()` statements
5. Use pandas (imported as `pd`) and numpy (imported as `np`) for data manipulation
6. When using `yf.download()`, include `auto_adjust=True` parameter: `data = yf.download('TICKER', period='6mo', auto_adjust=True)`
7. Suppress warnings at the start: `import warnings; warnings.filterwarnings('ignore')`

Example patterns:
- Single stock: `data = yf.download('AAPL', period='6mo', auto_adjust=True)`
- Multiple stocks: `data = yf.download(['AAPL', 'MSFT'], period='1y', auto_adjust=True); close_prices = data['Close']`
- Plotting: Create `fig = go.Figure(...)` then convert to `fig_json` using the pattern above

Return ONLY valid Python code, no explanations, no markdown formatting. Check your syntax carefully."""

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
        
        # Sanitize the code to fix common syntax errors
        code = sanitize_generated_code(code)
        
        # Validate syntax by trying to compile it
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            # If there's still a syntax error, try to fix it more aggressively
            # This is a fallback for cases the regex didn't catch
            lines = code.split('\n')
            error_line_num = e.lineno or 1
            if error_line_num <= len(lines):
                error_line = lines[error_line_num - 1]
                # Try common fixes on the error line
                fixed_error_line = error_line
                # Fix missing = operator more aggressively
                fixed_error_line = re.sub(r'(\w+)\s+(yf|pd|np|go|fig)\.', r'\1 = \2.', fixed_error_line)
                # Fix spaces before parentheses
                fixed_error_line = re.sub(r'(\w+)\s+\(', r'\1(', fixed_error_line)
                # Fix spaces after dots
                fixed_error_line = re.sub(r'(\w+)\.\s+(\w+)', r'\1.\2', fixed_error_line)
                # Fix unterminated strings: function('text) -> function('text')
                fixed_error_line = re.sub(r"\(\s*'([^')]+)\)", r"('\1')", fixed_error_line)
                # Fix unterminated strings before closing parens (e.g., "yaxis_title='text)" -> "yaxis_title='text')")
                fixed_error_line = re.sub(r"='([^']+)\)", r"='\1')", fixed_error_line)
                fixed_error_line = re.sub(r":\s*'([^']+)\)", r": '\1')", fixed_error_line)
                fixed_error_line = re.sub(r"\s+'([^']+)\)", r" '\1')", fixed_error_line)
                # Fix unterminated strings at end: function('text -> function('text')
                if "'" in fixed_error_line:
                    single_quotes = len([m for m in re.finditer(r"(?<!\\)'", fixed_error_line)])
                    if single_quotes % 2 == 1:
                        # Odd number of quotes = unterminated string
                        if re.search(r"\(\s*'[^']*$", fixed_error_line):
                            # Function call with unterminated string
                            open_parens = fixed_error_line.count('(')
                            close_parens = fixed_error_line.count(')')
                            if open_parens > close_parens:
                                fixed_error_line = fixed_error_line.rstrip() + "')"
                            else:
                                fixed_error_line = fixed_error_line.rstrip() + "'"
                
                if fixed_error_line != error_line:
                    lines[error_line_num - 1] = fixed_error_line
                    code = '\n'.join(lines)
                    # Try compiling again
                    try:
                        compile(code, '<string>', 'exec')
                    except SyntaxError:
                        pass  # If it still fails, return the sanitized code anyway
        
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
            # Ensure fig_json is always a string (JSON) for the response model
            if not isinstance(fig_json, str):
                # If it's still a Figure object or dict, convert it
                try:
                    import json
                    import plotly.graph_objects as go
                    if hasattr(fig_json, 'to_dict'):
                        fig_dict = fig_json.to_dict()
                        # Use the executor's clean_for_json logic
                        import numpy as np
                        import pandas as pd
                        import math
                        def clean_for_json(obj):
                            if isinstance(obj, dict):
                                return {k: clean_for_json(v) for k, v in obj.items()}
                            elif isinstance(obj, list):
                                return [clean_for_json(item) for item in obj]
                            elif isinstance(obj, np.ndarray):
                                return clean_for_json(obj.tolist())
                            elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                                return int(obj)
                            elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
                                if math.isnan(float(obj)) or math.isinf(float(obj)):
                                    return None
                                return float(obj)
                            elif isinstance(obj, (pd.Timestamp, pd.DatetimeTZDtype)):
                                return obj.isoformat()
                            elif isinstance(obj, float):
                                if math.isnan(obj) or math.isinf(obj):
                                    return None
                                return obj
                            elif hasattr(obj, 'tolist'):
                                return clean_for_json(obj.tolist())
                            return obj
                        fig_dict = clean_for_json(fig_dict)
                        fig_json = json.dumps(fig_dict)
                    elif isinstance(fig_json, dict):
                        fig_json = json.dumps(fig_json)
                    else:
                        fig_json = json.dumps(str(fig_json))
                except Exception as e:
                    fig_json = json.dumps({"error": f"Failed to serialize figure: {str(e)}"})
            
            # We have a Plotly chart (already converted to JSON string)
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

