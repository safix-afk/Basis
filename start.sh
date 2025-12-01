#!/bin/bash
# Basis Startup Script
# This script starts both the backend and frontend servers

echo "Starting Basis..."

# Check if .env exists in backend
if [ ! -f "backend/.env" ]; then
    echo "⚠️  Warning: backend/.env not found. Please create it with your OPENAI_API_KEY"
    echo "   Example: echo 'OPENAI_API_KEY=your_key_here' > backend/.env"
fi

# Start backend in background
echo "Starting backend server on port 8000..."
cd backend
python -m uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!
cd ..

# Wait a bit for backend to start
sleep 2

# Start frontend
echo "Starting frontend server on port 3000..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Basis is running!"
echo "   Backend: http://localhost:8000"
echo "   Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for user interrupt
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
