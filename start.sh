#!/bin/bash
# Start script for Render - runs from repo root
cd backend
uvicorn main:app --host 0.0.0.0 --port $PORT
