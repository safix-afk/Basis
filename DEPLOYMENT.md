# Deployment Guide for Basis

This guide will help you deploy Basis for free to show Y Combinator.

## Recommended Setup: Vercel (Frontend) + Railway (Backend)

### Option 1: Vercel + Railway (Recommended)

#### Frontend on Vercel (Free)

1. **Push your code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy to Vercel:**
   - Go to [vercel.com](https://vercel.com)
   - Sign up with GitHub
   - Click "New Project"
   - Import your repository
   - **Root Directory:** Set to `frontend`
   - **Framework Preset:** Next.js (auto-detected)
   - **Environment Variables:** Add `NEXT_PUBLIC_API_URL=https://your-railway-app.railway.app` (you'll get this after deploying backend)
   - Click "Deploy"

3. **Update API URL:**
   - After deployment, go to Project Settings → Environment Variables
   - Add: `NEXT_PUBLIC_API_URL` = your Railway backend URL

#### Backend on Railway (Free Tier Available)

1. **Create Railway account:**
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub

2. **Create new project:**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Select the `backend` folder

3. **Configure the service:**
   - Railway will auto-detect Python
   - Add environment variable: `OPENAI_API_KEY=your_key_here`
   - Railway will automatically install dependencies from `requirements.txt`

4. **Get your backend URL:**
   - Railway will provide a URL like: `https://your-app.railway.app`
   - Update your Vercel environment variable with this URL

5. **Update CORS in backend:**
   - In `backend/main.py`, update the CORS origins to include your Vercel URL:
   ```python
   allow_origins=[
       "http://localhost:3000",
       "https://your-app.vercel.app"  # Add your Vercel URL
   ]
   ```

---

### Option 2: Render (All-in-One)

#### Deploy Both on Render

1. **Backend on Render:**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub
   - Click "New +" → "Web Service"
   - Connect your GitHub repo
   - **Root Directory:** `backend`
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Add environment variable: `OPENAI_API_KEY`
   - Click "Create Web Service"

2. **Frontend on Render:**
   - Click "New +" → "Static Site"
   - Connect your GitHub repo
   - **Root Directory:** `frontend`
   - **Build Command:** `npm install && npm run build`
   - **Publish Directory:** `.next`
   - Add environment variable: `NEXT_PUBLIC_API_URL=https://your-backend.onrender.com`
   - Click "Create Static Site"

---

### Option 3: Fly.io (Alternative)

#### Deploy Both on Fly.io

1. **Install Fly CLI:**
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Backend:**
   ```bash
   cd backend
   fly launch
   # Follow prompts, select Python
   fly secrets set OPENAI_API_KEY=your_key_here
   fly deploy
   ```

3. **Frontend:**
   ```bash
   cd frontend
   fly launch
   # Follow prompts, select Node.js
   fly secrets set NEXT_PUBLIC_API_URL=https://your-backend.fly.dev
   fly deploy
   ```

---

## Quick Setup Scripts

### For Railway Backend

Create `backend/railway.json`:
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

### Update Backend for Production

Update `backend/main.py` CORS settings:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-app.vercel.app",  # Your Vercel URL
        "https://*.vercel.app",  # Or allow all Vercel previews
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Update Frontend API URL

Create `frontend/.env.production`:
```
NEXT_PUBLIC_API_URL=https://your-backend.railway.app
```

Update `frontend/app/page.tsx` to use environment variable:
```typescript
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
// Then use API_URL in fetch calls
```

---

## Recommended: Vercel + Railway

**Why this combination:**
- ✅ Vercel is made by Next.js creators - perfect integration
- ✅ Railway has generous free tier (500 hours/month)
- ✅ Both have easy GitHub integration
- ✅ Fast deployments
- ✅ Free SSL certificates
- ✅ Good for demos

**Free Tier Limits:**
- **Vercel:** Unlimited for personal projects, 100GB bandwidth
- **Railway:** $5 free credit/month (usually enough for demos)

---

## Steps Summary

1. Push code to GitHub
2. Deploy backend to Railway → Get URL
3. Deploy frontend to Vercel → Add backend URL as env var
4. Update CORS in backend with Vercel URL
5. Test the deployment!

---

## Troubleshooting

**CORS Errors:**
- Make sure backend CORS includes your frontend URL
- Check that environment variables are set correctly

**API Not Found:**
- Verify `NEXT_PUBLIC_API_URL` is set in Vercel
- Check that backend is running on Railway
- Test backend URL directly: `https://your-backend.railway.app/health`

**Build Failures:**
- Check Railway/Render logs
- Ensure `requirements.txt` has all dependencies
- Verify Python version compatibility

