# Deployment Guide for Basis

This guide will help you deploy Basis for free to show Y Combinator.

## 🆓 FREE Hosting Options (No Credit Card Required)

### Recommended Setup: Vercel (Frontend) + Render/Fly.io (Backend)

**Best Free Options for Backend:**
1. **Render** - Free tier, spins down after 15min inactivity (good for demos)
2. **Fly.io** - Free tier with 3 shared VMs (160GB/month)
3. **Koyeb** - Free tier with 512MB RAM
4. **PythonAnywhere** - Free tier (limited but works)

### Option 1: Vercel + Render (FREE - Recommended for Demos)

#### Why Render?
- ✅ **Completely FREE** - No credit card required
- ✅ Easy GitHub integration
- ✅ Auto-deploys on push
- ✅ Free SSL certificate
- ⚠️ Spins down after 15min inactivity (wakes up on first request - takes ~30 seconds)

#### Backend on Render (FREE)

1. **Create Render account:**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub (FREE, no credit card needed)

2. **Create new Web Service:**
   - Click "New +" → "Web Service"
   - Connect your GitHub account
   - Select the "Basis" repository
   - **Settings:**
     - **Name:** `basis-backend` (or any name)
     - **Root Directory:** `backend`
     - **Environment:** `Python 3`
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
     - **Plan:** Select **FREE** (not Starter)
   
3. **Add Environment Variables:**
   - Click "Environment" tab
   - Add: `OPENAI_API_KEY` = your OpenAI API key
   - Add: `PORT` = `10000` (Render free tier uses port 10000)

4. **Deploy:**
   - Click "Create Web Service"
   - Render will build and deploy (takes ~5 minutes)
   - Copy your service URL (e.g., `https://basis-backend.onrender.com`)

5. **Update CORS:**
   - In `backend/main.py`, the CORS is already set up to accept Render URLs
   - Or add your Render URL manually if needed

#### Frontend on Vercel (FREE)

1. **Deploy to Vercel:**
   - Go to [vercel.com](https://vercel.com)
   - Sign up with GitHub
   - Click "New Project"
   - Import "Basis" repository
   - **Settings:**
     - **Root Directory:** `frontend`
     - **Framework Preset:** Next.js (auto-detected)
   
2. **Add Environment Variable:**
   - Go to Project Settings → Environment Variables
   - Add: `NEXT_PUBLIC_API_URL` = `https://your-backend.onrender.com`
   - Click "Save"

3. **Deploy:**
   - Click "Deploy"
   - Vercel will build and deploy (takes ~2 minutes)
   - Your app will be live at `https://your-app.vercel.app`

---

### Option 2: Vercel + Fly.io (FREE - Always On)

#### Why Fly.io?
- ✅ **Completely FREE** - No credit card required
- ✅ Always on (doesn't spin down)
- ✅ Fast cold starts
- ✅ 3 shared VMs free

#### Backend on Fly.io (FREE)

1. **Install Fly CLI:**
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login to Fly:**
   ```bash
   fly auth login
   ```

3. **Deploy Backend:**
   ```bash
   cd backend
   fly launch
   # When prompted:
   # - App name: basis-backend (or any unique name)
   # - Region: choose closest to you
   # - Don't deploy yet (we'll add secrets first)
   ```

4. **Add Secrets:**
   ```bash
   fly secrets set OPENAI_API_KEY=your_key_here
   ```

5. **Deploy:**
   ```bash
   fly deploy
   ```

6. **Get your URL:**
   ```bash
   fly info
   # Your URL will be: https://basis-backend.fly.dev
   ```

#### Frontend on Vercel
- Same as Option 1 above, but use your Fly.io URL

---

### Option 3: Vercel + Koyeb (FREE)

#### Backend on Koyeb (FREE)

1. **Create account:**
   - Go to [koyeb.com](https://www.koyeb.com)
   - Sign up with GitHub

2. **Create App:**
   - Click "Create App"
   - Select "GitHub" → Choose "Basis" repo
   - **Settings:**
     - **Name:** `basis-backend`
     - **Root Directory:** `backend`
     - **Build Command:** `pip install -r requirements.txt`
     - **Run Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
     - **Plan:** FREE

3. **Add Environment Variables:**
   - `OPENAI_API_KEY` = your key
   - `PORT` = `8000`

4. **Deploy:**
   - Click "Deploy"
   - Get your URL: `https://basis-backend-xxxxx.koyeb.app`

---

### Option 4: Vercel + PythonAnywhere (FREE - Limited)

#### Backend on PythonAnywhere (FREE)

1. **Create account:**
   - Go to [pythonanywhere.com](https://www.pythonanywhere.com)
   - Sign up (FREE tier)

2. **Upload code:**
   - Go to Files tab
   - Upload your `backend` folder files
   - Or use Git: `git clone https://github.com/safix-afk/Basis.git`

3. **Install dependencies:**
   - Go to Bash console
   ```bash
   cd Basis/backend
   pip3.10 install --user -r requirements.txt
   ```

4. **Create Web App:**
   - Go to Web tab
   - Click "Add a new web app"
   - Choose Flask (we'll modify it)
   - Set source code to your backend folder

5. **Configure:**
   - Edit WSGI file to run FastAPI
   - Add environment variables in Web tab

**Note:** PythonAnywhere free tier has limitations (limited CPU time, external requests), but works for demos.

---

### Option 5: All-in-One on Render (FREE)

Deploy both frontend and backend on Render:

#### Backend (Web Service):
- Same as Option 1 above

#### Frontend (Static Site):
1. **Create Static Site:**
   - Click "New +" → "Static Site"
   - Connect GitHub → Select "Basis"
   - **Root Directory:** `frontend`
   - **Build Command:** `npm install && npm run build`
   - **Publish Directory:** `.next`
   - Add env var: `NEXT_PUBLIC_API_URL=https://your-backend.onrender.com`

---

## 🎯 Quick Comparison

| Platform | Free Tier | Always On | Cold Start | Best For |
|----------|-----------|-----------|------------|----------|
| **Render** | ✅ Yes | ❌ No (15min timeout) | ~30s | Demos, testing |
| **Fly.io** | ✅ Yes | ✅ Yes | ~1s | Production demos |
| **Koyeb** | ✅ Yes | ✅ Yes | ~2s | Production demos |
| **PythonAnywhere** | ✅ Yes | ⚠️ Limited | Fast | Simple demos |

## 🚀 Recommended for YC Demo: Vercel + Render

**Why?**
- ✅ Both completely free
- ✅ No credit card needed
- ✅ Easy setup (15 minutes)
- ✅ Professional URLs
- ✅ Auto-deploys from GitHub
- ⚠️ First request after 15min takes ~30s (but then stays warm)

---

### Option 1: Vercel + Railway (Paid after trial)

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

