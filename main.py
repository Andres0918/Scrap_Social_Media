"""
BACKEND - Social Media Sentiment Analysis API
FastAPI application with concurrent scraping and real-time updates
"""

from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import json
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
import pandas as pd
import numpy as np
import os
from collections import Counter

# Import your existing scraping functions
import sys

sys.path.append('..')
from Scraping import (
    scrap_linkedin_playwright,
    scrap_facebook_playwright,
    scrap_x_playwright,
    scrap_instagram_playwright,
    process_nlp_row
)

app = FastAPI(title="Social Sentiment Analyzer", version="1.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Connection
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGO_URL)
db = client.social_sentiment_db


# =====================================================================
# NUMPY TO PYTHON CONVERTER (FIX MONGODB ERROR)
# =====================================================================

def convert_numpy_to_python(obj):
    """Convierte tipos numpy a tipos Python nativos para MongoDB"""
    if isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj


# WebSocket Manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


# Data Models
class SearchQuery(BaseModel):
    query: str
    platforms: List[str] = ["linkedin", "facebook", "x", "instagram"]
    max_comments: int = 100


class SentimentResult(BaseModel):
    text: str
    platform: str
    sentiment: str
    explanation: str
    timestamp: datetime
    model_used: str


# Global task tracker
active_tasks = {}


# =====================================================================
# CORE SCRAPING ORCHESTRATOR
# =====================================================================

async def scrape_platform(platform: str, query: str, task_id: str):
    """Execute scraping for a single platform and update progress"""
    try:
        await manager.broadcast({
            "task_id": task_id,
            "platform": platform,
            "status": "scraping",
            "message": f"Starting scraping on {platform}..."
        })

        # Map platform to scraper function
        scrapers = {
            "linkedin": scrap_linkedin_playwright,
            "facebook": scrap_facebook_playwright,
            "x": scrap_x_playwright,
            "instagram": scrap_instagram_playwright
        }

        if platform in scrapers:
            # Run scraper (your existing async function)
            await scrapers[platform](query)

            # Load scraped data
            file_map = {
                "linkedin": "comentarios_linkedin.csv",
                "facebook": "comentarios_fb.csv",
                "x": "comentarios_x.csv",
                "instagram": "comentarios_ig.csv"
            }

            df = pd.read_csv(file_map[platform])

            # --- DEBUG: DUMP RAW COMMENTS ---
            try:
                debug_comments_file = 'debug_scraped_comments.json'
                current_dump = []
                if os.path.exists(debug_comments_file):
                    with open(debug_comments_file, 'r', encoding='utf-8') as f:
                        current_dump = json.load(f)
                
                # Convertir DF a dict
                for _, r in df.iterrows():
                    current_dump.append({
                        "platform": platform,
                        "text": r.get('texto', ''),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                with open(debug_comments_file, 'w', encoding='utf-8') as f:
                    json.dump(current_dump, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"Error dumping comments: {e}")
            # --------------------------------

            await manager.broadcast({
                "task_id": task_id,
                "platform": platform,
                "status": "nlp_processing",
                "comments_found": len(df),
                "message": f"Processing {len(df)} comments with NLP..."
            })

            # Process with NLP (concurrent)
            results = []
            for idx, row in df.iterrows():
                row_dict = row.to_dict()
                row_dict['origen'] = platform
                nlp_result = process_nlp_row(row_dict)

                if nlp_result:
                    # Convert numpy types before storing
                    nlp_result = convert_numpy_to_python(nlp_result)

                    # Store in MongoDB
                    doc = {
                        **nlp_result,
                        "query": query,
                        "timestamp": datetime.utcnow(),
                        "task_id": task_id
                    }
                    await db.comments.insert_one(doc)
                    results.append(nlp_result)

                    # Send progress update every 10 comments
                    if (idx + 1) % 10 == 0:
                        await manager.broadcast({
                            "task_id": task_id,
                            "platform": platform,
                            "status": "nlp_processing",
                            "progress": f"{idx + 1}/{len(df)}"
                        })

            await manager.broadcast({
                "task_id": task_id,
                "platform": platform,
                "status": "completed",
                "total_analyzed": len(results),
                "message": f"{platform} analysis completed!"
            })

    except Exception as e:
        await manager.broadcast({
            "task_id": task_id,
            "platform": platform,
            "status": "error",
            "error": str(e)
        })


async def run_concurrent_scraping(query: str, platforms: List[str], task_id: str):
    """Execute scraping on multiple platforms concurrently"""
    tasks = [scrape_platform(p, query, task_id) for p in platforms]
    await asyncio.gather(*tasks)

    # Final aggregation
    await manager.broadcast({
        "task_id": task_id,
        "status": "aggregating",
        "message": "Generating global statistics..."
    })

    # Compute global metrics
    results = await db.comments.find({"task_id": task_id}).to_list(length=10000)
    df_results = pd.DataFrame(results)

    stats = {
        "total_comments": int(len(df_results)),
        "sentiment_distribution": {},
        "platform_breakdown": {},
        "avg_processing_time": 0.0
    }

    if not df_results.empty and 'sentimiento' in df_results.columns:
        stats["sentiment_distribution"] = {k: int(v) for k, v in df_results['sentimiento'].value_counts().items()}
    
    if not df_results.empty and 'origen' in df_results.columns:
        stats["platform_breakdown"] = {k: int(v) for k, v in df_results['origen'].value_counts().items()}

    if not df_results.empty and 'tiempo_ejecucion' in df_results.columns:
        stats["avg_processing_time"] = float(df_results['tiempo_ejecucion'].mean())

    # Convert all numpy types to Python types
    stats = convert_numpy_to_python(stats)

    # --- DEBUG: DUMP PROCESSED COMMENTS ---
    try:
        debug_processed_file = 'debug_processed_comments.json'
        processed_dump = df_results.to_dict(orient='records')
        # Ensure conversion of timestamps or non-serializable objects
        for item in processed_dump:
            if 'timestamp' not in item:
                 item['timestamp'] = datetime.utcnow().isoformat()
        
        with open(debug_processed_file, 'w', encoding='utf-8') as f:
            json.dump(processed_dump, f, indent=4, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"Error dumping processed comments: {e}")
    # --------------------------------------

    await db.task_stats.insert_one({
        "task_id": task_id,
        "query": query,
        "stats": stats,
        "timestamp": datetime.utcnow()
    })

    await manager.broadcast({
        "task_id": task_id,
        "status": "finished",
        "stats": stats
    })


# =====================================================================
# API ENDPOINTS
# =====================================================================

@app.post("/api/scrape")
async def start_scraping(search: SearchQuery, background_tasks: BackgroundTasks):
    """Initiate concurrent scraping across platforms"""
    task_id = f"task_{datetime.utcnow().timestamp()}"
    active_tasks[task_id] = {
        "query": search.query,
        "platforms": search.platforms,
        "status": "started",
        "timestamp": datetime.utcnow()
    }

    # Run in background
    background_tasks.add_task(
        run_concurrent_scraping,
        search.query,
        search.platforms,
        task_id
    )

    return {
        "task_id": task_id,
        "message": "Scraping started",
        "platforms": search.platforms
    }


@app.get("/api/results/{task_id}")
async def get_results(task_id: str, platform: Optional[str] = None):
    """Retrieve analyzed comments for a task"""
    query_filter = {"task_id": task_id}
    if platform:
        query_filter["origen"] = platform

    results = await db.comments.find(query_filter).to_list(length=1000)

    # Remove MongoDB _id for JSON serialization
    for r in results:
        r.pop('_id', None)

    return {
        "task_id": task_id,
        "total": len(results),
        "platform": platform,
        "comments": results
    }


@app.get("/api/stats/{task_id}")
async def get_statistics(task_id: str):
    """Get aggregated statistics for a task"""
    stats = await db.task_stats.find_one({"task_id": task_id})

    if not stats:
        return {"error": "Task not found"}

    stats.pop('_id', None)
    return stats


@app.get("/api/storytelling/{task_id}")
async def generate_storytelling(task_id: str):
    """Generate narrative insights from the data"""
    results = await db.comments.find({"task_id": task_id}).to_list(length=10000)
    df = pd.DataFrame(results)

    if df.empty:
        return {"message": "No data available"}

    # Calculate storytelling metrics
    if 'sentimiento' not in df.columns or 'origen' not in df.columns:
         return {"message": "Insufficient data for storytelling"}

    insights = {
        "overall_sentiment": df['sentimiento'].mode()[0] if not df.empty else "Unknown",
        "platform_comparison": {},
        "key_insights": [],
        "sentiment_trends": {}
    }

    # Per-platform analysis
    for platform in df['origen'].unique():
        platform_df = df[df['origen'] == platform]
        if platform_df.empty: continue
        
        sentiment_dist = dict(platform_df['sentimiento'].value_counts())
        dominant = max(sentiment_dist, key=sentiment_dist.get) if sentiment_dist else "None"

        insights["platform_comparison"][platform] = {
            "dominant_sentiment": dominant,
            "total_comments": int(len(platform_df)),
            "distribution": {k: int(v) for k, v in sentiment_dist.items()},
            "avg_length": float(platform_df['texto_original'].apply(lambda x: len(str(x).split())).mean())
        }

    # Generate narrative insights
    positive_pct = (df['sentimiento'] == 'Positivo').sum() / len(df) * 100

    if positive_pct > 60:
        insights["key_insights"].append({
            "type": "positive_dominance",
            "message": f"Strong positive sentiment detected ({positive_pct:.1f}%). The topic resonates well with audiences."
        })
    elif positive_pct < 30:
        insights["key_insights"].append({
            "type": "negative_alert",
            "message": f"Concerning negative sentiment ({100 - positive_pct:.1f}%). Requires attention."
        })

    # Platform engagement comparison
    max_engagement_platform = max(
        insights["platform_comparison"].items(),
        key=lambda x: x[1]["total_comments"]
    )[0]

    insights["key_insights"].append({
        "type": "platform_engagement",
        "message": f"{max_engagement_platform.capitalize()} shows highest engagement with {insights['platform_comparison'][max_engagement_platform]['total_comments']} comments."
    })

    # Convert all numpy types before returning
    insights = convert_numpy_to_python(insights)

    return insights


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Keep connection alive
            await websocket.send_text(json.dumps({"status": "connected"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/")
async def root():
    return {
        "app": "Social Media Sentiment Analyzer",
        "version": "1.0",
        "endpoints": ["/api/scrape", "/api/results/{task_id}", "/api/stats/{task_id}"]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)