from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import asyncio
from datetime import datetime
import uuid
import os
import json
import sqlite3
from contextlib import asynccontextmanager

from services.schema_discovery import SchemaDiscovery
from services.document_processor import DocumentProcessor
from services.query_engine import QueryEngine
from services.cache_manager import CacheManager

schema_discovery = None
document_processor = None
query_engine = None
cache_manager = CacheManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global schema_discovery, document_processor, query_engine
    schema_discovery = SchemaDiscovery()
    document_processor = DocumentProcessor()
    query_engine = QueryEngine()
    
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    yield
    
    if query_engine:
        await query_engine.cleanup()

app = FastAPI(title="NLP Query Engine", version="1.0.0", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ekamapps-assignment.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

class DatabaseConnection(BaseModel):
    connection_string: str
    test_only: bool = False

class QueryRequest(BaseModel):
    query: str
    
class QueryResponse(BaseModel):
    query_id: str
    query_type: str  
    results: Dict[str, Any]
    sources: List[str]
    performance_metrics: Dict[str, Any]
    cached: bool = False

class JobStatus(BaseModel):
    job_id: str
    status: str  
    progress: int
    message: str
    results: Optional[Dict[str, Any]] = None

jobs = {}


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/connect-database")
async def connect_database(connection: DatabaseConnection):
    try:
        if connection.test_only:
            success = await schema_discovery.test_connection(connection.connection_string)
            return {"success": success, "message": "Connection successful" if success else "Connection failed"}
        
        schema = await schema_discovery.analyze_database(connection.connection_string)
        
        await query_engine.initialize(connection.connection_string, schema)
        
        return {
            "success": True,
            "schema": schema,
            "message": f"Connected successfully. Discovered {len(schema.get('tables', {}))} tables"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/upload-documents")
async def upload_documents(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    job_id = str(uuid.uuid4())
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        progress=0,
        message="Upload initiated"
    )
    
    file_paths = []
    for file in files:
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        file_paths.append(file_path)
    
    background_tasks.add_task(process_documents_background, job_id, file_paths)
    
    return {"job_id": job_id, "message": f"Started processing {len(files)} files"}

async def process_documents_background(job_id: str, file_paths: List[str]):
    try:
        jobs[job_id].status = "processing"
        jobs[job_id].message = "Processing documents..."
        
        total_files = len(file_paths)
        processed_count = 0
        
        for file_path in file_paths:
            await document_processor.process_document(file_path)
            processed_count += 1
            
            progress = int((processed_count / total_files) * 100)
            jobs[job_id].progress = progress
            jobs[job_id].message = f"Processed {processed_count}/{total_files} files"
        
        jobs[job_id].status = "completed"
        jobs[job_id].progress = 100
        jobs[job_id].message = f"Successfully processed {total_files} documents"
        jobs[job_id].results = {
            "processed_files": total_files,
            "total_chunks": await document_processor.get_total_chunks()
        }
        
    except Exception as e:
        jobs[job_id].status = "failed"
        jobs[job_id].message = str(e)

@app.get("/api/ingestion-status/{job_id}")
async def get_ingestion_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.post("/api/query")
async def process_query(query_request: QueryRequest):
    try:
        cached_result = cache_manager.get(query_request.query)
        if cached_result:
            cached_result['cached'] = True
            return QueryResponse(**cached_result)
        
        start_time = datetime.now()
        
        result = await query_engine.process_query(query_request.query)
        
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        
        query_response = QueryResponse(
            query_id=str(uuid.uuid4()),
            query_type=result["query_type"],
            results=result["results"],
            sources=result["sources"],
            performance_metrics={
                "response_time": response_time,
                "cache_hit": False
            },
            cached=False
        )
        
        cache_manager.set(query_request.query, query_response.dict())
        
        return query_response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/query/history")
async def get_query_history():
    return {"queries": cache_manager.get_recent_queries()}

@app.get("/api/schema")
async def get_schema():
    if query_engine.schema:
        return {"schema": query_engine.schema}
    else:
        return {"schema": {}, "message": "No database connected"}

@app.get("/api/metrics")
async def get_metrics():
    return {
        "cache_stats": cache_manager.get_stats(),
        "active_connections": 1 if getattr(query_engine, "engine", None) else 0,
        "total_documents": await document_processor.get_total_documents() if document_processor else 0,
        "total_chunks": await document_processor.get_total_chunks() if document_processor else 0
    }

# if __name__ == "__main__":
#     port = int(os.getenv("PORT", 8000)) 
#     uvicorn.run("main:app", host="0.0.0.0", port=port)