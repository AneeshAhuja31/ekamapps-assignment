import asyncio
import re
import json
import pickle
from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import logging
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from services.schema_discovery import SchemaDiscovery
from services.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class QueryEngine:
    def __init__(self):
        self.connection_string = None
        self.engine = None
        self.schema = None
        self.schema_discovery = SchemaDiscovery()
        self.document_processor = DocumentProcessor()
        
        self.llm_model = None
        self.embeddings = None
        self.vector_store = None
        self.schema_vector_store = None
        
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
            api_key = os.getenv("GOOGLE_API_KEY", "")
            
            if api_key:
                self.llm_model = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=api_key,
                    temperature=0.1
                )
                
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Initialized Hugging Face embeddings successfully")
            
            if api_key:
                logger.info("Initialized Gemini LLM successfully")
            else:
                logger.warning("Google API key not found. LLM features disabled but embeddings available.")
                
        except Exception as e:
            logger.warning(f"Could not initialize embeddings: {e}. Using fallback query processing.")
            self.llm_model = None
            self.embeddings = None
    

    async def initialize(self, connection_string: str, schema: Dict[str, Any]):
        """Initialize query engine with database connection and schema"""
        try:
            self.connection_string = connection_string
            self.schema = schema
            self.engine = create_engine(connection_string)
            
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            if self.embeddings:
                await self._initialize_schema_vectorstore()
                await self._initialize_document_vectorstore()
            
            logger.info("Query engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize query engine: {e}")
            raise

    
    async def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process natural language query with intelligent routing
        """
        try:
            if self.schema_vector_store:
                query_classification = await self._enhanced_classify_query(user_query)
            else:
                query_classification = await self._classify_query(user_query)
            
            if query_classification["type"] == "sql":
                return await self._process_sql_query(user_query, query_classification)
            elif query_classification["type"] == "document":
                return await self._process_document_query(user_query, query_classification)
            elif query_classification["type"] == "hybrid":
                return await self._process_hybrid_query(user_query, query_classification)
            else:
                return await self._process_hybrid_query(user_query, query_classification)
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "query_type": "error",
                "results": {"error": str(e)},
                "sources": []
            }
    
    async def _classify_query(self, query: str) -> Dict[str, Any]:
        """Classify the type of query and extract relevant information"""
        query_lower = query.lower()
        
        db_indicators = [
            "count", "sum", "average", "avg", "total", "how many",
            "salary", "department", "employee", "staff", "hired",
            "reports to", "manager", "position", "compensation"
        ]
        
        doc_indicators = [
            "resume", "cv", "document", "file", "review", "performance",
            "skills", "experience", "qualifications", "background"
        ]
        
        aggregation_keywords = ["count", "sum", "average", "avg", "total", "max", "min"]
        
        db_score = sum(1 for indicator in db_indicators if indicator in query_lower)
        doc_score = sum(1 for indicator in doc_indicators if indicator in query_lower)
        
        if db_score > doc_score and db_score > 0:
            query_type = "sql"
        elif doc_score > db_score and doc_score > 0:
            query_type = "document"
        elif db_score > 0 and doc_score > 0:
            query_type = "hybrid"
        else:
            query_type = "hybrid"
        
        schema_mapping = self.schema_discovery.map_natural_language_to_schema(query, self.schema or {})
        
        return {
            "type": query_type,
            "original_query": query,
            "db_score": db_score,
            "doc_score": doc_score,
            "has_aggregation": any(kw in query_lower for kw in aggregation_keywords),
            "schema_mapping": schema_mapping
        }
    
    async def _process_sql_query(self, query: str, classification: Dict) -> Dict[str, Any]:
        """Process queries that require database operations"""
        try:
            sql_query = await self._generate_sql(query, classification)
            
            if not sql_query:
                return {
                    "query_type": "sql",
                    "results": {"error": "Could not generate SQL query"},
                    "sources": []
                }
            
            results = await self._execute_sql_query(sql_query)
            
            return {
                "query_type": "sql",
                "results": {
                    "sql_query": sql_query,
                    "data": results["data"],
                    "columns": results["columns"],
                    "row_count": len(results["data"])
                },
                "sources": [f"Database query: {sql_query}"]
            }
            
        except Exception as e:
            logger.error(f"Error processing SQL query: {e}")
            return {
                "query_type": "sql",
                "results": {"error": str(e)},
                "sources": []
            }
    
    async def _process_document_query(self, query: str, classification: Dict) -> Dict[str, Any]:
        """Process queries that search through documents using vector embeddings"""
        try:
            search_results = await self._process_documents_with_vectors(query, limit=10)
            
            formatted_results = []
            sources = []
            
            for result in search_results:
                formatted_results.append({
                    "filename": result["filename"],
                    "file_type": result["file_type"],
                    "content": result["content"][:500] + "..." if len(result["content"]) > 500 else result["content"],
                    "similarity": result["similarity"],
                    "chunk_index": result["chunk_index"]
                })
                sources.append(f"{result['filename']} (chunk {result['chunk_index']})")
            
            return {
                "query_type": "document",
                "results": {
                    "documents": formatted_results,
                    "total_matches": len(search_results)
                },
                "sources": sources[:10]  # Limit sources to top 10
            }
            
        except Exception as e:
            logger.error(f"Error processing document query: {e}")
            return {
                "query_type": "document",
                "results": {"error": str(e)},
                "sources": []
            }
    
    async def _process_hybrid_query(self, query: str, classification: Dict) -> Dict[str, Any]:
        """Process queries that require both database and document search"""
        try:
            sql_task = asyncio.create_task(self._process_sql_query(query, classification))
            doc_task = asyncio.create_task(self._process_document_query(query, classification))
            
            sql_results, doc_results = await asyncio.gather(sql_task, doc_task, return_exceptions=True)
            
            combined_results = {
                "database_results": sql_results.get("results", {}) if not isinstance(sql_results, Exception) else {"error": str(sql_results)},
                "document_results": doc_results.get("results", {}) if not isinstance(doc_results, Exception) else {"error": str(doc_results)}
            }
            
            combined_sources = []
            if not isinstance(sql_results, Exception) and sql_results.get("sources"):
                combined_sources.extend(sql_results["sources"])
            if not isinstance(doc_results, Exception) and doc_results.get("sources"):
                combined_sources.extend(doc_results["sources"])
            
            return {
                "query_type": "hybrid",
                "results": combined_results,
                "sources": combined_sources
            }
            
        except Exception as e:
            logger.error(f"Error processing hybrid query: {e}")
            return {
                "query_type": "hybrid",
                "results": {"error": str(e)},
                "sources": []
            }
    
    async def _generate_sql(self, query: str, classification: Dict) -> Optional[str]:
        """Generate SQL query from natural language"""
        try:
            if not self.schema or not self.schema.get("tables"):
                return None
            
            if self.llm_model:
                return await self._generate_sql_with_llm(query, classification)
            else:
                return await self._generate_sql_with_patterns(query, classification)
                
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return None
    
    async def _generate_sql_with_llm(self, query: str, classification: Dict) -> Optional[str]:
        """Generate SQL using Gemini LLM with Langchain"""
        try:
            if not self.llm_model:
                return await self._generate_sql_with_patterns(query, classification)
            
            schema_info = self._format_schema_for_prompt()
            
            prompt = f"""
            Given the following database schema:
            {schema_info}
            
            Convert this natural language query to SQL:
            "{query}"
            
            Rules:
            1. Return only the SQL query, no explanation
            2. Use proper table and column names from the schema
            3. Include appropriate WHERE clauses, JOINs, and aggregations
            4. Limit results to 100 rows unless specifically asked for more
            5. Use proper SQL syntax
            
            SQL Query:
            """
            
            response = await self.llm_model.ainvoke(prompt)
            sql_query = response.content.strip()
            
            sql_query = re.sub(r'```sql\n?', '', sql_query)
            sql_query = re.sub(r'```\n?', '', sql_query)
            sql_query = sql_query.strip()
            
            # Basic validation
            if not sql_query.upper().startswith(('SELECT', 'WITH')):
                return None
            
            return sql_query
            
        except Exception as e:
            logger.error(f"Error generating SQL with LLM: {e}")
            return await self._generate_sql_with_patterns(query, classification)
    
    async def _generate_sql_with_patterns(self, query: str, classification: Dict) -> Optional[str]:
        """Generate SQL using pattern matching (fallback method)"""
        try:
            query_lower = query.lower()
            schema_mapping = classification.get("schema_mapping", {})
            potential_tables = schema_mapping.get("potential_tables", [])
            
            if not potential_tables:
                return None
            
            # Use the most relevant table
            main_table = potential_tables[0]["table"]
            
            if "count" in query_lower or "how many" in query_lower:
                return f"SELECT COUNT(*) as count FROM {main_table} LIMIT 100"
            
            elif "average" in query_lower or "avg" in query_lower:
                if "by department" in query_lower or "department" in query_lower:
                    dept_table = self._find_department_table()
                    salary_col = self._find_numeric_column(main_table)
                    dept_col = self._find_department_column(main_table)
                    
                    if dept_table and salary_col and dept_col:
                        return f"""
                            SELECT d.dept_name, AVG(e.{salary_col}) as avg_salary 
                            FROM {main_table} e 
                            JOIN {dept_table} d ON e.{dept_col} = d.dept_id 
                            GROUP BY d.dept_name 
                            ORDER BY avg_salary DESC
                            LIMIT 100
                        """
                    elif salary_col:
                        return f"SELECT AVG({salary_col}) as average FROM {main_table}"
                else:
                    # Try to find a numeric column
                    numeric_col = self._find_numeric_column(main_table)
                    if numeric_col:
                        return f"SELECT AVG({numeric_col}) as average FROM {main_table}"
                    else:
                        return f"SELECT COUNT(*) as count FROM {main_table} LIMIT 100"
            
            elif "list" in query_lower or "show" in query_lower or "all" in query_lower:
                return f"SELECT * FROM {main_table} LIMIT 100"
            
            else:
                return f"SELECT * FROM {main_table} LIMIT 100"
                
        except Exception as e:
            logger.error(f"Error generating SQL with patterns: {e}")
            return None

    def _find_department_table(self) -> Optional[str]:
        """Find the department table"""
        if not self.schema or not self.schema.get("tables"):
            return None
        
        for table_name, table_info in self.schema["tables"].items():
            purpose = table_info.get("purpose", "").lower()
            if "department" in purpose or "dept" in table_name.lower():
                return table_name
        
        return None

    def _find_department_column(self, table_name: str) -> Optional[str]:
        """Find the department reference column in a table"""
        if not self.schema or table_name not in self.schema.get("tables", {}):
            return None
        
        table_info = self.schema["tables"][table_name]
        
        for column in table_info.get("columns", []):
            col_name = column["name"].lower()
            if any(dept_term in col_name for dept_term in ["dept_id", "department_id", "dept", "department"]):
                return column["name"]
        
        return None
    
    def _format_schema_for_prompt(self) -> str:
        """Format schema information for LLM prompt"""
        if not self.schema or not self.schema.get("tables"):
            return "No schema available"
        
        schema_text = "Database Schema:\n"
        
        for table_name, table_info in self.schema["tables"].items():
            schema_text += f"\nTable: {table_name}\n"
            schema_text += f"Purpose: {table_info.get('purpose', 'unknown')}\n"
            schema_text += "Columns:\n"
            
            for column in table_info.get("columns", []):
                col_info = f"  - {column['name']} ({column['type']})"
                if column.get("is_primary_key"):
                    col_info += " [PRIMARY KEY]"
                if column.get("semantic_type"):
                    col_info += f" [{column['semantic_type']}]"
                schema_text += col_info + "\n"
        
        relationships = self.schema.get("relationships", [])
        if relationships:
            schema_text += "\nRelationships:\n"
            for rel in relationships:
                schema_text += f"  {rel['from_table']}.{rel['from_column']} -> {rel['to_table']}.{rel['to_column']}\n"
        
        return schema_text
    
    def _find_numeric_column(self, table_name: str) -> Optional[str]:
        """Find a numeric column in the given table"""
        if not self.schema or table_name not in self.schema.get("tables", {}):
            return None
        
        table_info = self.schema["tables"][table_name]
        
        for column in table_info.get("columns", []):
            semantic_type = column.get("semantic_type", "")
            if semantic_type == "money":
                return column["name"]
        
        for column in table_info.get("columns", []):
            col_type = str(column.get("type", "")).lower()
            if any(num_type in col_type for num_type in ["int", "float", "decimal", "numeric"]):
                return column["name"]
        
        return None
    
    async def _execute_sql_query(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query safely"""
        try:
            if not self._is_safe_query(sql_query):
                raise ValueError("Potentially unsafe query detected")
            
            with self.engine.connect() as conn:
                result = conn.execute(text(sql_query))
                columns = list(result.keys())
                rows = result.fetchall()
                
                data = [
                    {col: (str(row[i]) if row[i] is not None else None) for i, col in enumerate(columns)}
                    for row in rows
                ]
                
                return {
                    "columns": columns,
                    "data": data
                }
                
        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            raise
    
    def _is_safe_query(self, sql_query: str) -> bool:
        """Basic SQL injection protection"""
        sql_lower = sql_query.lower().strip()
        
        if not sql_lower.startswith(('select', 'with')):
            return False
        
        dangerous_keywords = [
            'drop', 'delete', 'update', 'insert', 'create', 'alter',
            'truncate', 'exec', 'execute', 'sp_', 'xp_'
        ]
        
        for keyword in dangerous_keywords:
            if keyword in sql_lower:
                return False
        
        return True
    
    async def _initialize_schema_vectorstore(self):
        """Initialize FAISS vector store for schema information"""
        try:
            if not self.embeddings or not self.schema:
                return
            
            # Create documents from schema information
            schema_docs = []
            
            for table_name, table_info in self.schema.get("tables", {}).items():
                table_doc_content = f"""
                Table: {table_name}
                Purpose: {table_info.get('purpose', 'Unknown')}
                Columns: {', '.join([col['name'] for col in table_info.get('columns', [])])}
                """
                
                schema_docs.append(Document(
                    page_content=table_doc_content.strip(),
                    metadata={
                        "table_name": table_name,
                        "type": "table_schema",
                        "purpose": table_info.get('purpose', 'Unknown')
                    }
                ))
                
                for column in table_info.get('columns', []):
                    column_doc_content = f"""
                    Column: {column['name']} in table {table_name}
                    Type: {column['type']}
                    Semantic Type: {column.get('semantic_type', 'Unknown')}
                    Description: {column.get('description', 'No description')}
                    """
                    
                    schema_docs.append(Document(
                        page_content=column_doc_content.strip(),
                        metadata={
                            "table_name": table_name,
                            "column_name": column['name'],
                            "type": "column_schema",
                            "data_type": column['type'],
                            "semantic_type": column.get('semantic_type', 'Unknown')
                        }
                    ))
            
            if schema_docs:
                self.schema_vector_store = await FAISS.afrom_documents(
                    documents=schema_docs,
                    embedding=self.embeddings
                )
                logger.info(f"Schema vector store initialized with {len(schema_docs)} documents")
            
        except Exception as e:
            logger.error(f"Error initializing schema vector store: {e}")
    
    async def _enhanced_classify_query(self, query: str) -> Dict[str, Any]:
        """Enhanced query classification using vector similarity"""
        try:
            # Get base classification
            base_classification = await self._classify_query(query)
            
            # If we have schema vector store, enhance with semantic search
            if self.schema_vector_store:
                relevant_docs = await self.schema_vector_store.asimilarity_search(
                    query, k=5
                )
                
                # Extract relevant tables and columns
                relevant_tables = set()
                relevant_columns = set()
                
                for doc in relevant_docs:
                    if doc.metadata.get("type") == "table_schema":
                        relevant_tables.add(doc.metadata.get("table_name"))
                    elif doc.metadata.get("type") == "column_schema":
                        relevant_tables.add(doc.metadata.get("table_name"))
                        relevant_columns.add(f"{doc.metadata.get('table_name')}.{doc.metadata.get('column_name')}")
                
                base_classification["semantic_tables"] = list(relevant_tables)
                base_classification["semantic_columns"] = list(relevant_columns)
                base_classification["vector_enhanced"] = True
            
            return base_classification
            
        except Exception as e:
            logger.error(f"Error in enhanced query classification: {e}")
            return await self._classify_query(query)
    

    async def _process_documents_with_vectors(self, query: str, limit: int = 10) -> List[Dict]:
        """Process document search using vector embeddings"""
        try:
            return await self.document_processor.search_documents_vector(query, limit=limit)
            
        except Exception as e:
            logger.error(f"Error processing documents with vectors: {e}")
            return await self.document_processor.search_documents(query, limit=limit)

    async def _initialize_document_vectorstore(self):
        """Initialize document vector store with existing documents"""
        try:
            if not self.embeddings:
                logger.warning("Embeddings not available, skipping document vector store initialization")
                return
            
            documents = await self.document_processor.get_all_documents()
            
            if not documents:
                logger.info("No documents found, creating empty vector store with placeholder")
                placeholder_text = "This is a placeholder document for the vector store initialization."
                self.vector_store = FAISS.from_texts(
                    [placeholder_text],
                    self.embeddings,
                    metadatas=[{"type": "placeholder", "filename": "placeholder"}]
                )
                return
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            
            all_texts = []
            all_metadatas = []
            
            for doc in documents:
                content = doc.get('content', '').strip()
                if not content or len(content) < 10:
                    logger.warning(f"Skipping document {doc.get('filename', 'unknown')} - insufficient content")
                    continue
                
                chunks = text_splitter.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    chunk_text = chunk.strip()
                    if len(chunk_text) < 5:  # Skip very short chunks
                        continue
                        
                    all_texts.append(chunk_text)
                    all_metadatas.append({
                        'doc_id': doc['id'],
                        'filename': doc['filename'],
                        'file_type': doc['file_type'],
                        'chunk_index': i,
                        'upload_date': doc['upload_date']
                    })
            
            if all_texts:
                self.vector_store = FAISS.from_texts(
                    all_texts,
                    self.embeddings,
                    metadatas=all_metadatas
                )
                
                logger.info(f"Document vector store initialized with {len(all_texts)} chunks from {len(documents)} documents")
            else:
                logger.info("No valid document content found, creating placeholder vector store")
                placeholder_text = "This is a placeholder document for the vector store initialization."
                self.vector_store = FAISS.from_texts(
                    [placeholder_text],
                    self.embeddings,
                    metadatas=[{"type": "placeholder", "filename": "placeholder"}]
                )
                
        except Exception as e:
            logger.error(f"Error initializing document vector store: {e}")
            
            try:
                if self.embeddings:
                    placeholder_text = "This is a fallback placeholder document for error recovery."
                    self.vector_store = FAISS.from_texts(
                        [placeholder_text],
                        self.embeddings,
                        metadatas=[{"type": "fallback_placeholder", "filename": "fallback"}]
                    )
                    logger.info("Created fallback vector store with placeholder")
            except Exception as fallback_error:
                logger.error(f"Failed to create fallback vector store: {fallback_error}")
                self.vector_store = None

    async def save_vector_store(self, path: str = "vector_store"):
        """Save vector store to disk"""
        try:
            if self.vector_store:
                self.vector_store.save_local(path)
                logger.info(f"Vector store saved to {path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
    
    async def load_vector_store(self, path: str = "vector_store"):
        """Load vector store from disk"""
        try:
            if self.embeddings and os.path.exists(path):
                self.vector_store = FAISS.load_local(
                    path, 
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Vector store loaded from {path}")
                return True
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
        return False
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.engine:
            self.engine.dispose()