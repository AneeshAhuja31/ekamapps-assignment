from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine, inspect, MetaData, text
from sqlalchemy.exc import SQLAlchemyError
import logging

logger = logging.getLogger(__name__)

class SchemaDiscovery:
    def __init__(self):
        self.engine = None
        self.metadata = None
        
    async def test_connection(self, connection_string: str) -> bool:
        """Test database connection without full schema discovery"""
        try:
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            engine.dispose()
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    async def analyze_database(self, connection_string: str) -> Dict[str, Any]:
        """
        Analyze database and discover schema dynamically.
        Works with any reasonable employee database structure.
        """
        try:
            self.engine = create_engine(connection_string)
            inspector = inspect(self.engine)
            
            table_names = inspector.get_table_names()
            
            schema = {
                "connection_string": connection_string,
                "tables": {},
                "relationships": [],
                "summary": {
                    "table_count": len(table_names),
                    "total_columns": 0
                }
            }
            
            for table_name in table_names:
                table_info = await self._analyze_table(inspector, table_name)
                schema["tables"][table_name] = table_info
                schema["summary"]["total_columns"] += len(table_info["columns"])
            
            relationships = await self._discover_relationships(inspector, table_names)
            schema["relationships"] = relationships
            schema["summary"]["relationship_count"] = len(relationships)
            
            schema["table_categories"] = await self._categorize_tables(schema["tables"])
            
            return schema
            
        except Exception as e:
            logger.error(f"Schema analysis failed: {e}")
            raise Exception(f"Failed to analyze database schema: {str(e)}")
    
    async def _analyze_table(self, inspector, table_name: str) -> Dict[str, Any]:
        """Analyze individual table structure and sample data"""
        try:
            columns = inspector.get_columns(table_name)
            
            pk_constraint = inspector.get_pk_constraint(table_name)
            primary_keys = pk_constraint.get('constrained_columns', []) if pk_constraint else []
            
            foreign_keys = inspector.get_foreign_keys(table_name)
            
            sample_data = await self._get_sample_data(table_name, limit=3)
            
            table_purpose = self._infer_table_purpose(table_name, [col['name'] for col in columns])
            
            return {
                "name": table_name,
                "purpose": table_purpose,
                "columns": [
                    {
                        "name": col["name"],
                        "type": str(col["type"]),
                        "nullable": col["nullable"],
                        "is_primary_key": col["name"] in primary_keys,
                        "semantic_type": self._infer_column_semantic_type(col["name"])
                    }
                    for col in columns
                ],
                "primary_keys": primary_keys,
                "foreign_keys": foreign_keys,
                "sample_data": sample_data,
                "row_count": await self._get_row_count(table_name)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing table {table_name}: {e}")
            return {
                "name": table_name,
                "purpose": "unknown",
                "columns": [],
                "primary_keys": [],
                "foreign_keys": [],
                "sample_data": [],
                "row_count": 0
            }
    
    async def _get_sample_data(self, table_name: str, limit: int = 3) -> List[Dict]:
        """Get sample data from table"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT {limit}"))
                columns = result.keys()
                rows = result.fetchall()
                
                return [
                    {col: str(row[i]) if row[i] is not None else None for i, col in enumerate(columns)}
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Error getting sample data for {table_name}: {e}")
            return []
    
    async def _get_row_count(self, table_name: str) -> int:
        """Get total row count for table"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                return result.scalar()
        except Exception as e:
            logger.error(f"Error getting row count for {table_name}: {e}")
            return 0
    
    async def _discover_relationships(self, inspector, table_names: List[str]) -> List[Dict]:
        """Discover relationships between tables"""
        relationships = []
        
        for table_name in table_names:
            try:
                foreign_keys = inspector.get_foreign_keys(table_name)
                
                for fk in foreign_keys:
                    relationships.append({
                        "from_table": table_name,
                        "from_column": fk["constrained_columns"][0] if fk["constrained_columns"] else "",
                        "to_table": fk["referred_table"],
                        "to_column": fk["referred_columns"][0] if fk["referred_columns"] else "",
                        "type": "foreign_key"
                    })
            except Exception as e:
                logger.error(f"Error discovering relationships for {table_name}: {e}")
        
        return relationships
    
    def _infer_table_purpose(self, table_name: str, columns: List[str]) -> str:
        """Infer the purpose of a table based on its name and columns"""
        table_name_lower = table_name.lower()
        column_names_lower = [col.lower() for col in columns]
        
        if any(keyword in table_name_lower for keyword in ['employee', 'staff', 'personnel', 'worker', 'emp']):
            return "employees"
        
        if any(keyword in table_name_lower for keyword in ['department', 'dept', 'division', 'team']):
            return "departments"
        
        if any(keyword in table_name_lower for keyword in ['document', 'file', 'attachment', 'doc']):
            return "documents"
        
        if any(keyword in table_name_lower for keyword in ['salary', 'compensation', 'pay', 'wage']):
            return "compensation"
        
        if any(keyword in table_name_lower for keyword in ['review', 'performance', 'evaluation']):
            return "reviews"
        
        
        if any(col in column_names_lower for col in ['employee_id', 'emp_id', 'staff_id']):
            if any(col in column_names_lower for col in ['salary', 'wage', 'compensation', 'pay']):
                return "compensation"
            else:
                return "employees"
        
        return "general"
    
    def _infer_column_semantic_type(self, column_name: str) -> str:
        """Infer the semantic meaning of a column"""
        column_lower = column_name.lower()
        
        if column_lower.endswith('_id') or column_lower == 'id':
            return "identifier"
        
        if any(keyword in column_lower for keyword in ['name', 'full_name', 'first_name', 'last_name']):
            return "name"
        
        
        if 'email' in column_lower:
            return "email"
        
        if any(keyword in column_lower for keyword in ['date', 'created', 'updated', 'hired', 'start', 'end']):
            return "date"
        
        if any(keyword in column_lower for keyword in ['salary', 'wage', 'pay', 'compensation', 'amount']):
            return "money"
        
        if any(keyword in column_lower for keyword in ['status', 'active', 'enabled']):
            return "status"
        
        if any(keyword in column_lower for keyword in ['department', 'dept', 'division']):
            return "department"
        
        if any(keyword in column_lower for keyword in ['position', 'title', 'role', 'job']):
            return "position"
        
        return "general"
    
    async def _categorize_tables(self, tables: Dict[str, Any]) -> Dict[str, List[str]]:
        """Categorize tables by their inferred purpose"""
        categories = {
            "employees": [],
            "departments": [],
            "documents": [],
            "compensation": [],
            "reviews": [],
            "general": []
        }
        
        for table_name, table_info in tables.items():
            purpose = table_info.get("purpose", "general")
            if purpose in categories:
                categories[purpose].append(table_name)
            else:
                categories["general"].append(table_name)
        
        return categories
    
    def map_natural_language_to_schema(self, query: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map natural language terms in query to actual database schema.
        Example: "salary" in query → "compensation" in database
        """
        mapping = {
            "potential_tables": [],
            "potential_columns": [],
            "query_intent": self._classify_query_intent(query)
        }
        
        query_lower = query.lower()
        
        term_mappings = {
            "employee": ["employees", "staff", "personnel", "worker", "emp"],
            "department": ["departments", "dept", "division", "team"],
            "salary": ["salary", "compensation", "pay", "wage"],
            "name": ["name", "full_name", "first_name", "last_name"],
            "hire": ["hired", "hire_date", "start_date", "join_date"]
        }
        
        for table_name, table_info in schema.get("tables", {}).items():
            table_relevance = self._calculate_table_relevance(query_lower, table_name, table_info)
            if table_relevance > 0:
                mapping["potential_tables"].append({
                    "table": table_name,
                    "relevance": table_relevance,
                    "purpose": table_info.get("purpose", "unknown")
                })
        
        mapping["potential_tables"] = sorted(
            mapping["potential_tables"], 
            key=lambda x: x["relevance"], 
            reverse=True
        )
        
        return mapping
    
    def _calculate_table_relevance(self, query: str, table_name: str, table_info: Dict) -> float:
        """Calculate how relevant a table is to the query"""
        relevance = 0.0
        
        if table_name.lower() in query:
            relevance += 1.0
        
        purpose = table_info.get("purpose", "")
        if purpose in query:
            relevance += 0.8
        
        for col in table_info.get("columns", []):
            col_name = col["name"].lower()
            if col_name in query:
                relevance += 0.3
            
            semantic_type = col.get("semantic_type", "")
            if semantic_type == "money" and any(term in query for term in ["salary", "pay", "wage"]):
                relevance += 0.5
            elif semantic_type == "name" and "name" in query:
                relevance += 0.5
        
        return relevance
    
    def _classify_query_intent(self, query: str) -> str:
        """Classify the intent of the natural language query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["count", "sum", "average", "avg", "total", "how many"]):
            return "aggregation"
        
        if any(word in query_lower for word in ["list", "show", "find", "get", "who", "which"]):
            return "list"
        
        if any(word in query_lower for word in ["highest", "lowest", "top", "bottom", "best", "worst"]):
            return "comparison"
        
        if any(word in query_lower for word in ["where", "with", "having", "in", "from"]):
            return "filter"
        
        return "general"