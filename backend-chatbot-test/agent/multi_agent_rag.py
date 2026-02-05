"""
Multi-Agent RAG system for oceanographic data analysis using LangGraph.
"""

from typing import Dict, List, Any, Optional, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
import json
import logging
from datetime import datetime, timedelta
import numpy as np
import re
import asyncio

from tools import ArgoToolFactory
from .config import GROQ_API_KEY, GROQ_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiAgentState(TypedDict):
    """State for the multi-agent RAG system"""
    messages: List[Any]
    query: str
    intent: Optional[Dict[str, Any]]
    measurement_results: Optional[Dict[str, Any]]
    metadata_results: Optional[Dict[str, Any]]
    semantic_results: Optional[Dict[str, Any]]
    final_response: Optional[str]
    error: Optional[str]

class MeasurementAgent:
    """Specialized agent for oceanographic measurements"""
    
    def __init__(self, tools: ArgoToolFactory):
        self.tools = tools
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=0.1
        )
    
    def process(self, query: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Process measurement-related queries"""
        try:
            logger.info("MeasurementAgent processing query")
            
            # Check if this is a query for all float IDs or requires custom SQL
            if "all float" in query.lower() or "float id" in query.lower() or "platform number" in query.lower():
                return self._handle_float_id_query(query)
            
            if intent.get("float_id"):
                measurements = self.tools.cockroach.get_measurements_by_float(
                    platform_number=intent["float_id"],
                    limit=1000
                )
            elif intent.get("spatial_filter"):
                sf = intent["spatial_filter"]
                measurements = self.tools.cockroach.get_measurements_by_region(
                    min_lat=sf["min_lat"],
                    max_lat=sf["max_lat"],
                    min_lon=sf["min_lon"],
                    max_lon=sf["max_lon"],
                    limit=1000
                )
            else:
                return {"error": "No valid parameters for measurement query"}
            
            if measurements:
                stats = {
                    "temp_stats": self._calculate_stats([m.temp_adjusted for m in measurements]),
                    "psal_stats": self._calculate_stats([m.psal_adjusted for m in measurements]),
                    "pres_stats": self._calculate_stats([m.pres_adjusted for m in measurements])
                }
                
                return {
                    "agent": "MeasurementAgent",
                    "count": len(measurements),
                    "statistics": stats,
                    "time_range": f"{measurements[0].time} to {measurements[-1].time}",
                    "spatial_coverage": self._get_spatial_coverage(measurements),
                    "summary": f"Found {len(measurements)} measurements with comprehensive statistics"
                }
            else:
                return {
                    "agent": "MeasurementAgent",
                    "count": 0,
                    "summary": "No measurements found for the specified criteria"
                }
                
        except Exception as e:
            logger.error(f"MeasurementAgent error: {e}")
            return {"agent": "MeasurementAgent", "error": str(e)}
    
    def _handle_float_id_query(self, query: str) -> Dict[str, Any]:
        """Handle queries asking for float IDs using custom SQL"""
        try:
            logger.info(f"Processing float ID query: {query}")
            
            # Use LLM to generate appropriate SQL query
            sql_prompt = f"""
            Generate a SQL query for CockroachDB to answer this question: "{query}"
            No Explanations
            """
            
            messages = [
                SystemMessage(content="""
                        You are an expert SQL agent specialized in **CockroachDB**, tasked with generating **efficient, high-performance queries** for a large oceanographic dataset (`argo_measurements`) containing 13.5M+ records.

                        ---

                        ## Table Context

                        The table `argo_measurements` has the following columns:

                        - `platform_number` (STRING, not null) → unique float ID  
                        - `time` (TIMESTAMPTZ, not null) → timestamp of measurement  
                        - `latitude` (FLOAT8, nullable) → latitude of the float  
                        - `longitude` (FLOAT8, nullable) → longitude of the float  
                        - `pres_adjusted` (FLOAT8, nullable) → pressure, used as a proxy for depth  
                        - `temp_adjusted` (FLOAT8, nullable) → temperature  
                        - `psal_adjusted` (FLOAT8, nullable) → salinity  
                        - `rowid` (INT8, primary key) → unique row identifier  

                        **Indexes available**:

                        - `argo_measurements_pkey` (rowid, platform_number, time, latitude, longitude, pres_adjusted, temp_adjusted, psal_adjusted)  
                        - `idx_platform_number` (platform_number ASC, rowid ASC)  
                        - `idx_platform_time` (platform_number ASC, time DESC, rowid ASC)  

                        ---

                        ## Key Domain Knowledge

                        1. **Latest month queries**:
                        - “Latest month” = most recent month present in the table (not necessarily current calendar month).  
                        - Use **indexes** to filter by month efficiently.  
                        - Prefer `ROW_NUMBER()` over `DISTINCT` or `GROUP BY` for per-float latest measurements.

                        2. **Cycles**:
                        - A “cycle” is a **cluster of measurements for a float where latitude/longitude remains roughly constant for 7–10 days**.  
                        - Identify cycles using **LAG() + window functions** with a spatial threshold (e.g., ±0.05°).  
                        - Optional: include `pres_adjusted` for depth variation.

                        3. **Depth**:
                        - No explicit depth column; **pressure (`pres_adjusted`) is a proxy for depth**.  
                        - 1 dbar ≈ 1 meter; use thresholds (e.g., ±10 dbar) for depth clustering.

                        4. **Performance principles**:
                        - Always **filter by indexed columns first** (`platform_number`, `time`).  
                        - Use **window functions** (`ROW_NUMBER()`, `LAG()`, `SUM() OVER`) instead of `DISTINCT`/`GROUP BY` when possible.  
                        - For large tables, consider **covering indexes** or **materialized views**.  
                        - Avoid `DATE_TRUNC` on full table scans; filter first by indexed ranges.

                        5. **Common queries to handle**:
                        - List of **floats reporting in the latest month**  
                        - **Latest measurement per float**  
                        - **Last cycle of a float** (7–10 day cluster based on location ± optional depth)  
                        - **Aggregates** (avg temperature, salinity, etc.) in a region/time range  
                        - **Pressure-based depth filtering**  

                        ---

                        ## Example Query Templates

                        **1. Latest month floats**
                        ```sql
                        WITH latest_month AS (
                            SELECT date_trunc('month', MAX(time)) AS month_start
                            FROM argo_measurements
                        )
                        SELECT platform_number
                        FROM argo_measurements
                        WHERE time >= (SELECT month_start FROM latest_month)
                        AND time < (SELECT month_start + INTERVAL '1 month' FROM latest_month)
                        GROUP BY platform_number;

                        Generate only the SQL query, no explanation.
                """),
                HumanMessage(content=sql_prompt)
            ]
            
            response = self.llm.invoke(messages)
            sql_query = response.content.strip()
            
            # Remove any markdown formatting
            if sql_query.startswith("```sql"):
                sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            elif sql_query.startswith("```"):
                sql_query = sql_query.replace("```", "").strip()
            
            logger.info(f"Generated SQL query: {sql_query}")
            
            # Execute the custom query
            results = self.tools.cockroach.execute_custom_query(sql_query)
            
            if results:
                # Extract float IDs from results
                float_ids = []
                for row in results:
                    if 'platform_number' in row:
                        float_ids.append(row['platform_number'])
                    elif len(row) == 1:  # Single column result
                        float_ids.append(list(row.values())[0])
                
                return {
                    "agent": "MeasurementAgent",
                    "query_type": "float_ids",
                    "sql_query": sql_query,
                    "float_ids": float_ids,
                    "count": len(float_ids),
                    "summary": f"Found {len(float_ids)} float IDs"
                }
            else:
                return {
                    "agent": "MeasurementAgent", 
                    "query_type": "float_ids",
                    "error": "No results found",
                    "sql_query": sql_query
                }
                
        except Exception as e:
            logger.error(f"Error in float ID query: {e}")
            return {
                "agent": "MeasurementAgent",
                "query_type": "float_ids", 
                "error": f"Failed to execute query: {str(e)}"
            }

    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate basic statistics"""
        if not values:
            return {}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values))
        }
    
    def _get_spatial_coverage(self, measurements) -> Dict[str, Any]:
        """Calculate spatial coverage"""
        lats = [m.latitude for m in measurements]
        lons = [m.longitude for m in measurements]
        return {
            "lat_range": [min(lats), max(lats)],
            "lon_range": [min(lons), max(lons)],
            "center": [np.mean(lats), np.mean(lons)]
        }

class MetadataAgent:
    """Specialized agent for float and region metadata"""
    
    def __init__(self, tools: ArgoToolFactory):
        self.tools = tools
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=0.1
        )
    
    def process(self, query: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Process metadata-related queries"""
        try:
            logger.info("MetadataAgent processing query")
            
            # Check if this requires custom graph query
            if any(keyword in query.lower() for keyword in [
                "all regions", "region list", "float count", "region hierarchy", 
                "parameters measured", "deployment info", "region statistics"
            ]):
                return self._handle_custom_metadata_query(query)
            
            if intent.get("float_id"):
                metadata = self.tools.neo4j.get_float_metadata(intent["float_id"])
                if metadata:
                    return {
                        "agent": "MetadataAgent",
                        "float_metadata": {
                            "platform_number": metadata.platform_number,
                            "parameters": metadata.parameters,
                            "region": metadata.subregion,
                            "deployment_date": getattr(metadata, 'deployment_date', None)
                        },
                        "summary": f"Float {intent['float_id']} measures {', '.join(metadata.parameters)} in {metadata.subregion}"
                    }
            
            if intent.get("region_name"):
                metadata = self.tools.neo4j.get_region_metadata(intent["region_name"])
                if metadata:
                    return {
                        "agent": "MetadataAgent",
                        "region_metadata": {
                            "name": metadata.name,
                            "parent_region": metadata.parent_region,
                            "float_count": metadata.float_count,
                            "subregions": metadata.subregions
                        },
                        "summary": f"{metadata.name} has {metadata.float_count} active floats"
                    }
            
            return {
                "agent": "MetadataAgent",
                "summary": "No metadata found for the specified criteria"
            }
            
        except Exception as e:
            logger.error(f"MetadataAgent error: {e}")
            return {"agent": "MetadataAgent", "error": str(e)}
    
    def _handle_custom_metadata_query(self, query: str) -> Dict[str, Any]:
        """Handle metadata queries requiring custom Cypher queries"""
        try:
            # Use LLM to generate appropriate Cypher query
            cypher_prompt = f"""
            Generate a Cypher query for Neo4j to answer this question: "{query}"
            
            The Neo4j database has these node types and relationships:
            - Float nodes: (f:Float {{platform_number: string, deployment_date: date}})
            - Region nodes: (r:Region {{name: string}})
            - Parameter nodes: (p:Parameter {{name: string}})
            
            Relationships:
            - (f:Float)-[:LOCATED_IN]->(r:Region)
            - (f:Float)-[:MEASURES]->(p:Parameter)
            - (r:Region)-[:PART_OF]->(parent:Region)
            
            IMPORTANT: Add LIMIT clause to prevent large result sets:
            - For listing queries, use LIMIT 50
            - For count queries, no limit needed
            
            Return ONLY the Cypher query, no explanation.
            
            Example queries:
            - "All regions" → MATCH (r:Region) RETURN r.name as region_name LIMIT 50
            - "Float count by region" → MATCH (f:Float)-[:LOCATED_IN]->(r:Region) RETURN r.name as region, count(f) as float_count
            - "Parameters measured" → MATCH (p:Parameter) RETURN p.name as parameter LIMIT 50
            - "Region hierarchy" → MATCH (r:Region)-[:PART_OF]->(parent:Region) RETURN r.name as region, parent.name as parent_region LIMIT 50
            """
            
            messages = [
                SystemMessage(content="You are a Cypher query expert. Generate only the Cypher query, no explanation."),
                HumanMessage(content=cypher_prompt)
            ]
            
            response = self.llm.invoke(messages)
            cypher_query = response.content.strip()
            
            # Remove any markdown formatting
            if cypher_query.startswith("```cypher"):
                cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
            elif cypher_query.startswith("```"):
                cypher_query = cypher_query.replace("```", "").strip()
            
            logger.info(f"Generated Cypher query: {cypher_query}")
            
            # Execute the custom query
            results = self.tools.neo4j.execute_custom_query(cypher_query)
            
            if results:
                return {
                    "agent": "MetadataAgent",
                    "query_type": "custom_metadata",
                    "cypher_query": cypher_query,
                    "results": results,
                    "count": len(results),
                    "summary": f"Found {len(results)} metadata records"
                }
            else:
                return {
                    "agent": "MetadataAgent", 
                    "query_type": "custom_metadata",
                    "error": "No results found",
                    "cypher_query": cypher_query
                }
                
        except Exception as e:
            logger.error(f"Error in custom metadata query: {e}")
            return {
                "agent": "MetadataAgent",
                "query_type": "custom_metadata", 
                "error": f"Failed to execute query: {str(e)}"
            }

class SemanticAgent:
    """Specialized agent for semantic search and pattern analysis"""
    
    def __init__(self, tools: ArgoToolFactory):
        self.tools = tools
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=0.1
        )
    
    def process(self, query: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Process semantic search queries"""
        try:
            logger.info("SemanticAgent processing query")
            
            # Generate embedding
            query_vector = self._get_query_embedding(query)
            
            # Perform search
            results = self.tools.pinecone.semantic_search(
                query_vector=query_vector,
                top_k=10,
                region_filter=intent.get("region_name")
            )
            
            if results:
                return {
                    "agent": "SemanticAgent",
                    "count": len(results),
                    "top_matches": [
                        {
                            "platform_number": r.platform_number,
                            "score": r.score,
                            "time": r.time.isoformat()
                        }
                        for r in results[:5]
                    ],
                    "summary": f"Found {len(results)} semantically similar measurements"
                }
            else:
                return {
                    "agent": "SemanticAgent",
                    "count": 0,
                    "summary": "No semantic matches found"
                }
                
        except Exception as e:
            logger.error(f"SemanticAgent error: {e}")
            return {"agent": "SemanticAgent", "error": str(e)}
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for semantic search"""
        try:
            import hashlib
            query_hash = hashlib.md5(query.lower().encode()).hexdigest()
            seed = int(query_hash[:8], 16)
            np.random.seed(seed)
            embedding = np.random.normal(0, 0.1, 384)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return [0.0] * 384

class CoordinatorAgent:
    """Coordinator agent that orchestrates other agents and synthesizes results"""
    
    def __init__(self):
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=0.3  # Slightly higher for creative synthesis
        )
    
    def synthesize_results(
        self,
        query: str,
        measurement_results: Optional[Dict[str, Any]],
        metadata_results: Optional[Dict[str, Any]],
        semantic_results: Optional[Dict[str, Any]]
    ) -> str:
        """Synthesize results from all agents into a comprehensive response"""
        try:
            logger.info("CoordinatorAgent synthesizing results")
            
            # Prepare context for synthesis
            context = {
                "query": query,
                "measurement_data": measurement_results,
                "metadata_data": metadata_results,
                "semantic_data": semantic_results
            }
            
            # Create synthesis prompt
            synthesis_prompt = f"""
            Answer this oceanographic query: "{query}"
            
            Data from agents:
            - Measurements: {json.dumps(measurement_results, indent=2) if measurement_results else "No data"}
            - Metadata: {json.dumps(metadata_results, indent=2) if metadata_results else "No data"}  
            - Semantic: {json.dumps(semantic_results, indent=2) if semantic_results else "No data"}
            
            Requirements:
            1. Give a direct, focused answer to the user's question
            2. Present key findings clearly using markdown tables for data
            3. Keep the response concise and actionable
            4. Only mention limitations if critical to understanding
            5. Do NOT include generic sections like "Recommendations" or "Areas for Further Investigation"
            6. Focus on the actual oceanographic insights from the data
            
            Use proper markdown formatting with tables for numerical data.
            """
            
            messages = [
                SystemMessage(content="""You are an expert oceanographer. Provide concise, focused answers to oceanographic queries. 
                Use markdown formatting: **bold** for emphasis, tables for data, ### for headings.
                Be direct and avoid unnecessary sections like "Recommendations" or "Areas for Further Investigation" unless specifically asked.
                Focus on answering the user's actual question with the available data."""),
                HumanMessage(content=synthesis_prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"CoordinatorAgent synthesis error: {e}")
            return f"Error synthesizing results: {str(e)}"

class MultiAgentArgoRAG:
    """Multi-agent RAG system for oceanographic data analysis"""
    
    def __init__(self):
        """Initialize the multi-agent system"""
        self.tools = ArgoToolFactory()
        
        # Initialize specialized agents
        self.measurement_agent = MeasurementAgent(self.tools)
        self.metadata_agent = MetadataAgent(self.tools)
        self.semantic_agent = SemanticAgent(self.tools)
        self.coordinator_agent = CoordinatorAgent()
        
        # Create the graph
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the multi-agent workflow graph"""
        
        def parse_intent(state: MultiAgentState) -> MultiAgentState:
            """Parse user query to determine which agents to activate"""
            try:
                query = state["query"]
                
                # Extract float ID
                float_id = None
                float_matches = re.findall(r'float (\d+)', query.lower())
                if float_matches:
                    float_id = float_matches[0]
                
                # Extract spatial information
                spatial_filter = None
                regions = {
                    "arabian sea": {"min_lat": 10, "max_lat": 25, "min_lon": 55, "max_lon": 75},
                    "bay of bengal": {"min_lat": 10, "max_lat": 25, "min_lon": 80, "max_lon": 95},
                    "equatorial indian ocean": {"min_lat": -5, "max_lat": 5, "min_lon": 40, "max_lon": 80},
                    "southern indian ocean": {"min_lat": -40, "max_lat": -20, "min_lon": 20, "max_lon": 80}
                }
                
                region_name = None
                for region, bounds in regions.items():
                    if region in query.lower():
                        spatial_filter = bounds
                        region_name = region.title()
                        break
                
                # Determine which agents to activate
                needs_measurements = any(word in query.lower() for word in 
                    ["temperature", "salinity", "pressure", "measurement", "data", "profile"])
                needs_metadata = any(word in query.lower() for word in 
                    ["metadata", "instrument", "parameter", "deployment", "coverage", "available"])
                needs_semantic = any(word in query.lower() for word in 
                    ["similar", "pattern", "inversion", "anomal", "compare", "find"])
                
                # If no specific indicators, activate all agents for comprehensive analysis
                if not any([needs_measurements, needs_metadata, needs_semantic]):
                    needs_measurements = needs_metadata = needs_semantic = True
                
                intent = {
                    "float_id": float_id,
                    "spatial_filter": spatial_filter,
                    "region_name": region_name,
                    "needs_measurements": needs_measurements,
                    "needs_metadata": needs_metadata,
                    "needs_semantic": needs_semantic
                }
                
                state["intent"] = intent
                logger.info(f"Multi-agent intent: {intent}")
                return state
                
            except Exception as e:
                logger.error(f"Error parsing intent: {e}")
                state["error"] = str(e)
                return state
        
        def execute_agents(state: MultiAgentState) -> MultiAgentState:
            """Execute relevant agents in parallel"""
            try:
                intent = state["intent"]
                query = state["query"]
                
                # Execute agents based on intent
                if intent["needs_measurements"]:
                    state["measurement_results"] = self.measurement_agent.process(query, intent)
                
                if intent["needs_metadata"]:
                    state["metadata_results"] = self.metadata_agent.process(query, intent)
                
                if intent["needs_semantic"]:
                    state["semantic_results"] = self.semantic_agent.process(query, intent)
                
                return state
                
            except Exception as e:
                logger.error(f"Error executing agents: {e}")
                state["error"] = str(e)
                return state
        
        def synthesize_response(state: MultiAgentState) -> MultiAgentState:
            """Synthesize results from all agents"""
            try:
                if state.get("error"):
                    state["final_response"] = f"Error: {state['error']}"
                    return state
                
                response = self.coordinator_agent.synthesize_results(
                    query=state["query"],
                    measurement_results=state.get("measurement_results"),
                    metadata_results=state.get("metadata_results"),
                    semantic_results=state.get("semantic_results")
                )
                
                state["final_response"] = response
                return state
                
            except Exception as e:
                logger.error(f"Error synthesizing response: {e}")
                state["final_response"] = f"Error synthesizing response: {str(e)}"
                return state
        
        # Create the graph
        workflow = StateGraph(MultiAgentState)
        
        # Add nodes
        workflow.add_node("parse_intent", parse_intent)
        workflow.add_node("execute_agents", execute_agents)
        workflow.add_node("synthesize_response", synthesize_response)
        
        # Add edges
        workflow.set_entry_point("parse_intent")
        workflow.add_edge("parse_intent", "execute_agents")
        workflow.add_edge("execute_agents", "synthesize_response")
        workflow.add_edge("synthesize_response", END)
        
        return workflow.compile()
    
    def query(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a query using the multi-agent system"""
        try:
            # Build conversation context
            messages = []
            if conversation_history:
                for msg in conversation_history[-5:]:  # Keep last 5 exchanges
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
            
            # Add current query
            messages.append(HumanMessage(content=query))
            
            # Initialize state
            initial_state = {
                "messages": messages,
                "query": query,
                "intent": None,
                "measurement_results": None,
                "metadata_results": None,
                "semantic_results": None,
                "final_response": None,
                "error": None
            }
            
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            return final_state.get("final_response", "No response generated")
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error processing query: {str(e)}"
    
    def close(self):
        """Clean up resources"""
        self.tools.close_all()