"""
Main Agent - Single entry point for all queries
Uses LLM to handle conversational queries and route all oceanographic queries to specialized agents
"""

from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
import json
import logging
from datetime import datetime

from .config import GROQ_API_KEY, GROQ_MODEL
from .cyclic_multi_agent import CyclicMultiAgentArgoRAG

logger = logging.getLogger(__name__)

class MainAgent:
    """
    Main Agent that handles all queries using LLM intelligence.
    It engages in simple conversation and routes all substantive oceanographic queries
    to a specialized multi-agent system.
    """
    
    def __init__(self):
        """Initialize the Main Agent"""
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=0.1,
            max_tokens=1000
        )
        
        # Initialize specialized oceanographic agent (lazy loading)
        self._oceanographic_agent = None
        
        # System prompt for the main agent, updated for the new logic
        self.system_prompt = """You are Oceanus, a friendly AI assistant who is the primary interface for an advanced oceanographic data analysis system. Your main role is to greet users, handle simple conversation, and route any and all oceanographic questions to your specialized analysis system.

Your capabilities:
1. Handle conversational queries (greetings, thanks, how are you) with friendly, professional responses.
2. Identify ANY query related to oceanography, data, floats, or scientific concepts and route it for specialized analysis.

IMPORTANT DECISION MAKING:
- For PURELY conversational queries (e.g., "Hello", "Thank you", "How's it going?"): Answer directly.
- For ANY query containing oceanographic terms, asking for data, or asking for a definition (e.g., "What is salinity?", "Tell me about Argo floats", "Analyze float data"): You MUST route it to the specialized agent. Do not attempt to answer these questions yourself.

Your goal is to be a helpful gatekeeper, ensuring that users get the most accurate and detailed answers from the expert system you work with."""

    @property
    def oceanographic_agent(self):
        """Lazy initialization of oceanographic agent"""
        if self._oceanographic_agent is None:
            self._oceanographic_agent = CyclicMultiAgentArgoRAG()
        return self._oceanographic_agent
    
    def query(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Main query processing method. Handles conversation or routes to specialized agents.
        
        Args:
            query: User's query.
            conversation_history: Previous conversation context.
            
        Returns:
            Response string (either direct LLM response or from specialized agents).
        """
        try:
            # Build conversation context
            messages = [SystemMessage(content=self.system_prompt)]
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-10:]:  # Keep last 10 exchanges
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
            
            # Add current query with updated, simpler routing instruction
            routing_query = f"""User Query: "{query}"

Please analyze this query and decide:

1. If this is a PURELY conversational query (like a greeting, thanks, or general chit-chat), respond directly with a friendly answer.

2. If the query is about ANYTHING related to oceanography (including concepts, data, floats, regions, trends, or definitions), respond with exactly this format:
   ROUTE_TO_OCEANOGRAPHIC_AGENT: [brief explanation of why routing is needed]

Examples of queries to ROUTE:
- "Show me temperature data for float 1901442"
- "Analyze salinity trends in the Arabian Sea"
- "What is salinity?"
- "What can you do?"
- "Tell me about Argo floats."
- "List all float IDs"

Examples of queries to answer DIRECTLY:
- "Hello"
- "Thank you"
- "How are you?"
- "That's great!"
"""

            messages.append(HumanMessage(content=routing_query))
            
            # Get LLM response
            response = self.llm.invoke(messages)
            response_content = response.content.strip()
            
            # Check if LLM wants to route to specialized agent
            if response_content.startswith("ROUTE_TO_OCEANOGRAPHIC_AGENT:"):
                logger.info("Main Agent routing to oceanographic specialist")
                routing_reason = response_content.replace("ROUTE_TO_OCEANOGRAPHIC_AGENT:", "").strip()
                logger.info(f"Routing reason: {routing_reason}")
                
                # Route to specialized oceanographic agent
                return self._route_to_oceanographic_agent(query, conversation_history)
            
            # Return direct LLM response for simple conversational queries
            logger.info("Main Agent handling conversational query directly")
            return response_content
            
        except Exception as e:
            logger.error(f"Main Agent error: {e}")
            return f"I apologize, but I encountered an error processing your query. Please try again or rephrase your question. Error: {str(e)}"
    
    def _route_to_oceanographic_agent(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Route complex oceanographic queries to specialized multi-agent system.
        
        Args:
            query: Original user query.
            conversation_history: Conversation context.
            
        Returns:
            Response from specialized oceanographic agent.
        """
        try:
            logger.info("Routing to specialized oceanographic multi-agent system")
            
            # Add a brief introduction to the specialized analysis
            intro_message = "That's a great question! I'll forward it to my specialized multi-agent system for a detailed analysis. This may take a moment.\n\n"
            
            # Get response from specialized agent
            # Use _execute_full_analysis to bypass classification and force multi-agent processing
            specialized_response = self.oceanographic_agent._execute_full_analysis(query, conversation_history)
            
            return intro_message + specialized_response
            
        except Exception as e:
            logger.error(f"Error routing to oceanographic agent: {e}")
            return f"I apologize, but I encountered an error while analyzing your oceanographic data query. Please try again or contact support if the issue persists. Error: {str(e)}"

    def close(self):
        """Clean up resources"""
        if self._oceanographic_agent:
            # Assuming the specialized agent has a 'close' method
            self._oceanographic_agent.close()