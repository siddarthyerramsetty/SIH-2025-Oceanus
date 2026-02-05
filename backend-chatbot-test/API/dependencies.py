"""
FastAPI dependencies
"""

from fastapi import Depends, HTTPException, Request
from core.agent_manager import AgentManager
from core.session_manager import SessionManager

def get_agent_manager(request: Request) -> AgentManager:
    """
    Get the agent manager from application state.
    
    This dependency provides access to the initialized multi-agent system.
    """
    if not hasattr(request.app.state, 'agent_manager'):
        raise HTTPException(
            status_code=503,
            detail="Agent system not initialized"
        )
    
    agent_manager = request.app.state.agent_manager
    
    if not agent_manager:
        raise HTTPException(
            status_code=503,
            detail="Agent system not available"
        )
    
    return agent_manager

def get_session_manager(request: Request) -> SessionManager:
    """
    Get the session manager from application state.
    
    This dependency provides access to the session management system.
    """
    if not hasattr(request.app.state, 'session_manager'):
        raise HTTPException(
            status_code=503,
            detail="Session system not initialized"
        )
    
    session_manager = request.app.state.session_manager
    
    if not session_manager:
        raise HTTPException(
            status_code=503,
            detail="Session system not available"
        )
    
    return session_manager