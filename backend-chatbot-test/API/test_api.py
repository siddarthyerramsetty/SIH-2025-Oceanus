"""
Test script for the FastAPI application
"""

import asyncio
import httpx
import json
from typing import Dict, Any

async def test_api():
    """Test the API endpoints"""
    
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient() as client:
        
        print("🧪 Testing Oceanographic Multi-Agent RAG API")
        print("=" * 50)
        
        # Test 1: Health check
        print("\n1. Testing health check...")
        try:
            response = await client.get(f"{base_url}/health")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test 2: Detailed health check
        print("\n2. Testing detailed health check...")
        try:
            response = await client.get(f"{base_url}/health/detailed")
            print(f"   Status: {response.status_code}")
            health_data = response.json()
            print(f"   Overall Status: {health_data.get('status')}")
            print(f"   Agent Status: {health_data.get('components', {}).get('agent_system', {}).get('status')}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test 3: Simple chat query
        print("\n3. Testing simple chat query...")
        try:
            chat_request = {
                "query": "Show me data for float 7902073",
                "timeout": 60
            }
            
            response = await client.post(
                f"{base_url}/api/v1/chat",
                json=chat_request,
                timeout=70
            )
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                chat_data = response.json()
                print(f"   Response length: {len(chat_data.get('response', ''))}")
                print(f"   Metadata: {chat_data.get('metadata', {})}")
                print(f"   First 200 chars: {chat_data.get('response', '')[:200]}...")
            else:
                print(f"   Error response: {response.json()}")
                
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test 4: Complex chat query
        print("\n4. Testing complex chat query...")
        try:
            complex_request = {
                "query": "Analyze float 7902073: show me its measurements, metadata, and find similar patterns in the Arabian Sea",
                "timeout": 120
            }
            
            response = await client.post(
                f"{base_url}/api/v1/chat",
                json=complex_request,
                timeout=130
            )
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                chat_data = response.json()
                print(f"   Response length: {len(chat_data.get('response', ''))}")
                print(f"   Response time: {chat_data.get('metadata', {}).get('response_time')}s")
                print(f"   Agent type: {chat_data.get('metadata', {}).get('agent_type')}")
            else:
                print(f"   Error response: {response.json()}")
                
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test 5: Get examples
        print("\n5. Testing examples endpoint...")
        try:
            response = await client.get(f"{base_url}/api/v1/chat/examples")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                examples = response.json()
                print(f"   Available example categories: {list(examples.get('examples', {}).keys())}")
                print(f"   Supported regions: {len(examples.get('supported_regions', []))}")
                print(f"   Supported parameters: {len(examples.get('supported_parameters', []))}")
            
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test 6: Metrics
        print("\n6. Testing metrics endpoint...")
        try:
            response = await client.get(f"{base_url}/metrics")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                metrics = response.json()
                print(f"   Metrics available: {list(metrics.get('metrics', {}).keys())}")
            
        except Exception as e:
            print(f"   Error: {e}")
        
        print("\n" + "=" * 50)
        print("✅ API testing completed!")

if __name__ == "__main__":
    print("Starting API tests...")
    print("Make sure the API is running on http://localhost:8000")
    print("Run: python -m uvicorn api.main:app --reload")
    print()
    
    asyncio.run(test_api())