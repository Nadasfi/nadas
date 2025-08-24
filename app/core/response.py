"""
Standardized API Response Helpers
"""

from typing import Any, Dict, Optional
from datetime import datetime
from pydantic import BaseModel


class APIResponse(BaseModel):
    """Standard API response model"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    message: Optional[str] = None
    timestamp: str
    
    class Config:
        arbitrary_types_allowed = True


def create_response(
    success: bool = True,
    data: Any = None,
    error: Optional[str] = None,
    message: Optional[str] = None
) -> Dict[str, Any]:
    """Create standardized API response"""
    
    response = {
        "success": success,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if data is not None:
        response["data"] = data
        
    if error:
        response["error"] = error
        response["success"] = False
        
    if message:
        response["message"] = message
    
    return response


def success_response(data: Any = None, message: str = "Success") -> Dict[str, Any]:
    """Create success response"""
    return create_response(success=True, data=data, message=message)


def error_response(error: str, data: Any = None) -> Dict[str, Any]:
    """Create error response"""
    return create_response(success=False, error=error, data=data)