"""
Supabase client configuration for backend services
"""

from typing import Optional, Dict, Any, List
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
from structlog import get_logger
from app.core.config import settings

logger = get_logger(__name__)

class SupabaseClient:
    """Supabase client wrapper for backend services"""
    
    def __init__(self):
        self.url = settings.SUPABASE_URL or "http://127.0.0.1:54321"
        self.service_role_key = settings.SUPABASE_SERVICE_ROLE_KEY or ""
        self.anon_key = settings.SUPABASE_ANON_KEY or ""
        
        # Use service role key for backend operations (bypass RLS)
        key = self.service_role_key if self.service_role_key else self.anon_key
        
        if not key:
            logger.warning("No Supabase keys configured - using dummy client")
            self.client: Optional[Client] = None
        else:
            try:
                self.client = create_client(
                    self.url, 
                    key,
                    options=ClientOptions(
                        postgrest_client_timeout=10,
                        storage_client_timeout=10
                    )
                )
                logger.info("Supabase client initialized", url=self.url)
            except Exception as e:
                logger.error("Failed to initialize Supabase client", error=str(e))
                self.client = None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Supabase connection health"""
        if not self.client:
            return {"status": "disabled", "connected": False}
            
        try:
            # Simple query to test connection
            result = self.client.table("sub_accounts").select("count", count="exact").limit(0).execute()
            return {
                "status": "healthy",
                "connected": True,
                "url": self.url,
                "total_sub_accounts": result.count if result.count else 0
            }
        except Exception as e:
            logger.error("Supabase health check failed", error=str(e))
            return {
                "status": "error", 
                "connected": False,
                "error": str(e)
            }
    
    # Sub-Accounts operations
    async def create_sub_account(self, sub_account_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new sub-account"""
        if not self.client:
            raise Exception("Supabase client not initialized")
            
        try:
            result = self.client.table("sub_accounts").insert(sub_account_data).execute()
            logger.info("Sub-account created", 
                       sub_account_id=sub_account_data.get("sub_account_id"),
                       user=sub_account_data.get("user_address"))
            return {"success": True, "data": result.data[0] if result.data else None}
        except Exception as e:
            logger.error("Failed to create sub-account", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def get_user_sub_accounts(self, user_address: str) -> List[Dict[str, Any]]:
        """Get all sub-accounts for a user"""
        if not self.client:
            return []
            
        try:
            result = self.client.table("sub_accounts")\
                .select("*")\
                .eq("user_address", user_address)\
                .order("created_at", desc=True)\
                .execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error("Failed to get user sub-accounts", 
                        user=user_address, error=str(e))
            return []
    
    async def update_sub_account(self, sub_account_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update sub-account"""
        if not self.client:
            raise Exception("Supabase client not initialized")
            
        try:
            result = self.client.table("sub_accounts")\
                .update(updates)\
                .eq("sub_account_id", sub_account_id)\
                .execute()
            return {"success": True, "data": result.data[0] if result.data else None}
        except Exception as e:
            logger.error("Failed to update sub-account", 
                        sub_account_id=sub_account_id, error=str(e))
            return {"success": False, "error": str(e)}
    
    # Automation Rules operations
    async def create_automation_rule(self, rule_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new automation rule"""
        if not self.client:
            raise Exception("Supabase client not initialized")
            
        try:
            result = self.client.table("automation_rules").insert(rule_data).execute()
            logger.info("Automation rule created", 
                       rule_id=rule_data.get("rule_id"),
                       rule_type=rule_data.get("rule_type"))
            return {"success": True, "data": result.data[0] if result.data else None}
        except Exception as e:
            logger.error("Failed to create automation rule", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def get_user_automation_rules(self, user_address: str) -> List[Dict[str, Any]]:
        """Get all automation rules for a user"""
        if not self.client:
            return []
            
        try:
            result = self.client.table("automation_rules")\
                .select("*")\
                .eq("user_address", user_address)\
                .order("created_at", desc=True)\
                .execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error("Failed to get user automation rules", 
                        user=user_address, error=str(e))
            return []
    
    async def update_automation_rule(self, rule_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update automation rule"""
        if not self.client:
            raise Exception("Supabase client not initialized")
            
        try:
            result = self.client.table("automation_rules")\
                .update(updates)\
                .eq("rule_id", rule_id)\
                .execute()
            return {"success": True, "data": result.data[0] if result.data else None}
        except Exception as e:
            logger.error("Failed to update automation rule", 
                        rule_id=rule_id, error=str(e))
            return {"success": False, "error": str(e)}
    
    # Execution History operations
    async def log_execution(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Log automation execution"""
        if not self.client:
            raise Exception("Supabase client not initialized")
            
        try:
            result = self.client.table("automation_executions").insert(execution_data).execute()
            return {"success": True, "data": result.data[0] if result.data else None}
        except Exception as e:
            logger.error("Failed to log execution", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def get_execution_history(self, user_address: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get execution history for user"""
        if not self.client:
            return []
            
        try:
            result = self.client.table("automation_executions")\
                .select("*")\
                .eq("user_address", user_address)\
                .order("executed_at", desc=True)\
                .limit(limit)\
                .execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error("Failed to get execution history", 
                        user=user_address, error=str(e))
            return []

# Global Supabase client instance
_supabase_client: Optional[SupabaseClient] = None

def get_supabase_client() -> SupabaseClient:
    """Get global Supabase client instance"""
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = SupabaseClient()
    return _supabase_client

# Convenience functions
async def supabase_health_check() -> Dict[str, Any]:
    """Check Supabase health"""
    client = get_supabase_client()
    return await client.health_check()