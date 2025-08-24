from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging

from app.services.transaction_simulator import HyperEVMSimulator, SimulationConfig, SimulationResult
from app.api.v1.auth import get_current_user
from app.models.user import User

logger = logging.getLogger(__name__)
router = APIRouter()

class TransactionRequest(BaseModel):
    to: str = Field(..., description="Target address")
    data: str = Field(default="0x", description="Transaction data")
    value: str = Field(default="0", description="Transaction value in wei")
    gas: Optional[str] = Field(None, description="Gas limit")
    gas_price: Optional[str] = Field(None, description="Gas price in wei")
    max_fee_per_gas: Optional[str] = Field(None, description="EIP-1559 max fee per gas")
    max_priority_fee_per_gas: Optional[str] = Field(None, description="EIP-1559 max priority fee")
    nonce: Optional[int] = Field(None, description="Transaction nonce")
    from_address: Optional[str] = Field(None, description="Sender address")

class StateOverride(BaseModel):
    balance: Optional[str] = Field(None, description="Override account balance")
    nonce: Optional[int] = Field(None, description="Override account nonce")
    code: Optional[str] = Field(None, description="Override contract code")
    state: Optional[Dict[str, str]] = Field(None, description="Override storage slots")
    state_diff: Optional[Dict[str, str]] = Field(None, description="Storage state diff")

class SimulationConfigRequest(BaseModel):
    block_number: Optional[str] = Field(None, description="Block number to simulate against")
    timestamp: Optional[int] = Field(None, description="Block timestamp override")
    gas_limit: Optional[str] = Field(None, description="Block gas limit override")
    base_fee: Optional[str] = Field(None, description="Block base fee override")
    state_overrides: Optional[Dict[str, StateOverride]] = Field(None, description="State overrides by address")
    trace_calls: bool = Field(True, description="Include call trace in results")
    validate_chainid: bool = Field(True, description="Validate chain ID")
    include_precompiles: bool = Field(True, description="Include HyperEVM precompile analysis")

class BundleSimulationRequest(BaseModel):
    transactions: List[TransactionRequest] = Field(..., description="List of transactions to simulate")
    config: Optional[SimulationConfigRequest] = Field(None, description="Simulation configuration")

class AccessListRequest(BaseModel):
    transaction: TransactionRequest = Field(..., description="Transaction to generate access list for")
    config: Optional[SimulationConfigRequest] = Field(None, description="Simulation configuration")

class GasEstimateRequest(BaseModel):
    transaction: TransactionRequest = Field(..., description="Transaction to estimate gas for")
    config: Optional[SimulationConfigRequest] = Field(None, description="Simulation configuration")

@router.post("/simulate", response_model=Dict[str, Any])
async def simulate_transaction(
    request: Dict[str, Any],
    config: Optional[SimulationConfigRequest] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Simulate a single transaction on HyperEVM.
    
    Returns detailed simulation results including gas usage, state changes,
    event logs, and execution trace.
    """
    try:
        simulator = HyperEVMSimulator()
        
        # Convert config if provided
        sim_config = None
        if config:
            state_overrides = {}
            if config.state_overrides:
                for addr, override in config.state_overrides.items():
                    state_overrides[addr] = {
                        k: v for k, v in override.dict().items() 
                        if v is not None
                    }
            
            sim_config = SimulationConfig(
                block_number=config.block_number,
                timestamp=config.timestamp,
                gas_limit=config.gas_limit,
                base_fee=config.base_fee,
                state_overrides=state_overrides,
                trace_calls=config.trace_calls,
                validate_chainid=config.validate_chainid,
                include_precompiles=config.include_precompiles
            )
        
        result = await simulator.simulate_transaction(request, sim_config)
        
        return {
            "success": True,
            "simulation_id": result.simulation_id,
            "gas_used": result.gas_used,
            "gas_limit": result.gas_limit,
            "effective_gas_price": result.effective_gas_price,
            "return_value": result.return_value,
            "revert_reason": result.revert_reason,
            "logs": result.logs,
            "state_changes": result.state_changes,
            "trace": result.trace,
            "precompile_calls": result.precompile_calls,
            "access_list": result.access_list,
            "balance_changes": result.balance_changes,
            "risk_analysis": result.risk_analysis,
            "block_number": result.block_number,
            "timestamp": result.timestamp
        }
        
    except Exception as e:
        logger.error(f"Transaction simulation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation failed: {str(e)}"
        )

@router.post("/simulate/bundle", response_model=Dict[str, Any])
async def simulate_bundle(
    request: BundleSimulationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Simulate a bundle of transactions on HyperEVM.
    
    Executes transactions sequentially and returns cumulative results.
    """
    try:
        simulator = HyperEVMSimulator()
        
        # Convert transactions and config
        transactions = [tx.dict() for tx in request.transactions]
        
        sim_config = None
        if request.config:
            state_overrides = {}
            if request.config.state_overrides:
                for addr, override in request.config.state_overrides.items():
                    state_overrides[addr] = {
                        k: v for k, v in override.dict().items() 
                        if v is not None
                    }
            
            sim_config = SimulationConfig(
                block_number=request.config.block_number,
                timestamp=request.config.timestamp,
                gas_limit=request.config.gas_limit,
                base_fee=request.config.base_fee,
                state_overrides=state_overrides,
                trace_calls=request.config.trace_calls,
                validate_chainid=request.config.validate_chainid,
                include_precompiles=request.config.include_precompiles
            )
        
        result = await simulator.simulate_bundle(transactions, sim_config)
        
        return {
            "success": True,
            "bundle_id": result.simulation_id,
            "transaction_results": [
                {
                    "gas_used": tx_result.gas_used,
                    "gas_limit": tx_result.gas_limit,
                    "effective_gas_price": tx_result.effective_gas_price,
                    "return_value": tx_result.return_value,
                    "revert_reason": tx_result.revert_reason,
                    "logs": tx_result.logs,
                    "state_changes": tx_result.state_changes,
                    "trace": tx_result.trace,
                    "precompile_calls": tx_result.precompile_calls,
                    "balance_changes": tx_result.balance_changes
                }
                for tx_result in result.transaction_results
            ],
            "total_gas_used": result.total_gas_used,
            "cumulative_state_changes": result.cumulative_state_changes,
            "bundle_trace": result.bundle_trace,
            "risk_analysis": result.risk_analysis,
            "block_number": result.block_number,
            "timestamp": result.timestamp
        }
        
    except Exception as e:
        logger.error(f"Bundle simulation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bundle simulation failed: {str(e)}"
        )

@router.post("/gas/estimate", response_model=Dict[str, Any])
async def estimate_gas(
    request: GasEstimateRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Estimate gas usage for a transaction on HyperEVM.
    
    Returns gas estimate with detailed breakdown.
    """
    try:
        simulator = HyperEVMSimulator()
        
        transaction = request.transaction.dict()
        
        sim_config = None
        if request.config:
            state_overrides = {}
            if request.config.state_overrides:
                for addr, override in request.config.state_overrides.items():
                    state_overrides[addr] = {
                        k: v for k, v in override.dict().items() 
                        if v is not None
                    }
            
            sim_config = SimulationConfig(
                block_number=request.config.block_number,
                timestamp=request.config.timestamp,
                gas_limit=request.config.gas_limit,
                base_fee=request.config.base_fee,
                state_overrides=state_overrides,
                trace_calls=request.config.trace_calls,
                validate_chainid=request.config.validate_chainid,
                include_precompiles=request.config.include_precompiles
            )
        
        gas_breakdown = await simulator.analyze_gas_breakdown(transaction, sim_config)
        
        return {
            "success": True,
            "gas_estimate": gas_breakdown["total_gas"],
            "gas_breakdown": gas_breakdown,
            "recommended_gas_limit": int(gas_breakdown["total_gas"] * 1.2),  # 20% buffer
            "estimated_cost": {
                "base_fee": gas_breakdown.get("base_fee", "0"),
                "priority_fee": gas_breakdown.get("priority_fee", "0"),
                "total_wei": str(int(gas_breakdown["total_gas"]) * int(gas_breakdown.get("gas_price", "0")))
            }
        }
        
    except Exception as e:
        logger.error(f"Gas estimation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gas estimation failed: {str(e)}"
        )

@router.post("/access-list/generate", response_model=Dict[str, Any])
async def generate_access_list(
    request: AccessListRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Generate optimized access list for a transaction on HyperEVM.
    
    Returns access list that can be used to reduce gas costs.
    """
    try:
        simulator = HyperEVMSimulator()
        
        transaction = request.transaction.dict()
        
        sim_config = None
        if request.config:
            state_overrides = {}
            if request.config.state_overrides:
                for addr, override in request.config.state_overrides.items():
                    state_overrides[addr] = {
                        k: v for k, v in override.dict().items() 
                        if v is not None
                    }
            
            sim_config = SimulationConfig(
                block_number=request.config.block_number,
                timestamp=request.config.timestamp,
                gas_limit=request.config.gas_limit,
                base_fee=request.config.base_fee,
                state_overrides=state_overrides,
                trace_calls=request.config.trace_calls,
                validate_chainid=request.config.validate_chainid,
                include_precompiles=request.config.include_precompiles
            )
        
        # First simulate to get access patterns
        result = await simulator.simulate_transaction(transaction, sim_config)
        
        return {
            "success": True,
            "access_list": result.access_list,
            "gas_savings": result.access_list.get("gas_savings", 0),
            "optimized_transaction": {
                **transaction,
                "access_list": result.access_list.get("access_list", [])
            }
        }
        
    except Exception as e:
        logger.error(f"Access list generation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Access list generation failed: {str(e)}"
        )

@router.get("/precompiles", response_model=Dict[str, Any])
async def get_precompiles(
    current_user: User = Depends(get_current_user)
):
    """
    Get list of available HyperEVM precompiles and their addresses.
    
    Returns information about HyperEVM-specific precompiled contracts.
    """
    try:
        simulator = HyperEVMSimulator()
        precompiles = await simulator.get_precompiles_info()
        
        return {
            "success": True,
            "precompiles": precompiles,
            "hypercore_bridge": precompiles.get("hypercore_bridge"),
            "name_service": precompiles.get("name_service"),
            "oracle_bridge": precompiles.get("oracle_bridge")
        }
        
    except Exception as e:
        logger.error(f"Failed to get precompiles info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get precompiles info: {str(e)}"
        )

@router.post("/debug/trace", response_model=Dict[str, Any])
async def debug_trace_transaction(
    request: Dict[str, Any],
    config: Optional[SimulationConfigRequest] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Generate detailed execution trace for a transaction on HyperEVM.
    
    Returns step-by-step execution trace with opcode-level details.
    """
    try:
        simulator = HyperEVMSimulator()
        
        # Convert config if provided
        sim_config = None
        if config:
            state_overrides = {}
            if config.state_overrides:
                for addr, override in config.state_overrides.items():
                    state_overrides[addr] = {
                        k: v for k, v in override.dict().items() 
                        if v is not None
                    }
            
            sim_config = SimulationConfig(
                block_number=config.block_number,
                timestamp=config.timestamp,
                gas_limit=config.gas_limit,
                base_fee=config.base_fee,
                state_overrides=state_overrides,
                trace_calls=True,  # Force trace calls for debug
                validate_chainid=config.validate_chainid,
                include_precompiles=config.include_precompiles
            )
        else:
            sim_config = SimulationConfig(trace_calls=True)
        
        result = await simulator.simulate_transaction(request, sim_config)
        
        return {
            "success": True,
            "trace": result.trace,
            "gas_used_by_step": result.trace.get("gas_by_step", []),
            "state_changes_by_step": result.trace.get("state_by_step", []),
            "precompile_interactions": result.precompile_calls,
            "execution_summary": {
                "total_steps": len(result.trace.get("steps", [])),
                "total_gas": result.gas_used,
                "reverted": bool(result.revert_reason),
                "revert_reason": result.revert_reason
            }
        }
        
    except Exception as e:
        logger.error(f"Debug trace failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Debug trace failed: {str(e)}"
        )

@router.post("/risk/analyze", response_model=Dict[str, Any])
async def analyze_transaction_risk(
    request: Dict[str, Any],
    config: Optional[SimulationConfigRequest] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Analyze transaction risks and MEV potential on HyperEVM.
    
    Returns risk assessment and MEV analysis.
    """
    try:
        simulator = HyperEVMSimulator()
        
        # Convert config if provided
        sim_config = None
        if config:
            state_overrides = {}
            if config.state_overrides:
                for addr, override in config.state_overrides.items():
                    state_overrides[addr] = {
                        k: v for k, v in override.dict().items() 
                        if v is not None
                    }
            
            sim_config = SimulationConfig(
                block_number=config.block_number,
                timestamp=config.timestamp,
                gas_limit=config.gas_limit,
                base_fee=config.base_fee,
                state_overrides=state_overrides,
                trace_calls=config.trace_calls,
                validate_chainid=config.validate_chainid,
                include_precompiles=config.include_precompiles
            )
        
        result = await simulator.simulate_transaction(request, sim_config)
        
        return {
            "success": True,
            "risk_analysis": result.risk_analysis,
            "mev_analysis": result.risk_analysis.get("mev_analysis", {}),
            "security_warnings": result.risk_analysis.get("security_warnings", []),
            "gas_optimization": result.risk_analysis.get("gas_optimization", {}),
            "recommendations": result.risk_analysis.get("recommendations", [])
        }
        
    except Exception as e:
        logger.error(f"Risk analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk analysis failed: {str(e)}"
        )

@router.get("/health", response_model=Dict[str, Any])
async def simulator_health():
    """
    Check HyperEVM simulator health and connectivity.
    
    Returns status of simulator components and RPC connections.
    """
    try:
        simulator = HyperEVMSimulator()
        health_status = await simulator.check_health()
        
        return {
            "success": True,
            "status": "healthy" if health_status["rpc_connected"] else "degraded",
            "rpc_connected": health_status["rpc_connected"],
            "latest_block": health_status["latest_block"],
            "chain_id": health_status["chain_id"],
            "precompiles_available": health_status["precompiles_available"],
            "timestamp": health_status["timestamp"]
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e),
            "timestamp": None
        }