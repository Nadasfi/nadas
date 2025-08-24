"""
WebSocket API endpoints for real-time data streaming
Provides live market data and user events via WebSocket connections
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from datetime import datetime
import json
import asyncio
import uuid

from app.api.v1.auth import get_current_user
from app.services.websocket_manager import get_websocket_manager, initialize_websocket_subscriptions
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class WebSocketSubscriptionRequest(BaseModel):
    """WebSocket subscription request"""
    subscription_type: str  # 'market_data', 'user_events', 'all_mids'
    symbols: Optional[List[str]] = None
    user_address: Optional[str] = None


class MarketDataSnapshot(BaseModel):
    """Market data snapshot response"""
    symbol: str
    live_price: Optional[float]
    orderbook: Dict[str, Any]
    recent_trades: List[Dict[str, Any]]
    market_depth: Dict[str, Any]
    volatility: float
    timestamp: datetime


# Active WebSocket connections
active_connections: Dict[str, WebSocket] = {}
connection_subscriptions: Dict[str, List[str]] = {}


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Main WebSocket endpoint for real-time data streaming"""
    await websocket.accept()
    active_connections[client_id] = websocket
    connection_subscriptions[client_id] = []
    
    logger.info("WebSocket client connected", client_id=client_id)
    
    # Initialize WebSocket manager
    ws_manager = get_websocket_manager()
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            await handle_websocket_message(client_id, message, ws_manager)
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected", client_id=client_id)
    except Exception as e:
        logger.error("WebSocket error", client_id=client_id, error=str(e))
    finally:
        # Cleanup
        active_connections.pop(client_id, None)
        connection_subscriptions.pop(client_id, None)


async def handle_websocket_message(client_id: str, message: Dict[str, Any], ws_manager):
    """Handle incoming WebSocket messages"""
    try:
        message_type = message.get('type')
        websocket = active_connections.get(client_id)
        
        if not websocket:
            return
        
        if message_type == 'subscribe':
            await handle_subscription(client_id, message, ws_manager)
        
        elif message_type == 'unsubscribe':
            await handle_unsubscription(client_id, message)
        
        elif message_type == 'get_snapshot':
            await handle_snapshot_request(client_id, message, ws_manager)
        
        elif message_type == 'ping':
            await websocket.send_text(json.dumps({
                'type': 'pong',
                'timestamp': datetime.utcnow().isoformat()
            }))
            
        else:
            await websocket.send_text(json.dumps({
                'type': 'error',
                'message': f'Unknown message type: {message_type}'
            }))
            
    except Exception as e:
        logger.error("Error handling WebSocket message", 
                    client_id=client_id, error=str(e))


async def handle_subscription(client_id: str, message: Dict[str, Any], ws_manager):
    """Handle subscription requests"""
    try:
        websocket = active_connections.get(client_id)
        if not websocket:
            return
        
        subscription_type = message.get('subscription_type')
        symbols = message.get('symbols', [])
        user_address = message.get('user_address')
        
        if subscription_type == 'market_data' and symbols:
            # Subscribe to market data
            ws_manager.subscribe_to_market_data(symbols)
            
            # Register callback for this client
            for symbol in symbols:
                subscription_key = f"trade_{symbol}"
                ws_manager.register_callback(
                    subscription_key,
                    lambda data, cid=client_id, s=symbol: asyncio.create_task(
                        send_market_update(cid, s, data)
                    )
                )
                
                orderbook_key = f"orderbook_{symbol}"
                ws_manager.register_callback(
                    orderbook_key,
                    lambda data, cid=client_id, s=symbol: asyncio.create_task(
                        send_orderbook_update(cid, s, data)
                    )
                )
            
            connection_subscriptions[client_id].extend(symbols)
            
            # Send confirmation
            await websocket.send_text(json.dumps({
                'type': 'subscription_confirmed',
                'subscription_type': 'market_data',
                'symbols': symbols,
                'timestamp': datetime.utcnow().isoformat()
            }))
            
        elif subscription_type == 'user_events' and user_address:
            # Subscribe to user events
            ws_manager.subscribe_to_user_events(user_address)
            
            # Register callback for user events
            user_event_key = f"user_{user_address}"
            ws_manager.register_callback(
                user_event_key,
                lambda data, cid=client_id: asyncio.create_task(
                    send_user_event(cid, data)
                )
            )
            
            fill_event_key = f"fill_{user_address}"
            ws_manager.register_callback(
                fill_event_key,
                lambda data, cid=client_id: asyncio.create_task(
                    send_user_event(cid, data)
                )
            )
            
            connection_subscriptions[client_id].append(f"user_{user_address}")
            
            # Send confirmation
            await websocket.send_text(json.dumps({
                'type': 'subscription_confirmed',
                'subscription_type': 'user_events',
                'user_address': user_address,
                'timestamp': datetime.utcnow().isoformat()
            }))
            
        elif subscription_type == 'all_mids':
            # Subscribe to all mid prices
            ws_manager.register_callback(
                'all_mids',
                lambda data, cid=client_id: asyncio.create_task(
                    send_all_mids_update(cid, data)
                )
            )
            
            connection_subscriptions[client_id].append('all_mids')
            
            # Send confirmation
            await websocket.send_text(json.dumps({
                'type': 'subscription_confirmed',
                'subscription_type': 'all_mids',
                'timestamp': datetime.utcnow().isoformat()
            }))
            
        logger.info("WebSocket subscription added", 
                   client_id=client_id, 
                   type=subscription_type,
                   symbols=symbols,
                   user_address=user_address)
        
    except Exception as e:
        logger.error("Error handling subscription", 
                    client_id=client_id, error=str(e))


async def handle_unsubscription(client_id: str, message: Dict[str, Any]):
    """Handle unsubscription requests"""
    # TODO: Implement unsubscription logic
    pass


async def handle_snapshot_request(client_id: str, message: Dict[str, Any], ws_manager):
    """Handle snapshot requests"""
    try:
        websocket = active_connections.get(client_id)
        if not websocket:
            return
        
        symbol = message.get('symbol')
        if not symbol:
            return
        
        # Get current snapshot
        live_price = ws_manager.get_live_price(symbol)
        orderbook = ws_manager.get_latest_orderbook(symbol)
        recent_trades = ws_manager.get_recent_trades(symbol, 20)
        market_depth = ws_manager.get_market_depth(symbol)
        volatility = ws_manager.get_volatility_estimate(symbol)
        
        snapshot = {
            'type': 'market_snapshot',
            'symbol': symbol,
            'live_price': live_price,
            'orderbook': orderbook.to_dict() if orderbook else None,
            'recent_trades': [trade.to_dict() for trade in recent_trades],
            'market_depth': market_depth,
            'volatility': volatility,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await websocket.send_text(json.dumps(snapshot))
        
    except Exception as e:
        logger.error("Error handling snapshot request", 
                    client_id=client_id, error=str(e))


async def send_market_update(client_id: str, symbol: str, trade_data):
    """Send market update to specific client"""
    try:
        websocket = active_connections.get(client_id)
        if websocket:
            message = {
                'type': 'market_update',
                'symbol': symbol,
                'trade': trade_data.to_dict() if hasattr(trade_data, 'to_dict') else trade_data,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await websocket.send_text(json.dumps(message))
            
    except Exception as e:
        logger.error("Error sending market update", 
                    client_id=client_id, error=str(e))


async def send_orderbook_update(client_id: str, symbol: str, orderbook_data):
    """Send orderbook update to specific client"""
    try:
        websocket = active_connections.get(client_id)
        if websocket:
            message = {
                'type': 'orderbook_update',
                'symbol': symbol,
                'orderbook': orderbook_data.to_dict() if hasattr(orderbook_data, 'to_dict') else orderbook_data,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await websocket.send_text(json.dumps(message))
            
    except Exception as e:
        logger.error("Error sending orderbook update", 
                    client_id=client_id, error=str(e))


async def send_user_event(client_id: str, event_data):
    """Send user event to specific client"""
    try:
        websocket = active_connections.get(client_id)
        if websocket:
            message = {
                'type': 'user_event',
                'event': event_data.to_dict() if hasattr(event_data, 'to_dict') else event_data,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await websocket.send_text(json.dumps(message))
            
    except Exception as e:
        logger.error("Error sending user event", 
                    client_id=client_id, error=str(e))


async def send_all_mids_update(client_id: str, mids_data):
    """Send all mids update to specific client"""
    try:
        websocket = active_connections.get(client_id)
        if websocket:
            message = {
                'type': 'all_mids_update',
                'mids': mids_data,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await websocket.send_text(json.dumps(message))
            
    except Exception as e:
        logger.error("Error sending all mids update", 
                    client_id=client_id, error=str(e))


# REST API endpoints for WebSocket management

@router.get("/connections")
async def get_active_connections(current_user: dict = Depends(get_current_user)):
    """Get active WebSocket connections"""
    return {
        'active_connections': len(active_connections),
        'connection_ids': list(active_connections.keys()),
        'total_subscriptions': sum(len(subs) for subs in connection_subscriptions.values())
    }


@router.get("/market-snapshot/{symbol}")
async def get_market_snapshot(symbol: str, current_user: dict = Depends(get_current_user)):
    """Get current market snapshot for a symbol"""
    try:
        ws_manager = get_websocket_manager()
        
        # Ensure subscription
        ws_manager.subscribe_to_market_data([symbol])
        
        live_price = ws_manager.get_live_price(symbol)
        orderbook = ws_manager.get_latest_orderbook(symbol)
        recent_trades = ws_manager.get_recent_trades(symbol, 50)
        market_depth = ws_manager.get_market_depth(symbol)
        volatility = ws_manager.get_volatility_estimate(symbol)
        
        return MarketDataSnapshot(
            symbol=symbol,
            live_price=live_price,
            orderbook=orderbook.to_dict() if orderbook else {},
            recent_trades=[trade.to_dict() for trade in recent_trades],
            market_depth=market_depth,
            volatility=volatility,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error("Error getting market snapshot", symbol=symbol, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get market snapshot: {str(e)}")


@router.get("/websocket-stats")
async def get_websocket_stats(current_user: dict = Depends(get_current_user)):
    """Get WebSocket connection and data statistics"""
    try:
        ws_manager = get_websocket_manager()
        connection_stats = ws_manager.get_connection_stats()
        
        return {
            'websocket_manager': connection_stats,
            'api_connections': {
                'active_connections': len(active_connections),
                'total_subscriptions': sum(len(subs) for subs in connection_subscriptions.values()),
                'subscription_breakdown': {
                    client_id: len(subs) 
                    for client_id, subs in connection_subscriptions.items()
                }
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error getting WebSocket stats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.post("/initialize-subscriptions")
async def initialize_ws_subscriptions(current_user: dict = Depends(get_current_user)):
    """Initialize common WebSocket subscriptions"""
    try:
        await initialize_websocket_subscriptions()
        
        return {
            'status': 'success',
            'message': 'WebSocket subscriptions initialized',
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error initializing subscriptions", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to initialize: {str(e)}")


# Broadcast functions for sending data to all connected clients

async def broadcast_market_update(symbol: str, data: Dict[str, Any]):
    """Broadcast market update to all subscribed clients"""
    message = {
        'type': 'market_broadcast',
        'symbol': symbol,
        'data': data,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    message_str = json.dumps(message)
    
    for client_id, websocket in active_connections.items():
        subscriptions = connection_subscriptions.get(client_id, [])
        if symbol in subscriptions or 'all_mids' in subscriptions:
            try:
                await websocket.send_text(message_str)
            except Exception as e:
                logger.error("Error broadcasting to client", 
                           client_id=client_id, error=str(e))


async def broadcast_system_message(message: str):
    """Broadcast system message to all connected clients"""
    system_message = {
        'type': 'system_message',
        'message': message,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    message_str = json.dumps(system_message)
    
    for client_id, websocket in active_connections.items():
        try:
            await websocket.send_text(message_str)
        except Exception as e:
            logger.error("Error broadcasting system message", 
                       client_id=client_id, error=str(e))