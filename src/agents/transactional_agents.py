"""
Transactional Agents for handling order, return, and product queries
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import uuid


@dataclass
class Order:
    """Order data structure"""
    order_id: str
    customer_email: str
    status: str
    items: List[Dict[str, Any]]
    total_amount: float
    order_date: datetime
    estimated_delivery: datetime


@dataclass 
class Product:
    """Product data structure"""
    product_id: str
    name: str
    price: float
    stock_quantity: int
    category: str
    description: str


@dataclass
class Return:
    """Return data structure"""
    return_id: str
    order_id: str
    product_id: str
    reason: str
    status: str
    return_date: datetime


class OrderStatusAgent:
    """Agent for handling order-related queries"""
    
    def __init__(self):
        # Mock order database
        self.orders = self._generate_mock_orders()
    
    def _generate_mock_orders(self) -> Dict[str, Order]:
        """Generate mock order data"""
        orders = {}
        
        base_date = datetime.now()
        
        mock_orders_data = [
            {
                "order_id": "ORD-001",
                "customer_email": "john@example.com",
                "status": "shipped",
                "items": [{"product": "Laptop", "quantity": 1, "price": 999.99}],
                "total_amount": 999.99,
                "days_ago": 3
            },
            {
                "order_id": "ORD-002", 
                "customer_email": "jane@example.com",
                "status": "processing",
                "items": [{"product": "Phone", "quantity": 1, "price": 599.99}],
                "total_amount": 599.99,
                "days_ago": 1
            },
            {
                "order_id": "ORD-003",
                "customer_email": "bob@example.com", 
                "status": "delivered",
                "items": [{"product": "Headphones", "quantity": 2, "price": 149.99}],
                "total_amount": 299.98,
                "days_ago": 7
            }
        ]
        
        for order_data in mock_orders_data:
            order_date = base_date - timedelta(days=order_data["days_ago"])
            estimated_delivery = order_date + timedelta(days=5)
            
            orders[order_data["order_id"]] = Order(
                order_id=order_data["order_id"],
                customer_email=order_data["customer_email"],
                status=order_data["status"],
                items=order_data["items"],
                total_amount=order_data["total_amount"],
                order_date=order_date,
                estimated_delivery=estimated_delivery
            )
        
        return orders
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status by ID"""
        if order_id not in self.orders:
            return {
                "status": "error",
                "error": f"Order {order_id} not found"
            }
        
        order = self.orders[order_id]
        return {
            "status": "success",
            "data": {
                "order_id": order.order_id,
                "status": order.status,
                "items": order.items,
                "total_amount": order.total_amount,
                "order_date": order.order_date.strftime("%Y-%m-%d"),
                "estimated_delivery": order.estimated_delivery.strftime("%Y-%m-%d")
            }
        }
    
    def track_order(self, order_id: str) -> Dict[str, Any]:
        """Track order progress"""
        if order_id not in self.orders:
            return {
                "status": "error", 
                "error": f"Order {order_id} not found"
            }
        
        order = self.orders[order_id]
        tracking_info = {
            "order_id": order_id,
            "current_status": order.status,
            "timeline": self._get_order_timeline(order.status),
            "estimated_delivery": order.estimated_delivery.strftime("%Y-%m-%d")
        }
        
        return {
            "status": "success",
            "data": tracking_info
        }
    
    def _get_order_timeline(self, current_status: str) -> List[Dict[str, Any]]:
        """Get order timeline based on current status"""
        timeline_templates = {
            "processing": [
                {"step": "Order Received", "status": "completed", "date": "2024-11-20"},
                {"step": "Processing", "status": "current", "date": "2024-11-21"},
                {"step": "Shipped", "status": "pending", "date": "2024-11-22"},
                {"step": "Delivered", "status": "pending", "date": "2024-11-25"}
            ],
            "shipped": [
                {"step": "Order Received", "status": "completed", "date": "2024-11-20"},
                {"step": "Processing", "status": "completed", "date": "2024-11-21"},
                {"step": "Shipped", "status": "current", "date": "2024-11-22"},
                {"step": "Delivered", "status": "pending", "date": "2024-11-25"}
            ],
            "delivered": [
                {"step": "Order Received", "status": "completed", "date": "2024-11-20"},
                {"step": "Processing", "status": "completed", "date": "2024-11-21"}, 
                {"step": "Shipped", "status": "completed", "date": "2024-11-22"},
                {"step": "Delivered", "status": "completed", "date": "2024-11-24"}
            ]
        }
        
        return timeline_templates.get(current_status, [])
    
    def process_query(self, query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process order-related queries"""
        query_lower = query.lower()
        
        # Extract order ID if present
        order_id = self._extract_order_id(query)
        
        if "track" in query_lower or "tracking" in query_lower:
            if order_id:
                return self.track_order(order_id)
            else:
                return {
                    "status": "error",
                    "error": "Please provide an order ID to track your order"
                }
        
        elif "status" in query_lower or "check" in query_lower:
            if order_id:
                return self.get_order_status(order_id)
            else:
                return {
                    "status": "error", 
                    "error": "Please provide an order ID to check status"
                }
        
        else:
            return {
                "status": "success",
                "message": "I can help you track orders or check order status. Please provide your order ID."
            }
    
    def _extract_order_id(self, query: str) -> Optional[str]:
        """Extract order ID from query text"""
        import re
        # Look for patterns like ORD-001, ORDER-123, etc.
        patterns = [
            r'ORD-\d+',
            r'ORDER-\d+', 
            r'ord-\d+',
            r'order-\d+'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group().upper()
        
        return None


class ReturnAgent:
    """Agent for handling return-related queries"""
    
    def __init__(self):
        self.returns = self._generate_mock_returns()
        self.return_policy = {
            "return_window_days": 30,
            "conditions": [
                "Items must be in original condition",
                "Tags must be attached",
                "Original packaging preferred"
            ],
            "excluded_items": [
                "Personalized items",
                "Perishable goods",
                "Underwear and swimwear"
            ]
        }
    
    def _generate_mock_returns(self) -> Dict[str, Return]:
        """Generate mock return data"""
        returns = {}
        
        mock_returns_data = [
            {
                "return_id": "RET-001",
                "order_id": "ORD-001", 
                "product_id": "PROD-123",
                "reason": "Wrong size",
                "status": "processing"
            },
            {
                "return_id": "RET-002",
                "order_id": "ORD-002",
                "product_id": "PROD-456", 
                "reason": "Damaged item",
                "status": "approved"
            }
        ]
        
        for return_data in mock_returns_data:
            returns[return_data["return_id"]] = Return(
                return_id=return_data["return_id"],
                order_id=return_data["order_id"],
                product_id=return_data["product_id"],
                reason=return_data["reason"],
                status=return_data["status"],
                return_date=datetime.now() - timedelta(days=2)
            )
        
        return returns
    
    def initiate_return(self, order_id: str, product_id: str, reason: str) -> Dict[str, Any]:
        """Initiate a return request"""
        return_id = f"RET-{uuid.uuid4().hex[:6].upper()}"
        
        new_return = Return(
            return_id=return_id,
            order_id=order_id,
            product_id=product_id,
            reason=reason,
            status="initiated",
            return_date=datetime.now()
        )
        
        self.returns[return_id] = new_return
        
        return {
            "status": "success",
            "data": {
                "return_id": return_id,
                "status": "initiated",
                "next_steps": [
                    "Package the item in original packaging",
                    "Print the return label (check email)",
                    "Drop off at any authorized location"
                ]
            }
        }
    
    def get_return_status(self, return_id: str) -> Dict[str, Any]:
        """Get return status"""
        if return_id not in self.returns:
            return {
                "status": "error",
                "error": f"Return {return_id} not found"
            }
        
        return_obj = self.returns[return_id]
        return {
            "status": "success",
            "data": {
                "return_id": return_obj.return_id,
                "order_id": return_obj.order_id,
                "status": return_obj.status,
                "reason": return_obj.reason,
                "return_date": return_obj.return_date.strftime("%Y-%m-%d")
            }
        }
    
    def get_return_policy(self) -> Dict[str, Any]:
        """Get return policy information"""
        return {
            "status": "success",
            "data": self.return_policy
        }
    
    def process_query(self, query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process return-related queries - mainly for return status checking"""
        query_lower = query.lower()
        
        # Only handle specific return status checks and return initiation
        if "status" in query_lower and ("ret-" in query_lower or "return" in query_lower):
            # Look for return ID
            return_id = self._extract_return_id(query)
            if return_id:
                return self.get_return_status(return_id)
            else:
                return {
                    "status": "success",
                    "data": {
                        "message": "To check return status, I need your return ID (format: RET-XXXXXX). You can find this in your return confirmation email."
                    }
                }
        
        elif "initiate" in query_lower or "start" in query_lower or "how to return" in query_lower:
            return {
                "status": "success",
                "data": {
                    "message": "To start a return: 1) Go to our returns page, 2) Enter your order number and email, 3) Select items to return and reason, 4) Print the return label, 5) Package and ship the item back. You'll receive a return ID for tracking."
                }
            }
        
        else:
            # For policy questions, redirect to knowledge base
            return {
                "status": "success",
                "data": {
                    "message": "For return policy questions, please ask about our return policy and I'll provide detailed information from our knowledge base."
                }
            }
    
    def _extract_return_id(self, query: str) -> Optional[str]:
        """Extract return ID from query"""
        import re
        patterns = [r'RET-[A-Z0-9]+', r'ret-[a-z0-9]+']
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group().upper()
        
        return None


class ProductAgent:
    """Agent for handling product and inventory queries"""
    
    def __init__(self):
        self.products = self._generate_mock_products()
    
    def _generate_mock_products(self) -> Dict[str, Product]:
        """Generate mock product data"""
        products = {}
        
        mock_products_data = [
            {
                "product_id": "PROD-001",
                "name": "Gaming Laptop",
                "price": 1299.99,
                "stock_quantity": 15,
                "category": "Electronics",
                "description": "High-performance gaming laptop with RTX graphics"
            },
            {
                "product_id": "PROD-002",
                "name": "Wireless Headphones",
                "price": 199.99,
                "stock_quantity": 50,
                "category": "Audio",
                "description": "Premium noise-canceling wireless headphones"
            },
            {
                "product_id": "PROD-003",
                "name": "Smartphone",
                "price": 699.99,
                "stock_quantity": 0,
                "category": "Electronics", 
                "description": "Latest smartphone with advanced camera"
            },
            {
                "product_id": "PROD-004",
                "name": "Running Shoes",
                "price": 129.99,
                "stock_quantity": 25,
                "category": "Sports",
                "description": "Professional running shoes with premium comfort"
            }
        ]
        
        for product_data in mock_products_data:
            products[product_data["product_id"]] = Product(**product_data)
        
        return products
    
    def check_availability(self, product_id: str) -> Dict[str, Any]:
        """Check product availability"""
        if product_id not in self.products:
            return {
                "status": "error",
                "error": f"Product {product_id} not found"
            }
        
        product = self.products[product_id]
        is_available = product.stock_quantity > 0
        
        return {
            "status": "success",
            "data": {
                "product_id": product.product_id,
                "name": product.name,
                "available": is_available,
                "stock_quantity": product.stock_quantity,
                "price": product.price
            }
        }
    
    def search_products(self, query: str) -> Dict[str, Any]:
        """Search products by name or category"""
        query_lower = query.lower()
        matching_products = []
        
        for product in self.products.values():
            if (query_lower in product.name.lower() or 
                query_lower in product.category.lower() or
                query_lower in product.description.lower()):
                matching_products.append({
                    "product_id": product.product_id,
                    "name": product.name,
                    "price": product.price,
                    "category": product.category,
                    "available": product.stock_quantity > 0,
                    "stock_quantity": product.stock_quantity
                })
        
        return {
            "status": "success",
            "data": matching_products
        }
    
    def get_product_details(self, product_id: str) -> Dict[str, Any]:
        """Get detailed product information"""
        if product_id not in self.products:
            return {
                "status": "error",
                "error": f"Product {product_id} not found"
            }
        
        product = self.products[product_id]
        return {
            "status": "success", 
            "data": {
                "product_id": product.product_id,
                "name": product.name,
                "price": product.price,
                "stock_quantity": product.stock_quantity,
                "category": product.category,
                "description": product.description,
                "available": product.stock_quantity > 0
            }
        }
    
    def process_query(self, query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process product-related queries"""
        query_lower = query.lower()
        
        # Extract product ID if present
        product_id = self._extract_product_id(query)
        
        if "availability" in query_lower or "stock" in query_lower or "available" in query_lower:
            if product_id:
                return self.check_availability(product_id)
            else:
                return self.search_products(query)
        
        elif "search" in query_lower or "find" in query_lower:
            return self.search_products(query)
        
        elif "details" in query_lower or "info" in query_lower:
            if product_id:
                return self.get_product_details(product_id)
            else:
                return {
                    "status": "error",
                    "error": "Please provide a product ID for detailed information"
                }
        
        else:
            # General product search
            return self.search_products(query)
    
    def _extract_product_id(self, query: str) -> Optional[str]:
        """Extract product ID from query"""
        import re
        patterns = [r'PROD-\d+', r'prod-\d+']
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group().upper()
        
        return None


# Global agent instances
_order_agent_instance = None
_return_agent_instance = None
_product_agent_instance = None

def get_order_agent() -> OrderStatusAgent:
    """Get singleton order agent instance"""
    global _order_agent_instance
    if _order_agent_instance is None:
        _order_agent_instance = OrderStatusAgent()
    return _order_agent_instance

def get_return_agent() -> ReturnAgent:
    """Get singleton return agent instance"""
    global _return_agent_instance
    if _return_agent_instance is None:
        _return_agent_instance = ReturnAgent()
    return _return_agent_instance

def get_product_agent() -> ProductAgent:
    """Get singleton product agent instance"""
    global _product_agent_instance
    if _product_agent_instance is None:
        _product_agent_instance = ProductAgent()
    return _product_agent_instance