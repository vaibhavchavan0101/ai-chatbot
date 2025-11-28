"""
Streamlit Chat Interface for E-Commerce Orchestrator
Live chatbot running on localhost with session management
"""
import streamlit as st
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, List

# Add src to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src'))

# Import orchestrator components
try:
    from orchestrator import get_orchestrator
    from tools.ecom_rag_tool import ecom_rag_tool
    from tools.order_tool import order_tool
    from tools.returns_tool import returns_tool
    from tools.inventory_tool import inventory_tool
    ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    ORCHESTRATOR_AVAILABLE = False
    st.error(f"Orchestrator not available: {e}")


class StreamlitChatbot:
    """Streamlit chatbot interface"""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        
        # Tool registry
        self.tools = {
            "ecom_rag_tool": ecom_rag_tool,
            "order_tool": order_tool,
            "returns_tool": returns_tool,
            "inventory_tool": inventory_tool
        }
        
        # Initialize orchestrator
        if ORCHESTRATOR_AVAILABLE:
            self.orchestrator = get_orchestrator()
        else:
            self.orchestrator = None
    
    def setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="E-Commerce AI Assistant",
            page_icon="🛒",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hello! I'm your E-Commerce AI Assistant. I can help you with:\n\n• **Orders** - Track orders, check status\n• **Returns** - Return policy, process returns\n• **Products** - Search inventory, check availability\n• **Information** - Policies, guides, FAQ\n\nWhat can I help you with today?",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }
            ]
        
        if "session_id" not in st.session_state:
            st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if "user_context" not in st.session_state:
            st.session_state.user_context = {}
    
    def render_sidebar(self):
        """Render sidebar with controls and information"""
        with st.sidebar:
            st.title("🛒 E-Commerce AI")
            st.markdown("---")
            
            # Session info
            st.subheader("Session Info")
            st.write(f"**Session ID:** {st.session_state.session_id}")
            st.write(f"**Messages:** {len(st.session_state.messages)}")
            
            st.markdown("---")
            
            # Quick examples
            st.subheader("💡 Try These Examples")
            
            example_queries = [
                "What is your return policy?",
                "Track order ORD-001",
                "Is product PROD-001 available?",
                "Check return status RET-001",
                "Search for gaming laptop",
                "How do I contact customer service?"
            ]
            
            for query in example_queries:
                if st.button(query, key=f"example_{hash(query)}"):
                    st.session_state.example_query = query
            
            st.markdown("---")
            
            # System status
            st.subheader("🔧 System Status")
            if ORCHESTRATOR_AVAILABLE:
                st.success("✅ Orchestrator Online")
            else:
                st.error("❌ Orchestrator Offline")
            
            # Clear chat button
            if st.button("🗑️ Clear Chat", type="secondary"):
                st.session_state.messages = st.session_state.messages[:1]  # Keep welcome message
                st.rerun()
            
            st.markdown("---")
            
            # Routing information
            with st.expander("ℹ️ How Routing Works"):
                st.markdown("""
                **RAG Keywords** → Knowledge Base
                - policy, rules, FAQ, guide, manual, terms
                
                **Transactional Keywords** → Live Data  
                - order, track, return, product, availability
                
                **Other** → Clarification Request
                """)
    
    def render_chat_interface(self):
        """Render main chat interface"""
        st.title("💬 E-Commerce AI Assistant")
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "timestamp" in message:
                        st.caption(f"🕐 {message['timestamp']}")
                    
                    # Show tool call information for assistant messages
                    if message["role"] == "assistant" and "tool_info" in message:
                        with st.expander("🔧 Technical Details"):
                            st.json(message["tool_info"])
        
        # Handle example query from sidebar
        if hasattr(st.session_state, 'example_query'):
            user_input = st.session_state.example_query
            del st.session_state.example_query
        else:
            # Chat input
            user_input = st.chat_input("Type your message here...")
        
        # Process user input
        if user_input:
            self.handle_user_input(user_input)
            st.rerun()
    
    def handle_user_input(self, user_input: str):
        """Handle user input and generate response"""
        # Add user message
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        })
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            response_data = self.process_query(user_input)
        
        # Add assistant response
        assistant_message = {
            "role": "assistant", 
            "content": response_data["content"],
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        
        if "tool_info" in response_data:
            assistant_message["tool_info"] = response_data["tool_info"]
        
        st.session_state.messages.append(assistant_message)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process user query through orchestrator"""
        if not self.orchestrator:
            return {
                "content": "Sorry, the orchestrator system is not available. Please check the system configuration.",
                "tool_info": {"error": "Orchestrator not initialized"}
            }
        
        try:
            # Route query through orchestrator
            routing_result = self.orchestrator.process_query(query, st.session_state.user_context)
            
            # Handle different response types
            if isinstance(routing_result, dict) and "tool" in routing_result:
                # Execute tool
                return self.execute_tool_call(routing_result)
            
            elif isinstance(routing_result, dict):
                # Direct response from orchestrator
                return self.format_orchestrator_response(routing_result)
            
            else:
                # Clarification text
                return {
                    "content": routing_result,
                    "tool_info": {"type": "clarification", "query": query}
                }
        
        except Exception as e:
            return {
                "content": f"I encountered an error processing your request: {str(e)}",
                "tool_info": {"error": str(e), "query": query}
            }
    
    def execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call and format response"""
        tool_name = tool_call["tool"]
        tool_args = tool_call["arguments"]
        
        try:
            if tool_name in self.tools:
                # Execute the tool
                tool_result = self.tools[tool_name](**tool_args)
                
                # Format response based on tool result
                content = self.format_tool_response(tool_result, tool_name)
                
                return {
                    "content": content,
                    "tool_info": {
                        "tool": tool_name,
                        "arguments": tool_args,
                        "result": tool_result
                    }
                }
            else:
                return {
                    "content": f"Tool '{tool_name}' is not available.",
                    "tool_info": {"error": f"Unknown tool: {tool_name}"}
                }
        
        except Exception as e:
            return {
                "content": f"Error executing {tool_name}: {str(e)}",
                "tool_info": {"error": str(e), "tool": tool_name}
            }
    
    def format_tool_response(self, tool_result: Dict[str, Any], tool_name: str) -> str:
        """Format tool response for display"""
        if not tool_result:
            return "I couldn't process your request. Please try again."
        
        if tool_result.get("status") == "error":
            return f"❌ **Error:** {tool_result.get('error', 'Unknown error occurred')}"
        
        if tool_result.get("status") == "success":
            # Handle different tool response formats
            if "answer" in tool_result:
                # RAG tool response
                answer = tool_result["answer"]
                sources = tool_result.get("sources", [])
                
                response = f"📚 **Answer:** {answer}"
                
                if sources:
                    response += f"\n\n📖 **Sources:** {len(sources)} documents referenced"
                
                return response
            
            elif "data" in tool_result:
                # Transactional tool response
                data = tool_result["data"]
                return self.format_transactional_data(data, tool_name)
            
            elif "message" in tool_result:
                return f"✅ {tool_result['message']}"
        
        return str(tool_result)
    
    def format_transactional_data(self, data: Any, tool_name: str) -> str:
        """Format transactional data for display"""
        if tool_name == "order_tool":
            return self.format_order_data(data)
        elif tool_name == "returns_tool":
            return self.format_return_data(data)
        elif tool_name == "inventory_tool":
            return self.format_inventory_data(data)
        else:
            return str(data)
    
    def format_order_data(self, data: Dict[str, Any]) -> str:
        """Format order data"""
        if isinstance(data, dict):
            if "timeline" in data:
                # Order tracking
                response = f"📦 **Order Tracking: {data.get('order_id', 'N/A')}**\n\n"
                response += f"**Current Status:** {data.get('current_status', 'Unknown')}\n"
                response += f"**Estimated Delivery:** {data.get('estimated_delivery', 'Unknown')}\n\n"
                
                timeline = data.get('timeline', [])
                if timeline:
                    response += "**Timeline:**\n"
                    for step in timeline:
                        status_icon = {"completed": "✅", "current": "🔄", "pending": "⏳"}.get(step.get('status'), "❓")
                        response += f"{status_icon} {step.get('step')} - {step.get('date')}\n"
                
                return response
            
            else:
                # Order status
                response = f"📦 **Order Status**\n\n"
                response += f"**Order ID:** {data.get('order_id', 'N/A')}\n"
                response += f"**Status:** {data.get('status', 'Unknown')}\n"
                response += f"**Total:** ${data.get('total_amount', 0.00):.2f}\n"
                response += f"**Order Date:** {data.get('order_date', 'Unknown')}\n"
                
                items = data.get('items', [])
                if items:
                    response += "\n**Items:**\n"
                    for item in items:
                        response += f"• {item.get('product', 'Unknown')} x{item.get('quantity', 1)}\n"
                
                return response
        
        return str(data)
    
    def format_return_data(self, data: Dict[str, Any]) -> str:
        """Format return data"""
        if isinstance(data, dict):
            if "return_window_days" in data:
                # Return policy
                response = "🔄 **Return Policy**\n\n"
                response += f"**Return Window:** {data.get('return_window_days', 30)} days\n\n"
                
                conditions = data.get('conditions', [])
                if conditions:
                    response += "**Conditions:**\n"
                    for condition in conditions:
                        response += f"• {condition}\n"
                
                excluded = data.get('excluded_items', [])
                if excluded:
                    response += "\n**Excluded Items:**\n"
                    for item in excluded:
                        response += f"• {item}\n"
                
                return response
            
            else:
                # Return status
                response = "🔄 **Return Status**\n\n"
                response += f"**Return ID:** {data.get('return_id', 'N/A')}\n"
                response += f"**Order ID:** {data.get('order_id', 'N/A')}\n"
                response += f"**Status:** {data.get('status', 'Unknown')}\n"
                response += f"**Reason:** {data.get('reason', 'N/A')}\n"
                response += f"**Return Date:** {data.get('return_date', 'Unknown')}\n"
                
                if "next_steps" in data:
                    response += "\n**Next Steps:**\n"
                    for step in data["next_steps"]:
                        response += f"• {step}\n"
                
                return response
        
        return str(data)
    
    def format_inventory_data(self, data: Any) -> str:
        """Format inventory data"""
        if isinstance(data, list):
            # Product search results
            if not data:
                return "🔍 **Search Results:** No products found"
            
            response = f"🔍 **Search Results:** Found {len(data)} products\n\n"
            
            for product in data[:5]:  # Show first 5 results
                available = "✅ In Stock" if product.get('available', False) else "❌ Out of Stock"
                response += f"**{product.get('name', 'Unknown Product')}**\n"
                response += f"• Price: ${product.get('price', 0.00):.2f}\n"
                response += f"• Status: {available}\n"
                response += f"• Category: {product.get('category', 'Unknown')}\n\n"
            
            return response
        
        elif isinstance(data, dict):
            # Single product info
            available = "✅ In Stock" if data.get('available', False) else "❌ Out of Stock"
            response = f"🏷️ **Product Information**\n\n"
            response += f"**Name:** {data.get('name', 'Unknown')}\n"
            response += f"**Price:** ${data.get('price', 0.00):.2f}\n"
            response += f"**Status:** {available}\n"
            response += f"**Stock:** {data.get('stock_quantity', 0)} units\n"
            response += f"**Category:** {data.get('category', 'Unknown')}\n"
            
            if "description" in data:
                response += f"**Description:** {data['description']}\n"
            
            return response
        
        return str(data)
    
    def format_orchestrator_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Format direct orchestrator response"""
        if response.get("status") == "success":
            content = response.get("message", "Request processed successfully.")
        else:
            content = f"❌ **Error:** {response.get('error', 'Unknown error')}"
        
        return {
            "content": content,
            "tool_info": response
        }
    
    def run(self):
        """Run the Streamlit app"""
        self.render_sidebar()
        self.render_chat_interface()


def main():
    """Main function to run the Streamlit app"""
    chatbot = StreamlitChatbot()
    chatbot.run()


if __name__ == "__main__":
    main()