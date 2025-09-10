#!/usr/bin/env python3
"""
Anemoi-Style Semi-Centralized Multi-Agent System with RL Retrieval
Combines agent-to-agent communication with RL-style metrics and retrieval.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import requests
import threading
from queue import Queue

# ---------------------------------------------------------------------------
#   Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#   Agent Types and Communication Protocol
# ---------------------------------------------------------------------------

class AgentType(Enum):
    PLANNER = "planner"
    EXECUTOR = "executor"
    RETRIEVAL = "retrieval"
    EVALUATOR = "evaluator"

@dataclass
class AgentMessage:
    sender: str
    receiver: str
    message_type: str
    content: Any
    task_id: str
    timestamp: float
    priority: int = 0

@dataclass
class TaskResult:
    task_id: str
    agent_id: str
    result: str
    confidence: float
    quality: float
    efficiency: float
    reward: float
    metadata: Dict[str, Any]

# ---------------------------------------------------------------------------
#   Ollama Manager
# ---------------------------------------------------------------------------

class OllamaManager:
    """Manages Ollama models automatically."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.available_models = set()
        self.required_models = {
            "minicpm-4.1-8b": "meta-planner",
            "gpt-oss-20b": "executor"
        }
    
    async def start_ollama(self):
        """Start Ollama service if not running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Ollama is already running")
                return True
        except:
            logger.info("Starting Ollama service...")
            try:
                subprocess.Popen(["ollama", "serve"], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
                await asyncio.sleep(3)  # Wait for service to start
                return True
            except FileNotFoundError:
                logger.error("Ollama not found. Please install Ollama first.")
                return False
    
    async def ensure_models(self):
        """Ensure required models are available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                self.available_models = {model["name"] for model in models}
                
                for model_name, role in self.required_models.items():
                    if model_name not in self.available_models:
                        logger.info(f"Pulling model {model_name} for {role}...")
                        await self._pull_model(model_name)
                    else:
                        logger.info(f"Model {model_name} already available for {role}")
                
                return True
        except Exception as e:
            logger.error(f"Failed to check models: {e}")
            return False
    
    async def _pull_model(self, model_name: str):
        """Pull a model from Ollama registry."""
        try:
            process = await asyncio.create_subprocess_exec(
                "ollama", "pull", model_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                logger.info(f"Successfully pulled {model_name}")
                self.available_models.add(model_name)
            else:
                logger.error(f"Failed to pull {model_name}: {stderr.decode()}")
        except Exception as e:
            logger.error(f"Error pulling {model_name}: {e}")

# ---------------------------------------------------------------------------
#   Agent-to-Agent Communication System
# ---------------------------------------------------------------------------

class A2ACommunicationHub:
    """Semi-centralized communication hub for agent-to-agent messaging."""
    
    def __init__(self):
        self.message_queue = Queue()
        self.agents: Dict[str, 'Agent'] = {}
        self.message_history: List[AgentMessage] = []
        self.running = False
    
    def register_agent(self, agent: 'Agent'):
        """Register an agent with the communication hub."""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_id} ({agent.agent_type.value})")
    
    def send_message(self, message: AgentMessage):
        """Send a message between agents."""
        self.message_queue.put(message)
        self.message_history.append(message)
    
    async def process_messages(self):
        """Process messages in the queue."""
        self.running = True
        while self.running:
            try:
                if not self.message_queue.empty():
                    message = self.message_queue.get_nowait()
                    await self._deliver_message(message)
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def _deliver_message(self, message: AgentMessage):
        """Deliver a message to the target agent."""
        if message.receiver in self.agents:
            agent = self.agents[message.receiver]
            await agent.receive_message(message)
        else:
            logger.warning(f"Unknown receiver: {message.receiver}")

# ---------------------------------------------------------------------------
#   Base Agent Class
# ---------------------------------------------------------------------------

class Agent:
    """Base agent class with A2A communication capabilities."""
    
    def __init__(self, agent_id: str, agent_type: AgentType, model_name: str, hub: A2ACommunicationHub):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.model_name = model_name
        self.hub = hub
        self.base_url = "http://localhost:11434/v1"
        self.message_buffer: List[AgentMessage] = []
        
        # Register with communication hub
        self.hub.register_agent(self)
    
    async def send_message(self, receiver: str, message_type: str, content: Any, task_id: str, priority: int = 0):
        """Send a message to another agent."""
        message = AgentMessage(
            sender=self.agent_id,
            receiver=receiver,
            message_type=message_type,
            content=content,
            task_id=task_id,
            timestamp=time.time(),
            priority=priority
        )
        self.hub.send_message(message)
    
    async def receive_message(self, message: AgentMessage):
        """Receive a message from another agent."""
        self.message_buffer.append(message)
        await self.process_message(message)
    
    async def process_message(self, message: AgentMessage):
        """Process received message - to be overridden by subclasses."""
        pass
    
    async def call_model(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> Dict[str, Any]:
        """Call the local model."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False
        }
        if tools:
            payload["tools"] = tools
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            
            choice = data["choices"][0]
            message = choice["message"]
            
            # Extract tool calls if present
            tool_calls = None
            if "tool_calls" in message and message["tool_calls"]:
                tool_calls = []
                for tc in message["tool_calls"]:
                    tool_calls.append({
                        "id": tc["id"],
                        "type": tc["type"],
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    })
            
            return {
                "content": message.get("content"),
                "tool_calls": tool_calls
            }
        except Exception as e:
            logger.error(f"Model call failed: {e}")
            return {"content": f"Error: {e}", "tool_calls": None}

# ---------------------------------------------------------------------------
#   Specialized Agents
# ---------------------------------------------------------------------------

class PlannerAgent(Agent):
    """Meta-planner agent that breaks down tasks."""
    
    def __init__(self, hub: A2ACommunicationHub):
        super().__init__("planner", AgentType.PLANNER, "minicpm-4.1-8b", hub)
        self.system_prompt = """You are the META-PLANNER in a semi-centralized multi-agent system. 
        Break down user queries into executable tasks and coordinate with other agents.
        Reply in JSON format: {"plan": [{"id": INT, "description": STRING, "agent": STRING}]}
        If final answer is ready, output: FINAL ANSWER: <answer>"""
    
    async def process_query(self, query: str, task_id: str) -> str:
        """Process a user query and coordinate with other agents."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Query: {query}\nTask ID: {task_id}"}
        ]
        
        response = await self.call_model(messages)
        content = response.get("content", "")
        
        if content and content.startswith("FINAL ANSWER:"):
            return content
        
        try:
            # Try to parse JSON plan
            if "{" in content and "}" in content:
                # Extract JSON from content
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
                plan = json.loads(json_str)
            else:
                # Create a simple plan if no JSON found
                plan = {
                    "plan": [{
                        "id": 1,
                        "description": query,
                        "agent": "executor"
                    }]
                }
            
            tasks = plan.get("plan", [])
            
            # Send tasks to appropriate agents
            for task in tasks:
                agent_type = task.get("agent", "executor")
                await self.send_message(
                    receiver=agent_type,
                    message_type="task_assignment",
                    content=task,
                    task_id=task_id,
                    priority=1
                )
            
            return f"Plan created with {len(tasks)} tasks"
        except Exception as e:
            # Fallback: create a simple task
            await self.send_message(
                receiver="executor",
                message_type="task_assignment",
                content={"id": 1, "description": query, "agent": "executor"},
                task_id=task_id,
                priority=1
            )
            return f"Created fallback task for: {query}"

class ExecutorAgent(Agent):
    """Executor agent that performs tasks with RL metrics."""
    
    def __init__(self, hub: A2ACommunicationHub):
        super().__init__("executor", AgentType.EXECUTOR, "gpt-oss-20b", hub)
        self.system_prompt = """You are an EXECUTOR agent. Execute tasks assigned by the planner.
        Use available tools and provide detailed results with confidence assessment."""
        self.tools = self._setup_tools()
        self.task_results: Dict[str, TaskResult] = {}
    
    def _setup_tools(self) -> List[Dict]:
        """Setup available tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "num_results": {"type": "integer", "default": 5}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "calculate",
                    "description": "Perform calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"}
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]
    
    async def process_message(self, message: AgentMessage):
        """Process task assignment messages."""
        if message.message_type == "task_assignment":
            await self.execute_task(message.content, message.task_id)
    
    async def execute_task(self, task: Dict[str, Any], task_id: str) -> TaskResult:
        """Execute a task with RL metrics."""
        start_time = time.time()
        task_desc = f"Task {task['id']}: {task['description']}"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task_desc}
        ]
        
        step_count = 0
        result_text = ""
        
        while step_count < 5:  # Max 5 steps per task
            step_count += 1
            response = await self.call_model(messages, self.tools)
            
            if response.get("content"):
                result_text = response["content"]
                break
            
            # Handle tool calls
            for call in response.get("tool_calls", []):
                tool_name = call["function"]["name"]
                tool_args = json.loads(call["function"].get("arguments", "{}"))
                tool_result = await self._execute_tool(tool_name, tool_args)
                
                messages.extend([
                    {"role": "assistant", "content": None, "tool_calls": [call]},
                    {"role": "tool", "tool_call_id": call["id"], "name": tool_name, "content": tool_result}
                ])
        
        # Calculate RL metrics
        duration = time.time() - start_time
        confidence = self._calculate_confidence(result_text)
        quality = self._calculate_quality(result_text, task_desc)
        efficiency = self._calculate_efficiency(step_count, duration, len(result_text))
        reward = self._calculate_reward(confidence, quality, efficiency)
        
        # Create task result
        task_result = TaskResult(
            task_id=task_id,
            agent_id=self.agent_id,
            result=result_text,
            confidence=confidence,
            quality=quality,
            efficiency=efficiency,
            reward=reward,
            metadata={
                "steps": step_count,
                "duration": duration,
                "task_description": task_desc
            }
        )
        
        self.task_results[task_id] = task_result
        
        # Send result back to planner
        await self.send_message(
            receiver="planner",
            message_type="task_result",
            content=task_result,
            task_id=task_id
        )
        
        return task_result
    
    async def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a tool."""
        if tool_name == "search":
            query = args.get("query", "")
            return f"Search results for '{query}': [Mock results - implement with real search]"
        elif tool_name == "calculate":
            expr = args.get("expression", "")
            try:
                result = eval(expr)
                return str(result)
            except:
                return "Calculation error"
        return f"Unknown tool: {tool_name}"
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score."""
        if not text:
            return 0.0
        
        confidence = 0.5
        text_lower = text.lower()
        
        # Negative indicators
        if any(word in text_lower for word in ["not sure", "don't know", "cannot", "unsure"]):
            confidence -= 0.3
        
        # Positive indicators
        if any(word in text_lower for word in ["definitely", "certainly", "clearly", "exactly"]):
            confidence += 0.2
        
        # Length and specificity
        if len(text) > 50:
            confidence += 0.1
        if any(char in text for char in ["%", "$", ":", "="]):
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_quality(self, result: str, task: str) -> float:
        """Calculate quality score."""
        if not result:
            return 0.0
        
        quality = 0.0
        
        # Length appropriateness
        if 10 <= len(result) <= 200:
            quality += 3.0
        elif len(result) > 200:
            quality += 2.0
        else:
            quality += 1.0
        
        # Completeness
        if any(word in result.lower() for word in ["complete", "finished", "done"]):
            quality += 2.0
        
        # Specificity
        if any(char in result for char in ["%", "$", ":", "=", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]):
            quality += 2.0
        
        # Relevance
        task_words = set(task.lower().split())
        result_words = set(result.lower().split())
        overlap = len(task_words.intersection(result_words))
        if overlap > 0:
            quality += min(3.0, overlap * 0.5)
        
        return min(10.0, max(0.0, quality))
    
    def _calculate_efficiency(self, steps: int, duration: float, result_length: int) -> float:
        """Calculate efficiency score."""
        step_score = max(0.0, 4.0 - (steps - 1) * 0.5)
        time_score = max(0.0, 3.0 - duration / 30.0)
        
        if result_length == 0:
            output_score = 0.0
        elif 50 <= result_length <= 200:
            output_score = 3.0
        else:
            output_score = 2.0
        
        return min(10.0, step_score + time_score + output_score)
    
    def _calculate_reward(self, confidence: float, quality: float, efficiency: float) -> float:
        """Calculate gradient reward."""
        base_reward = (confidence * 0.4 + quality * 0.4 + efficiency * 0.2) / 10.0
        
        if confidence >= 0.8:
            base_reward += 0.2
        if quality >= 8.0:
            base_reward += 0.2
        if efficiency >= 7.0:
            base_reward += 0.1
        
        return max(0.0, min(1.0, base_reward))

class RetrievalAgent(Agent):
    """Retrieval agent for RAG functionality."""
    
    def __init__(self, hub: A2ACommunicationHub):
        super().__init__("retrieval", AgentType.RETRIEVAL, "gpt-oss-20b", hub)
        self.system_prompt = """You are a RETRIEVAL agent. Provide relevant information to help other agents.
        Focus on accuracy and relevance."""
    
    async def process_message(self, message: AgentMessage):
        """Process retrieval requests."""
        if message.message_type == "retrieval_request":
            await self.handle_retrieval_request(message)
    
    async def handle_retrieval_request(self, message: AgentMessage):
        """Handle a retrieval request."""
        query = message.content.get("query", "")
        task_id = message.task_id
        
        # Mock retrieval - implement with real search
        retrieval_result = f"Retrieval results for '{query}': [Relevant information from knowledge base]"
        
        await self.send_message(
            receiver=message.sender,
            message_type="retrieval_result",
            content={"result": retrieval_result, "query": query},
            task_id=task_id
        )

# ---------------------------------------------------------------------------
#   Main Anemoi System
# ---------------------------------------------------------------------------

class AnemoiSystem:
    """Main semi-centralized multi-agent system."""
    
    def __init__(self):
        self.hub = A2ACommunicationHub()
        self.ollama_manager = OllamaManager()
        self.agents: Dict[str, Agent] = {}
        self.running = False
    
    async def initialize(self):
        """Initialize the system."""
        logger.info("Initializing Anemoi system...")
        
        # Start Ollama and ensure models
        if not await self.ollama_manager.start_ollama():
            raise RuntimeError("Failed to start Ollama")
        
        if not await self.ollama_manager.ensure_models():
            raise RuntimeError("Failed to ensure required models")
        
        # Create agents
        self.agents["planner"] = PlannerAgent(self.hub)
        self.agents["executor"] = ExecutorAgent(self.hub)
        self.agents["retrieval"] = RetrievalAgent(self.hub)
        
        # Start message processing
        asyncio.create_task(self.hub.process_messages())
        
        self.running = True
        logger.info("Anemoi system initialized successfully")
    
    async def process_query(self, query: str) -> str:
        """Process a user query through the multi-agent system."""
        task_id = str(uuid.uuid4())
        logger.info(f"Processing query: {query} (Task ID: {task_id})")
        
        # Start with planner
        planner = self.agents["planner"]
        result = await planner.process_query(query, task_id)
        
        # Wait for task completion with timeout
        max_wait = 10  # seconds
        start_wait = time.time()
        
        while time.time() - start_wait < max_wait:
            executor = self.agents["executor"]
            if task_id in executor.task_results:
                task_result = executor.task_results[task_id]
                return f"FINAL ANSWER: {task_result.result}\n\nMetrics: Confidence={task_result.confidence:.3f}, Quality={task_result.quality:.2f}, Reward={task_result.reward:.3f}"
            await asyncio.sleep(0.5)
        
        # If no result after timeout, return planning result
        return f"RESULT: {result}\n\nNote: Task execution may still be in progress."
    
    async def shutdown(self):
        """Shutdown the system."""
        self.running = False
        self.hub.running = False
        logger.info("Anemoi system shutdown")

# ---------------------------------------------------------------------------
#   Main execution
# ---------------------------------------------------------------------------

async def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Anemoi Semi-Centralized Multi-Agent System")
    parser.add_argument("-q", "--query", type=str, help="Query to process")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()
    
    system = AnemoiSystem()
    
    try:
        await system.initialize()
        
        if args.query:
            result = await system.process_query(args.query)
            print(f"\nResult: {result}")
        elif args.interactive:
            print("Anemoi Interactive Mode - Type 'exit' to quit")
            while True:
                query = input("\nQuery: ").strip()
                if query.lower() in ["exit", "quit"]:
                    break
                result = await system.process_query(query)
                print(f"\nResult: {result}")
        else:
            print("Use --query for single query or --interactive for interactive mode")
    
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await system.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
