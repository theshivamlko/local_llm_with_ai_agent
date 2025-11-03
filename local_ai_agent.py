from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import httpx
import json
from typing import Optional, List, Dict
import re

# Initialize FastAPI app
app = FastAPI(title="ReAct AI Agent with Web Search")

# 1. Path to your model
model_path = "Qwen/Qwen3-4B-Instruct-2507"
api_key = "<API_KEY>"  # Replace with your actual API key
cx = "<CUSTOM_SEARCH_ENGINE_ID>"  # Replace with your actual Custom Search Engine ID


# 2. Initialize vLLM with your model
llm = LLM(
    model=model_path,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    max_model_len=27152,  # Set to fit available KV cache memory
    trust_remote_code=True,
    disable_custom_all_reduce=True,
)

# 3. Choose sampling parameters
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512, stop=["Observation:"])


# Pydantic models for request/response
class GenerateRequest(BaseModel):
    prompt: str
    max_iterations: int = 5
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9


class Step(BaseModel):
    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None


class GenerateResponse(BaseModel):
    prompt: str
    final_answer: str
    reasoning_steps: List[Step]
    tool_calls: int


# Tool: Web Search using Google Custom Search API
async def web_search_tool(query: str) -> str:
    """
    Web Search Tool - Searches the internet using Google Custom Search API and scrapes page content
    Args:
        query: The search query string
    Returns:
        Formatted string with scraped web page content
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:

            search_url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={query}"
            
            # Get search results
            search_response = await client.get(search_url)
            
            if search_response.status_code != 200:
                return f"Error: Could not reach search service (Status: {search_response.status_code})"
            
            search_data = search_response.json()
            
            # Parse items array
            items = search_data.get("items", [])
            
            if not items:
                return "No search results found. Try rephrasing your query."
            
            # Scrape each page
            scraped_pages = []
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            for i, item in enumerate(items[:3], 1):  # Limit to first 3 results
                link = item.get("link", "")
                if not link:
                    continue
                
                try:
                    # Scrape the web page
                    page_response = await client.get(link, headers=headers, timeout=10.0, follow_redirects=True)
                    
                    if page_response.status_code == 200:
                        # Extract text content from HTML
                        html_content = page_response.text
                        
                        # Remove script and style tags
                        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
                        html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
                        
                        # Remove HTML tags
                        text_content = re.sub(r'<[^>]+>', ' ', html_content)
                        
                        # Clean up whitespace
                        text_content = re.sub(r'\s+', ' ', text_content).strip()
                        
                        # Limit content length to avoid overwhelming the LLM
                        max_length = 2000
                        if len(text_content) > max_length:
                            text_content = text_content[:max_length] + "..."
                        
                        scraped_pages.append(f"Page {i}:\nURL: {link}\n{text_content}")
                    else:
                        scraped_pages.append(f"Page {i}:\nURL: {link}\nError: Could not fetch page (Status: {page_response.status_code})")
                
                except Exception as e:
                    scraped_pages.append(f"Page {i}:\nURL: {link}\nError scraping page: {str(e)}")
            
            if scraped_pages:
                return "\n\n" + "="*80 + "\n\n".join(scraped_pages)
            else:
                return "No pages could be scraped successfully."
                
    except Exception as e:
        return f"Error performing search: {str(e)}"


# Available tools
TOOLS = {
    "WebSearch": {
        "function": web_search_tool,
        "description": "Searches the internet using Google Custom Search API and scrapes full page content for current information, facts, news, or any topic. Returns detailed content from top search results. Use this when you need recent information or facts you don't know."
    }
}


# ReAct Prompt Template
REACT_PROMPT_TEMPLATE = """You are an AI assistant that uses the ReAct (Reasoning + Acting) framework to answer questions.

You have access to the following tools:
- WebSearch: Searches the internet for current information. Use this when you need recent information or facts you don't know.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [WebSearch, Final Answer]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Action: Final Answer
Action Input: the final answer to the original input question

Begin!

Question: {question}
Thought:"""


def parse_react_output(text: str) -> Dict[str, str]:
    """Parse the LLM output to extract Thought, Action, and Action Input"""
    result = {
        "thought": "",
        "action": "",
        "action_input": ""
    }
    
    # Extract Thought
    thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', text, re.DOTALL)
    if thought_match:
        result["thought"] = thought_match.group(1).strip()
    
    # Extract Action
    action_match = re.search(r'Action:\s*(.+?)(?=Action Input:|$)', text, re.DOTALL)
    if action_match:
        result["action"] = action_match.group(1).strip()
    
    # Extract Action Input
    action_input_match = re.search(r'Action Input:\s*(.+?)(?=Observation:|$)', text, re.DOTALL)
    if action_input_match:
        result["action_input"] = action_input_match.group(1).strip()
    
    return result


async def react_agent(question: str, max_iterations: int = 5, 
                     temperature: float = 0.7, top_p: float = 0.9) -> tuple[str, List[Step], int]:
    """
    ReAct Agent that uses vLLM and decides when to use tools
    
    Returns:
        (final_answer, reasoning_steps, tool_calls_count)
    """
    reasoning_steps = []
    tool_calls = 0
    
    # Start with the initial prompt
    current_prompt = REACT_PROMPT_TEMPLATE.format(question=question)
    
    for iteration in range(max_iterations):
        try:
            # Generate LLM response using vLLM
            custom_sampling = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=512,
                stop=["Observation:"]
            )
            
            outputs = llm.generate([current_prompt], custom_sampling)
            llm_output = outputs[0].outputs[0].text.strip()
            
            # Parse the output
            parsed = parse_react_output(llm_output)
            
            # Create step object
            step = Step(
                thought=parsed["thought"],
                action=parsed["action"],
                action_input=parsed["action_input"]
            )
            
            # Check if we have a final answer
            if "final answer" in parsed["action"].lower():
                step.observation = "Task completed"
                reasoning_steps.append(step)
                return parsed["action_input"], reasoning_steps, tool_calls
            
            # Execute tool if action is recognized
            if parsed["action"] == "WebSearch":
                tool_calls += 1
                print(f"WebSearch Call ##{tool_calls}: WebSearch with input: {parsed['action_input']}")
                observation = await web_search_tool(parsed["action_input"])
                step.observation = observation
                reasoning_steps.append(step)
                
                # Add observation to prompt for next iteration
                current_prompt += f"{llm_output}\nObservation: {observation}\nThought:"
            else:
                # If action not recognized, try to get final answer
                step.observation = "Invalid action, attempting to provide answer"
                reasoning_steps.append(step)
                
                # Try one more time with a simpler prompt using vLLM
                simple_prompt = f"Question: {question}\n\nBased on the context, provide a clear and concise answer:\nAnswer:"
                try:
                    outputs = llm.generate([simple_prompt], custom_sampling)
                    final_answer = outputs[0].outputs[0].text.strip()
                except Exception as e:
                    final_answer = f"Error generating final answer: {str(e)}"
                
                return final_answer, reasoning_steps, tool_calls
        
        except Exception as e:
            # Handle vLLM generation errors
            error_step = Step(
                thought=f"Error during generation: {str(e)}",
                action="Error",
                action_input="",
                observation=f"Generation failed at iteration {iteration + 1}"
            )
            reasoning_steps.append(error_step)
            return f"Error during reasoning: {str(e)}", reasoning_steps, tool_calls
    
    # If max iterations reached
    return "I apologize, but I couldn't complete the reasoning process within the iteration limit.", reasoning_steps, tool_calls


# FastAPI endpoint
@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(request: GenerateRequest):
    """
    Generate AI response using ReAct framework with Web Search tool
    
    The agent will autonomously decide whether to use the WebSearch tool based on the question.
    
    - **prompt**: The user's question
    - **max_iterations**: Maximum reasoning iterations (default: 5)
    - **temperature**: Sampling temperature (default: 0.7)
    - **top_p**: Top-p sampling parameter (default: 0.9)
    """
    try:
        final_answer, reasoning_steps, tool_calls = await react_agent(
            question=request.prompt,
            max_iterations=request.max_iterations,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return GenerateResponse(
            prompt=request.prompt,
            final_answer=final_answer,
            reasoning_steps=reasoning_steps,
            tool_calls=tool_calls
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "ReAct AI Agent with Web Search Tool is running",
        "model": model_path,
        "framework": "ReAct (Reasoning + Acting)",
        "tools": list(TOOLS.keys()),
        "endpoints": {
            "/generate": "POST - Generate AI responses using ReAct framework",
        }
    }


# Run with: uvicorn local_ai_agent:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
