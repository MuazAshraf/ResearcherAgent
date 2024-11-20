import os
from dotenv import load_dotenv
from firecrawl.firecrawl import FirecrawlApp
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
from flask import Flask, request, jsonify

load_dotenv('.env')
app = Flask(__name__)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")

def validate_api_keys():
    if not SERP_API_KEY:
        raise ValueError("SERP_API_KEY not found in environment variables")
    if not FIRECRAWL_API_KEY:
        raise ValueError("FIRECRAWL_API_KEY not found in environment variables")

# Add this after load_dotenv()
validate_api_keys()

# 1. Tool for search
def search(query):
    url = "https://serpapi.com/search.json"
    params = {
        "q": query,
        "api_key": SERP_API_KEY
    }
    headers = {
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raises an exception for HTTP errors
        
        search_results = response.json()
        
        # Check if we have organic results
        if "organic_results" in search_results:
            # Format the results in a more usable way
            formatted_results = []
            for result in search_results["organic_results"][:5]:  # Get top 5 results
                formatted_results.append({
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", "")
                })
            return json.dumps(formatted_results, indent=2)
        else:
            return "No search results found."
            
    except requests.exceptions.RequestException as e:
        if response.status_code == 403:
            return "Error: Invalid or expired API key. Please check your SERP_API_KEY."
        return f"Error during search: {str(e)}"


# 2. Tool for scraping
def scrape_website(objective: str, url: str):
    """
    Scrape a given URL and extract content in Markdown format.
    """
    print("Scraping website with Firecrawl...")

    app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
    params = {
        "formats": ["markdown"]
    }
    
    try:
        response = app.scrape_url(url, params=params)
        
        if response and isinstance(response, dict):
            # Extract markdown content
            markdown_content = response.get('markdown', '')
            if markdown_content:
                # If content is too long, summarize it
                if len(markdown_content) > 10000:
                    return summary(objective, markdown_content)
                return markdown_content
            
            # If no markdown content but we have data
            if 'data' in response and isinstance(response['data'], dict):
                markdown_content = response['data'].get('markdown', '')
                if markdown_content:
                    if len(markdown_content) > 10000:
                        return summary(objective, markdown_content)
                    return markdown_content
                
        print("No markdown content found in the response.")
        return None
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"Error scraping {url}: {str(e)}"



def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output


class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name: str = "scrape_website" 
    description: str = (
        "Useful when you need to get data from a website URL, passing both URL and objective to the function; "
        "DO NOT make up any URL, the URL should only be from the search results."
    )
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")



# 3. Create langchain agent with the tools above
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results. 

    Follow these steps for EVERY research request:
    1. Start with a Search tool query to find recent and relevant information
    2. For each relevant search result, use the scrape_website tool to gather detailed information
    3. After gathering information from at least 2-3 sources, analyze and compile the findings
    4. If needed, perform additional targeted searches to fill any information gaps
    5. Compile all information into a comprehensive response that includes:
        - Main findings and analysis
        - Supporting data and statistics
        - Multiple perspectives if available
        - All sources used (URLs)
        
    Rules:
    - Use both Search and scrape_website tools for thorough research
    - Always verify information across multiple sources
    - Never make claims without backing them up with sources
    - Include URLs for all sources used
    - If a scraping attempt fails, try another source
    - Provide a structured response with clear sections
    - Maximum 3 iterations of search and scrape
    
    Format your response as:
    
    MAIN FINDINGS:
    [Key findings and analysis]
    
    SUPPORTING DATA:
    [Relevant statistics and data points]
    
    MULTIPLE PERSPECTIVES:
    [Different viewpoints if available]
    
    SOURCES:
    [List of all URLs used]
    """
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)


class Query(BaseModel):
    query: str


@app.route("/research", methods=['POST'])
def researchAgent():
    try:
        query_data = request.get_json()
        if not query_data or 'query' not in query_data:
            return jsonify({
                "status": "error",
                "message": "Invalid request. 'query' field is required"
            }), 400
        
        query = query_data.get("query")
        
        # Validate API keys before proceeding
        if "Invalid or expired API key" in search(query):
            return jsonify({
                "status": "error",
                "message": "API key validation failed. Please check your configuration."
            }), 500
        
        # Call the agent with the query
        content = agent({"input": query})
        actual_content = content['output']
        
        if not actual_content:
            return jsonify({
                "status": "error",
                "message": "No research results found"
            }), 404
        
        return jsonify({
            "status": "success",
            "response": actual_content
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
