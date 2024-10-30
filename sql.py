from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import OpenAI, ChatOpenAI
from langchain.sql_database import SQLDatabase
from langchain.agents.types import AgentType
from dotenv import load_dotenv
import os
import traceback

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'username': 'root',
    'password': 'root',
    'database': 'sba'
}

def create_financial_agent(db, model):
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    
    # Enhanced prompt template for financial analysis
    prefix = """
    You are an expert financial data analyst specializing in mutual fund performance analysis. Follow these guidelines:

    1. Data Interpretation Rules:
       - Always provide returns as percentages with 2 decimal places
       - For date ranges, use the format DD-MMM-YYYY
       - When calculating returns, ensure proper period alignment
       - If data is missing or incomplete, clearly state this in the response

    2. Response Format:
       - For mutual fund returns, always include the time period in the response
       - Present numerical values in a clear, consistent format
       - If multiple periods are available, specify which period is being reported

    3. Special Considerations:
       - For NAV calculations, verify the dates are business days
       - For year-to-date returns, adjust for partial periods
       - Handle missing data points gracefully

    4. Error Handling:
       - If data is not available, return "Data not available for the specified period"
       - If the query is outside the database scope, return "Information not available in database"
       - For any calculation errors, return "Unable to calculate returns due to insufficient data"

    Remember:
    - Validate all date ranges before processing
    - Ensure return calculations follow industry standards
    - Double-check the data consistency before returning results
    """

    return create_sql_agent(
        llm=model,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        prefix=prefix
    )

@app.route('/api/query', methods=['GET'])
def run_sql_query():
    try:
        user_query = request.args.get('query')
        
        if not user_query:
            return jsonify({
                "success": False, 
                "error": "Query parameter is required"
            }), 400

        # Initialize OpenAI model
        model = ChatOpenAI(model_name="gpt-4o", temperature=0)
        
        # Create database connection
        db_uri = f"mysql+pymysql://{DB_CONFIG['username']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
        db = SQLDatabase.from_uri(db_uri, sample_rows_in_table_info=3)
        
        # Create and execute agent
        agent_executor = create_financial_agent(db, model)
        result = agent_executor.run(user_query)
        
        # Format the response
        formatted_result = {
            "success": True,
            "result": result.strip() if isinstance(result, str) else result
        }
        
        return jsonify(formatted_result)

    except Exception as e:
        error_details = {
            "success": False,
            "error": str(e),
            "details": traceback.format_exc()
        }
        return jsonify(error_details), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)