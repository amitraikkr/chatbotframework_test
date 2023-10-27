import os
import json
import logging
import boto3
import sys

""" Testing in local SAM """
#sys.path.append("/opt/")
#sys.path.append("/opt/python/")
#sys.path.append("/opt/python/site-packages/")

from datetime import datetime, timedelta
from dbaseconnect import DynamoDBHelper
from openaiconnect import OpenAIAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Read configurations from parameters.txt
def load_configurations():
    with open("parameters.json", "r") as file:
        return json.load(file)

configurations = load_configurations()

# Constants
MAX_MSG_LIMIT = configurations["max_msg_limit"]

region_name = configurations["region_name"]

system_prompt = configurations["system_prompt"]


# Initialize the database and agent
dynamodb_resource = boto3.resource('dynamodb')
chat_context_db = DynamoDBHelper(dynamodb_resource, configurations["table_name"])


agent_config = configurations["agent_config"]
openai_agent = OpenAIAgent(agent_config,region_name)

def process_request(user_message: str, user_id: str) -> dict:
    """
    Process the user's request and generate a response using OpenAI.

    Parameters:
    - user_message (str): The user's message to the assistant.
    - user_id (str): The unique ID of the user.

    Returns:
    - dict: Contains the status and the response message.
    """
    # Retrieve existing context from DynamoDB
    try:
        user_data = chat_context_db.get_item(user_id)
        user_context = json.loads(user_data.get('context', '[]'))
        timestamps = json.loads(user_data.get('timestamps', '[]'))
    except Exception as e:
        logger.error(f"Error retrieving data from database: {e}")
        return {"status": "error", "message": "Internal Server Error"}

    # Manage 24-hour message limit
    now = datetime.utcnow()
    limit_time = now - timedelta(hours=24)
    valid_timestamps = [ts for ts in timestamps if datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S.%fZ') >= limit_time]

    if len(valid_timestamps) >= MAX_MSG_LIMIT:
        return {"status": "error", "message": "Message limit exceeded in 24 hours."}

    valid_timestamps.append(now.strftime('%Y-%m-%dT%H:%M:%S.%fZ'))

    # Update context & generate response
    if not user_context:
        system_message = {
            "role": "system",
            "content": system_prompt
        }
        user_context.append(system_message)

    user_context.append({"role": "user", "content": user_message})
    try:
        gpt_response = openai_agent.get_gpt_response(user_context)
        user_context.append({"role": "assistant", "content": gpt_response})
    except Exception as e:
        logger.error(f"Error generating response from OpenAI: {e}")
        return {"status": "error", "message": "Failed to generate a response."}

    # Update the database with the new context
    item = {
        'user_id': user_id,
        'context': json.dumps(user_context),
        'timestamps': json.dumps(valid_timestamps)
    }
    try:
        chat_context_db.put_item(item)
    except Exception as e:
        logger.error(f"Error updating context in database: {e}")
        return {"status": "error", "message": "Internal Server Error"}

    return {"status": "success", "message": gpt_response}


def lambda_handler(event, context):
    """
    AWS Lambda function handler to process incoming requests.
    """
    user_message = event['body']['text']
    user_id = event['body']['userId']

    response = process_request(user_message, user_id)
    if response["status"] == "success":
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': response["message"]
        }
    else:
        return {
            'statusCode': 500,
            'body': response["message"]
        }
