import boto3
import json
import logging
from botocore.exceptions import ClientError

# Set up basic logging for better feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMDataValidator:
    """
    A class to orchestrate data extraction using a 'Worker' LLM and 
    validation of that extraction using a 'Judge' LLM via AWS Bedrock.
    """
    def __init__(self, aws_region: str = "us-east-1"):
        """
        Initializes the validator with a Bedrock runtime client.

        Args:
            aws_region (str): The AWS region where Bedrock is enabled.
        """
        try:
            self.bedrock_runtime = boto3.client(
                service_name='bedrock-runtime', 
                region_name=aws_region
            )
            logging.info(f"Successfully connected to Bedrock in {aws_region}")
        except ClientError as e:
            logging.error(f"Could not connect to Bedrock. Please check your AWS credentials and region. Error: {e}")
            raise

    def _invoke_llm(self, model_id: str, system_prompt: str, user_prompt: str) -> str:
        """
        Private method to invoke a Bedrock LLM with a given prompt.
        This example uses the Anthropic Claude 3 message format.

        Args:
            model_id (str): The ID of the Bedrock model to use.
            system_prompt (str): The system-level instruction for the model.
            user_prompt (str): The user-provided prompt.

        Returns:
            str: The text content from the model's response.
        """
        try:
            # Anthropic Claude 3 Message API format
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2048,
                "temperature": 0.0, # Set to 0 for deterministic, factual responses
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            })

            response = self.bedrock_runtime.invoke_model(
                body=body, 
                modelId=model_id,
                contentType='application/json',
                accept='application/json'
            )
            response_body = json.loads(response.get('body').read())
            
            # The actual text is in the first content block
            return response_body['content'][0]['text']

        except ClientError as e:
            logging.error(f"Error invoking model {model_id}: {e}")
            return f"ERROR: Could not get a response from model {model_id}."
        except Exception as e:
            logging.error(f"An unexpected error occurred during model invocation: {e}")
            return f"ERROR: An unexpected error occurred."


    def extract_data(self, context: str, field_to_extract: str, worker_model_id: str) -> str:
        """
        Uses the 'Worker' LLM to extract a specific piece of data.

        Args:
            context (str): The source text from the document.
            field_to_extract (str): The name of the field to extract (e.g., "Invoice Number").
            worker_model_id (str): The Bedrock model ID for the worker.

        Returns:
            str: The extracted data.
        """
        logging.info(f"WORKER: Attempting to extract '{field_to_extract}' using {worker_model_id}...")
        
        system_prompt = "You are an expert data extraction assistant. Your task is to accurately extract information from the provided text. Respond with only the extracted value and nothing else."
        
        user_prompt = f"""
<document_context>
{context}
</document_context>

Based on the document context above, what is the value for: {field_to_extract}?
"""
        extracted_value = self._invoke_llm(worker_model_id, system_prompt, user_prompt)
        logging.info(f"WORKER: Extracted value: '{extracted_value}'")
        return extracted_value

    def judge_extraction(self, context: str, field_to_extract: str, extracted_data: str, judge_model_id: str) -> dict:
        """
        Uses the 'Judge' LLM to validate the extracted data.

        Args:
            context (str): The original source text.
            field_to_extract (str): The name of the field that was extracted.
            extracted_data (str): The data provided by the 'Worker' LLM.
            judge_model_id (str): The Bedrock model ID for the judge.

        Returns:
            dict: A dictionary containing the judgment, reasoning, and corrected answer.
        """
        logging.info(f"JUDGE: Evaluating extraction for '{field_to_extract}' using {judge_model_id}...")
        
        system_prompt = """You are an impartial and meticulous AI judge. Your task is to evaluate an answer provided by another AI based on a given document context.
You must determine if the provided answer correctly and completely answers the extraction question, based *only* on the information within the document context.
Respond in a single, valid JSON object following the schema provided. Do not add any text before or after the JSON object."""

        user_prompt = f"""
Please evaluate the provided answer and respond in a single JSON object with the following schema:
{{
  "judgment": "A string that must be one of: 'CORRECT', 'INCORRECT', or 'PARTIALLY_CORRECT'.",
  "reasoning": "A brief, one-sentence explanation for your judgment. Explain *why* the answer is correct or incorrect by referencing the context.",
  "corrected_answer": "If the judgment is 'INCORRECT' or 'PARTIALLY_CORRECT', provide the correct and complete answer from the context. Otherwise, this should be null."
}}

Here is the information to evaluate:

<document_context>
{context}
</document_context>

<extraction_question>
What is the {field_to_extract}?
</extraction_question>

<provided_answer>
{extracted_data}
</provided_answer>
"""
        
        judgment_str = self._invoke_llm(judge_model_id, system_prompt, user_prompt)

        try:
            # The LLM should return a clean JSON string, but we clean it up just in case
            clean_json_str = judgment_str.strip().replace("```json", "").replace("```", "")
            judgment_json = json.loads(clean_json_str)
            logging.info(f"JUDGE: Received judgment: {judgment_json}")
            return judgment_json
        except json.JSONDecodeError:
            logging.error(f"JUDGE: Failed to parse JSON response: {judgment_str}")
            return {
                "judgment": "VALIDATION_FAILED",
                "reasoning": "The judge model did not return a valid JSON object.",
                "corrected_answer": None
            }

    def run_extraction_and_judgment(self, context: str, field_to_extract: str, worker_model_id: str, judge_model_id: str) -> dict:
        """
        Runs the full pipeline: extract data with the worker and validate with the judge.

        Returns:
            dict: A consolidated dictionary with the original extraction and its judgment.
        """
        extracted_data = self.extract_data(context, field_to_extract, worker_model_id)
        
        # If worker failed, no need to judge
        if "ERROR:" in extracted_data:
             return {
                "field": field_to_extract,
                "worker_extraction": extracted_data,
                "judgment_result": {
                    "judgment": "PIPELINE_ERROR",
                    "reasoning": "Worker LLM failed to produce an output.",
                    "corrected_answer": None
                }
            }
            
        judgment = self.judge_extraction(context, field_to_extract, extracted_data, judge_model_id)
        
        return {
            "field": field_to_extract,
            "worker_extraction": extracted_data,
            "judgment_result": judgment
        }

### How to Use the Class

if __name__ == '__main__':
    # --- Configuration ---
    # Use a cheaper, faster model for the bulk work
    WORKER_MODEL = "anthropic.claude-3-haiku-20240307-v1:0" 
    # Use a more powerful, reasoning-focused model for judging
    JUDGE_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0" 
    AWS_REGION = "us-east-1" # Change if you use a different region

    # --- Sample Data (Simulating text extracted from a PDF) ---
    pdf_text_context = """
    Invoice
    
    Billed to:
    John Doe
    123 Main St
    Anytown, USA 12345
    
    Invoice Number: INV-2024-001
    Date of Issue: May 15, 2024
    Due Date: June 14, 2024
    
    Description        |  Quantity  |  Unit Price  |  Amount
    ----------------------------------------------------------
    Cloud Consulting   |  10 hours  |  $150.00     |  $1500.00
    Data Migration     |  1 project |  $2500.00    |  $2500.00
    
    Subtotal: $4000.00
    Tax (8%): $320.00
    
    Total Amount Due: $4320.00
    
    Payment Instructions: Please pay via bank transfer to Account #987654321.
    """

    fields_to_validate = [
        "Invoice Number",
        "Total Amount Due",
        "Billed to",
        "Shipping Company" # This field does not exist, to test the judge's response
    ]
    
    # --- Execution ---
    validator = LLMDataValidator(aws_region=AWS_REGION)
    
    print("-" * 50)
    print("Starting Data Extraction and Judgment Pipeline")
    print("-" * 50)

    for field in fields_to_validate:
        result = validator.run_extraction_and_judgment(
            context=pdf_text_context,
            field_to_extract=field,
            worker_model_id=WORKER_MODEL,
            judge_model_id=JUDGE_MODEL
        )
        
        print(f"\n[ VALIDATING FIELD: '{result['field']}' ]")
        print(f"  -> Worker's Answer: '{result['worker_extraction']}'")
        
        judgment_res = result['judgment_result']
        print(f"  -> Judge's Verdict: {judgment_res.get('judgment')}")
        print(f"  -> Judge's Reasoning: {judgment_res.get('reasoning')}")
        
        if judgment_res.get('judgment') in ["INCORRECT", "PARTIALLY_CORRECT"]:
            print(f"  -> Judge's Correction: {judgment_res.get('corrected_answer')}")

    print("\n" + "-" * 50)
    print("Pipeline finished.")
    print("-" * 50)