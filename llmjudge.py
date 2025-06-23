import boto3
import json
import os
import fitz  # PyMuPDF
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper function to create a dummy PDF for demonstration ---
def create_dummy_pdf(filename="invoice_sample.pdf"):
    """Creates a simple dummy invoice PDF using reportlab."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    if os.path.exists(filename):
        logging.info(f"'{filename}' already exists. Skipping creation.")
        return

    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    c.drawString(72, height - 72, "INVOICE")
    c.drawString(72, height - 100, "Invoice Number: INV-2024-001")
    c.drawString(72, height - 120, "Client Name: Acme Corporation")
    c.drawString(72, height - 140, "Date: July 26, 2024")
    c.drawString(72, height - 180, "Description: Consulting Services")
    c.drawString(72, height - 200, "Total Amount: $1,500.75")
    c.drawString(72, height - 220, "Due Date: August 25, 2024")
    c.save()
    logging.info(f"Created dummy PDF: '{filename}'")


class BedrockEnsembleExtractor:
    """
    An LLM-as-a-judge system using an ensemble of AWS Bedrock models for data extraction.
    """

    def __init__(self, model_ids: list[str], aws_region: str = "us-east-1"):
        """
        Initializes the extractor with a list of Bedrock model IDs.

        Args:
            model_ids (list[str]): A list of model IDs to use in the ensemble.
                                   e.g., ['anthropic.claude-3-sonnet-20240229-v1:0', 
                                          'anthropic.claude-3-haiku-20240307-v1:0']
            aws_region (str): The AWS region where Bedrock is hosted.
        """
        if not model_ids:
            raise ValueError("model_ids list cannot be empty.")
            
        self.model_ids = model_ids
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=aws_region
        )
        logging.info(f"Initialized extractor with models: {self.model_ids} in region {aws_region}")

    def _invoke_model(self, model_id: str, prompt: str) -> dict | None:
        """
        Invokes a single Bedrock model and returns its structured response.
        Handles different request body formats for different model providers.
        """
        try:
            logging.info(f"Invoking model: {model_id}")
            
            # Use the Anthropic Messages API format for Claude models
            if "anthropic" in model_id:
                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2048,
                    "temperature": 0.0, # Set to 0 for deterministic extraction
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}]
                        }
                    ]
                }
            # Add other model provider formats here if needed
            # elif "meta" in model_id: ...
            else:
                logging.error(f"Model provider for '{model_id}' is not supported yet.")
                return None

            response = self.bedrock_runtime.invoke_model(
                body=json.dumps(request_body),
                modelId=model_id,
                contentType='application/json',
                accept='application/json'
            )
            response_body = json.loads(response['body'].read())
            
            # Extract text based on provider
            if "anthropic" in model_id:
                raw_text = response_body['content'][0]['text']
                return self._parse_json_from_response(model_id, raw_text)

        except Exception as e:
            logging.error(f"Error invoking model {model_id}: {e}")
            return None
    
    def _parse_json_from_response(self, model_id: str, text: str) -> dict | None:
        """
        Safely parses a JSON object from the LLM's raw text response.
        Handles cases where the LLM might add ```json ... ``` markers.
        """
        try:
            # Find the start of the JSON block
            json_start = text.find('{')
            # Find the end of the JSON block
            json_end = text.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = text[json_start:json_end]
                return json.loads(json_str)
            else:
                logging.warning(f"No valid JSON object found in response from {model_id}.")
                return None
        except json.JSONDecodeError:
            logging.warning(f"Failed to decode JSON from model {model_id}. Response: {text[:200]}...")
            return None

    def _get_consensus(self, results: list[dict]) -> dict:
        """
        The "Judge" function. Analyzes results from all models to find a consensus.
        """
        valid_extractions = [res for res in results if res is not None]

        if not valid_extractions:
            return {
                "status": "FAILURE",
                "reason": "No valid JSON data was extracted by any model.",
                "consensus_answer": None,
                "confidence": 0.0,
                "is_unanimous": False,
                "all_responses": results
            }

        # To make dictionaries hashable for the Counter, we convert them to a tuple of sorted items.
        hashable_extractions = [tuple(sorted(d.items())) for d in valid_extractions]
        
        # Count the frequency of each unique extraction
        extraction_counts = Counter(hashable_extractions)
        
        # Find the most common extraction
        most_common_tuple, agreement_count = extraction_counts.most_common(1)[0]
        
        # Convert the winning tuple back to a dictionary
        consensus_answer = dict(most_common_tuple)

        total_valid_responses = len(valid_extractions)
        confidence = agreement_count / total_valid_responses
        is_unanimous = agreement_count == total_valid_responses
        
        return {
            "status": "SUCCESS",
            "consensus_answer": consensus_answer,
            "confidence": confidence,
            "is_unanimous": is_unanimous,
            "agreement_count": agreement_count,
            "total_valid_responses": total_valid_responses,
            "all_responses": results
        }


    def extract(self, pdf_path: str, fields_to_extract: list[str]) -> dict:
        """
        Main method to extract data from a PDF using the model ensemble.

        Args:
            pdf_path (str): The path to the PDF file.
            fields_to_extract (list[str]): A list of field names you want to extract.

        Returns:
            dict: A dictionary containing the consensus result and detailed analysis.
        """
        # 1. Read text from PDF
        try:
            with fitz.open(pdf_path) as doc:
                pdf_text = "".join(page.get_text() for page in doc)
            if not pdf_text.strip():
                raise ValueError("PDF text is empty.")
        except Exception as e:
            return {"status": "FAILURE", "reason": f"Failed to read PDF: {e}"}

        # 2. Create the prompt
        prompt_template = f"""
You are an expert data extraction AI. Your task is to extract specific fields from the provided document text.

The document text is:
---DOCUMENT TEXT---
{pdf_text}
---END DOCUMENT TEXT---

Please extract the following fields: {', '.join(fields_to_extract)}.

Respond ONLY with a single, valid JSON object containing the extracted fields. Do not include any explanations, apologies, or markdown formatting like ```json.
If a field is not found, its value should be null.
"""

        # 3. Invoke models in parallel
        all_results = []
        with ThreadPoolExecutor(max_workers=len(self.model_ids)) as executor:
            future_to_model = {executor.submit(self._invoke_model, model_id, prompt_template): model_id for model_id in self.model_ids}
            for future in as_completed(future_to_model):
                model_id = future_to_model[future]
                try:
                    result = future.result()
                    all_results.append({"model_id": model_id, "extraction": result})
                except Exception as e:
                    logging.error(f"Future for model {model_id} generated an exception: {e}")
                    all_results.append({"model_id": model_id, "extraction": None})
        
        # 4. Get consensus from the results
        model_extractions = [res['extraction'] for res in all_results]
        consensus_result = self._get_consensus(model_extractions)
        
        # Add the raw model responses to the final result for full transparency
        consensus_result["raw_model_outputs"] = all_results
        
        return consensus_result

# --- Main execution block ---
if __name__ == "__main__":
    # Create a sample PDF to work with
    pdf_file = "invoice_sample.pdf"
    create_dummy_pdf(pdf_file)

    # Define the models for your ensemble. 
    # Ensure you have access to these models in your AWS account.
    # Using two different models (Sonnet and Haiku) is a good starting point.
    ensemble_models = [
        'anthropic.claude-3-sonnet-20240229-v1:0',
        'anthropic.claude-3-haiku-20240307-v1:0',
        # You could add another model here for a 3-way vote if available
        # 'meta.llama3-8b-instruct-v1:0', # Example if you add Meta Llama 3
    ]
    
    # Define the fields you want to pull from the document
    fields = ["Invoice Number", "Client Name", "Total Amount", "Due Date"]
    
    # Initialize the extractor
    # Make sure to use the region where your Bedrock models are enabled
    extractor = BedrockEnsembleExtractor(model_ids=ensemble_models, aws_region="us-east-1")
    
    # Run the extraction and judging process
    final_result = extractor.extract(pdf_path=pdf_file, fields_to_extract=fields)
    
    # Print the results in a readable format
    print("\n" + "="*50)
    print(" LLM-as-a-Judge Extraction Result")
    print("="*50)
    print(f"Status: {final_result.get('status')}")
    if final_result.get('status') == 'SUCCESS':
        print(f"Confidence: {final_result.get('confidence'):.0%}")
        print(f"Unanimous Agreement: {final_result.get('is_unanimous')}")
        print(f"Agreement Count: {final_result.get('agreement_count')} out of {final_result.get('total_valid_responses')} valid responses")
        print("\n--- Consensus Answer ---")
        print(json.dumps(final_result.get('consensus_answer'), indent=2))
    else:
        print(f"Reason for Failure: {final_result.get('reason')}")
    
    print("\n--- Raw Model Outputs ---")
    print(json.dumps(final_result.get('raw_model_outputs'), indent=2))
    print("="*50)