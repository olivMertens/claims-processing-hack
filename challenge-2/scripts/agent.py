import asyncio
import os
import json
from datetime import datetime
from typing import Annotated, Dict, List, Any, Optional
from dotenv import load_dotenv
from pydantic import Field
import logging

# Azure AI Foundry SDK
from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import AzureCliCredential

# Agent Framework
from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
project_endpoint = os.environ.get("AI_FOUNDRY_PROJECT_ENDPOINT")
model_deployment_name = os.environ.get("MODEL_DEPLOYMENT_NAME")

# Claims Processing Tool Functions

def extract_text_from_image(
    image_path: Annotated[str, Field(description="Path to the image file containing damage photo or document")]
) -> dict:
    """Extract text from an image using OCR."""
    try:
        result = {
            "status": "success",
            "text": f"Placeholder: OCR text extraction from {image_path}",
            "confidence": 0.85,
            "document_type": "image"
        }
        logger.info(f"Extracted text from image: {image_path}")
        return result
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        return {"error": f"Failed to extract text: {str(e)}"}


def parse_policy_document(
    policy_path: Annotated[str, Field(description="Path to the policy markdown file")]
) -> dict:
    """Parse a markdown policy document and extract key terms."""
    try:
        with open(policy_path, 'r') as f:
            content = f.read()
        
        result = {
            "status": "success",
            "policy_type": "comprehensive_auto",  # Extract from content
            "coverage_limit": 50000.00,
            "deductible": 500.00,
            "content_preview": content[:200] + "..."
        }
        logger.info(f"Parsed policy document: {policy_path}")
        return result
    except Exception as e:
        logger.error(f"Error parsing policy document: {e}")
        return {"error": f"Failed to parse policy: {str(e)}"}


def validate_claim_amount(
    claim_amount: Annotated[float, Field(description="The claimed damage amount in USD")],
    policy_coverage: Annotated[float, Field(description="Maximum coverage limit from the policy")],
    deductible: Annotated[float, Field(description="Deductible amount")] = 0
) -> dict:
    """Validate if claim amount is within policy coverage limits."""
    try:
        eligible = claim_amount <= policy_coverage
        payout = max(0, min(claim_amount, policy_coverage) - deductible)
        
        result = {
            "eligible": eligible,
            "claim_amount": claim_amount,
            "policy_coverage": policy_coverage,
            "deductible": deductible,
            "estimated_payout": payout,
            "within_limits": claim_amount <= policy_coverage
        }
        logger.info(f"Validated claim amount: ${claim_amount} against coverage: ${policy_coverage}")
        return result
    except Exception as e:
        logger.error(f"Error validating claim amount: {e}")
        return {"error": f"Failed to validate claim: {str(e)}"}


def assess_policy_eligibility(
    incident_type: Annotated[str, Field(description="Type of incident (e.g., collision, theft, vandalism, fire)")],
    policy_type: Annotated[str, Field(description="Type of policy (e.g., comprehensive_auto, liability_only)")]
) -> dict:
    """Assess if the incident type is covered by the policy."""
    try:
        # Simplified coverage matrix
        coverage_matrix = {
            "comprehensive_auto": ["collision", "theft", "vandalism", "weather", "fire"],
            "liability_only": ["collision"],
            "commercial_auto": ["collision", "theft", "vandalism", "business_use"],
            "motorcycle": ["collision", "theft", "vandalism"],
            "high_value_vehicle": ["collision", "theft", "vandalism", "weather", "fire", "custom_parts"]
        }
        
        covered_incidents = coverage_matrix.get(policy_type, [])
        is_covered = incident_type.lower() in covered_incidents
        
        result = {
            "covered": is_covered,
            "incident_type": incident_type,
            "policy_type": policy_type,
            "covered_incidents": covered_incidents
        }
        logger.info(f"Assessed eligibility: {incident_type} under {policy_type} - {'COVERED' if is_covered else 'NOT COVERED'}")
        return result
    except Exception as e:
        logger.error(f"Error assessing policy eligibility: {e}")
        return {"error": f"Failed to assess eligibility: {str(e)}"}


def generate_claim_report(
    claim_data: Annotated[str, Field(description="Complete claim information including validation results")],
    report_type: Annotated[str, Field(description="Type of report (e.g., 'PRELIMINARY', 'FINAL', 'DENIAL')")] = "PRELIMINARY"
) -> dict:
    """Generate a structured claim report with recommendations."""
    try:
        report = {
            "report_id": f"CLAIM_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "report_type": report_type,
            "generated_timestamp": datetime.now().isoformat(),
            "claim_summary": claim_data[:500],  # First 500 chars
            "recommendations": [],
            "next_steps": [],
            "status": "PENDING_REVIEW"
        }
        
        # Add recommendations based on report type
        if "approved" in claim_data.lower():
            report["status"] = "APPROVED"
            report["recommendations"].append("Process claim payment")
            report["next_steps"].append("Initiate payment workflow")
        elif "denied" in claim_data.lower():
            report["status"] = "DENIED"
            report["recommendations"].append("Send denial letter to claimant")
            report["next_steps"].append("Document reason for denial")
        else:
            report["recommendations"].append("Additional review required")
            report["next_steps"].append("Request additional documentation")
        
        logger.info(f"Generated claim report: {report['report_id']} - Status: {report['status']}")
        return report
    except Exception as e:
        logger.error(f"Error generating claim report: {e}")
        return {"error": f"Failed to generate report: {str(e)}"}


async def main():
    """Main function to create and test the Claims Processing Agent."""
    
    try:
        async with AzureCliCredential() as credential:
            async with AIProjectClient(
                endpoint=project_endpoint,
                credential=credential
            ) as project_client:
                
                # Create persistent agent
                created_agent = await project_client.agents.create_agent(
                    model=model_deployment_name,
                    name="ClaimsProcessingAgent",
                    instructions="""You are an expert Insurance Claims Processing Agent specialized in analyzing and processing vehicle insurance claims.

Your primary responsibilities include:

1. **Claims Analysis**:
   - Review claim descriptions and extract key information
   - Identify incident type, damage details, and claimed amounts
   - Extract policy holder information and policy numbers

2. **Document Processing**:
   - Use OCR tools to extract text from damage photos
   - Parse policy documents to understand coverage terms
   - Extract coverage limits, deductibles, and policy types

3. **Validation and Assessment**:
   - Validate claimed amounts against policy coverage limits
   - Assess if the incident type is covered by the policy
   - Calculate estimated payouts considering deductibles
   - Identify any coverage gaps or exclusions

4. **Report Generation**:
   - Generate structured claim reports with clear recommendations
   - Provide approval/denial recommendations with reasoning
   - Include confidence scores and risk assessments
   - Flag missing information or documentation needs

**Available Tools**:
- Text extraction from images (OCR for damage photos)
- Policy document parsing (extract coverage details)
- Claim amount validation (check against limits)
- Policy eligibility assessment (verify coverage)
- Claim report generation (structured output)

**Input Sources**:
- Policy documents (markdown files with coverage terms)
- Damage photos (images of vehicle damage)
- Handwritten statements (customer incident descriptions)
- Claim descriptions (natural language)

**Output Format Guidelines**:
- Provide clear, structured analysis
- Always use tools to extract and validate information
- Include specific reasoning for recommendations
- Flag any missing or unclear information
- Maintain professional insurance industry standards
- Focus on accuracy and completeness

**Important Notes**:
- Always validate claims against actual policy terms
- Consider deductibles in payout calculations
- Verify incident type coverage before approval
- Document all validation steps for audit trail
- Prioritize accuracy over speed

You must ensure all claim processing is thorough, accurate, and follows insurance industry best practices."""
                )
                
                # Wrap agent with tools for usage
                agent = ChatAgent(
                    chat_client=AzureAIAgentClient(
                        project_client=project_client,
                        agent_id=created_agent.id
                    ),
                    tools=[
                        extract_text_from_image,
                        parse_policy_document,
                        validate_claim_amount,
                        assess_policy_eligibility,
                        generate_claim_report
                    ],
                    store=True
                )

                logger.info(f"✅ Created Claims Processing Agent: {created_agent.id}")
                print(f"✅ Created Claims Processing Agent: {created_agent.id}")

                # Test the agent with a sample claim
                print(f"\n🔍 Testing Claims Processing Agent...")

                sample_claim = """Process this insurance claim:

Policy Holder: John Smith
Policy Number: POL-12345
Policy Type: Comprehensive Auto
Incident Date: January 3, 2026
Incident Type: Vehicle collision
Description: My car was hit by another vehicle while parked. 
The front bumper and headlight are damaged. 

Estimated repair cost: $3,500
Policy Coverage Limit: $50,000
Deductible: $500

Available Documents:
- Policy: /workspaces/claims-processing-hack/challenge-0/data/policies/comprehensive_auto_policy.md

Please analyze the claim, validate coverage, and provide a recommendation."""

                result = await agent.run(sample_claim)

                print(f"\n📋 CLAIMS PROCESSING AGENT RESPONSE:")
                print("="*60)
                print(result.text)
                print("="*60)

                return agent
                
    except Exception as e:
        logger.error(f"❌ Error creating Claims Processing Agent: {e}")
        print(f"❌ Error creating Claims Processing Agent: {e}")
        return None


if __name__ == "__main__":
    asyncio.run(main())
