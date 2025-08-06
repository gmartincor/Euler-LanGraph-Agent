"""
Robust JSON parser for LLM outputs that handles common formatting errors.
"""

import json
import re
from typing import Any, Dict
from langchain_core.output_parsers import BaseOutputParser
from ..core.logging import get_logger

logger = get_logger(__name__)


class RobustJsonOutputParser(BaseOutputParser[Dict[str, Any]]):
    """
    Robust JSON parser that handles common LLM JSON formatting errors.
    
    This parser applies several cleanup strategies to fix malformed JSON
    before attempting to parse, reducing workflow failures.
    """
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse potentially malformed JSON output from LLM."""
        
        # Log the raw output for debugging
        logger.debug(f"Parsing LLM output: {text[:200]}...")
        
        # Step 1: Extract JSON from markdown code blocks if present
        json_text = self._extract_json_from_markdown(text)
        
        # Step 2: Apply common fixes
        fixed_json = self._fix_common_json_errors(json_text)
        
        # Step 3: Try to parse
        try:
            result = json.loads(fixed_json)
            logger.debug("Successfully parsed JSON")
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed even after fixes: {e}")
            logger.warning(f"Fixed JSON was: {fixed_json}")
            
            # Step 4: Fallback to extraction
            return self._extract_fallback_structure(json_text)
    
    def _extract_json_from_markdown(self, text: str) -> str:
        """Extract JSON from markdown code blocks."""
        
        # Pattern to match ```json ... ``` blocks
        json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        match = re.search(json_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        # If no code block, return original text
        return text.strip()
    
    def _fix_common_json_errors(self, json_text: str) -> str:
        """Apply common fixes to malformed JSON."""
        
        # Fix 1: Remove trailing commas before closing brackets/braces
        json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
        
        # Fix 2: Fix malformed array endings like "Farr],"
        json_text = re.sub(r'\s+\w+\]', ']', json_text)
        
        # Fix 3: Fix incomplete strings at end of arrays
        json_text = re.sub(r'"\s+[^"]*\]', '"]', json_text)
        
        # Fix 4: Ensure proper string quoting
        json_text = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_text)
        
        # Fix 5: Fix incomplete objects/arrays
        open_braces = json_text.count('{')
        close_braces = json_text.count('}')
        if open_braces > close_braces:
            json_text += '}' * (open_braces - close_braces)
        
        open_brackets = json_text.count('[')
        close_brackets = json_text.count(']')
        if open_brackets > close_brackets:
            json_text += ']' * (open_brackets - close_brackets)
        
        return json_text
    
    def _extract_fallback_structure(self, text: str) -> Dict[str, Any]:
        """Extract structure when JSON parsing completely fails."""
        
        logger.warning("Using fallback extraction for malformed JSON")
        
        result = {
            "approach": "",
            "steps": [],
            "tools_needed": [],
            "confidence": 0.5
        }
        
        # Extract approach
        approach_match = re.search(r'"approach":\s*"([^"]*)"', text, re.DOTALL)
        if approach_match:
            result["approach"] = approach_match.group(1)
        
        # Extract tools_needed array
        tools_match = re.search(r'"tools_needed":\s*\[(.*?)\]', text, re.DOTALL)
        if tools_match:
            tools_str = tools_match.group(1)
            # Extract quoted strings
            tools = re.findall(r'"([^"]*)"', tools_str)
            result["tools_needed"] = tools
        
        # Extract steps array
        steps_match = re.search(r'"steps":\s*\[(.*?)\]', text, re.DOTALL)
        if steps_match:
            steps_str = steps_match.group(1)
            # Extract quoted strings
            steps = re.findall(r'"([^"]*)"', steps_str)
            result["steps"] = steps
        
        # Extract confidence
        conf_match = re.search(r'"confidence":\s*([0-9.]+)', text)
        if conf_match:
            try:
                result["confidence"] = float(conf_match.group(1))
            except ValueError:
                pass
        
        # Special handling for visualization keywords
        if any(keyword in text.lower() for keyword in ["show", "plot", "visualize", "area under", "graph"]):
            if "plot_generator" not in result["tools_needed"]:
                result["tools_needed"].append("plot_generator")
                logger.info("Added plot_generator based on visualization keywords")
        
        # Always include integral_calculator for integral problems
        if "integral" in text.lower() and "integral_calculator" not in result["tools_needed"]:
            result["tools_needed"].append("integral_calculator")
            logger.info("Added integral_calculator based on integral keywords")
        
        logger.info(f"Fallback extraction result: {result}")
        return result

    def get_format_instructions(self) -> str:
        """Return format instructions for the prompt."""
        return """
Return a valid JSON object with these exact fields:
{
    "approach": "string describing the mathematical approach",
    "steps": ["step1", "step2", "step3"],
    "tools_needed": ["tool1", "tool2"],
    "confidence": 0.9
}

IMPORTANT: 
- Use proper JSON syntax with double quotes
- End arrays and objects properly
- For visualization requests, always include "plot_generator"
- For integrals/derivatives, always include "integral_calculator"
"""
