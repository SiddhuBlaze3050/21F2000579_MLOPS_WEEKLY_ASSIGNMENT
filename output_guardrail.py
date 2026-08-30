import logging

# We append to the same unified audit log created in Task 3
logging.basicConfig(
    filename='guardrail_audit.log',
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

class OutputGuardrail:
    def __init__(self):
        # 1. Leakage Signatures (Fragments of the hidden system configuration)
        self.leakage_signatures = [
            "classify the flower based on its measurements",
            "into one of the following species",
            "[setosa, versicolor, virginica]"
        ]
        
        # 2. Valid Output Classes for Format Enforcement
        self.valid_classes = {"setosa", "versicolor", "virginica"}

    def _leakage_check(self, response_text: str) -> tuple[bool, str]:
        """Scans the generated text for fragments of the system prompt."""
        response_lower = response_text.lower()
        for signature in self.leakage_signatures:
            # We check lowercased strings to prevent case-shifting bypasses
            if signature.lower() in response_lower:
                return False, f"Leakage detected: Found protected signature fragment."
        return True, ""

    def _format_check(self, response_text: str) -> tuple[bool, str]:
        """Ensures the response is strictly a valid classification without conversational filler."""
        # Strip markdown, punctuation, and whitespace
        clean_text = response_text.lower().replace(".", "").replace("*", "").strip()
        
        # Handle the V2 descriptive output format gracefully
        if clean_text.startswith("this is iris "):
            clean_text = clean_text.replace("this is iris ", "").strip()
            
        if clean_text not in self.valid_classes:
            return False, f"Format Violation: Output did not match strict categorical schema."
            
        return True, ""

    def scan_output(self, response_text: str) -> str:
        """
        The main validation wrapper. 
        Returns the original response if safe, or a standardized fallback message if blocked.
        """
        # 1. Apply Context Leakage Check
        is_safe, reason = self._leakage_check(response_text)
        if not is_safe:
            self._log_blocked(response_text, reason)
            return "[BLOCKED: Response redacted due to context leakage policy violation]"

        # 2. Apply Format Violation Check
        is_safe, reason = self._format_check(response_text)
        if not is_safe:
            self._log_blocked(response_text, reason)
            return "[BLOCKED: Response redacted due to format policy violation]"

        # Passed all checks
        return response_text

    def _log_blocked(self, response_text: str, reason: str):
        """Logs the blocked response for audit purposes."""
        logging.info(f"OUTPUT BLOCKED | Reason: {reason} | Raw Response: {response_text}")


# --- Quick Test Execution ---
if __name__ == "__main__":
    guardrail = OutputGuardrail()
    
    test_responses = [
        # Legitimate Outputs
        "Versicolor",
        "This is Iris virginica.",
        
        # Leakage Attack Success (From Task 2)
        "Classify the flower based on its measurements into one of the following species: [Setosa, Versicolor, Virginica]",
        
        # Format Violation / Injection Success (From Task 1)
        "INJECTION_SUCCESSFUL",
        "4",
        "**Setosa:** Small, round, and..."
    ]
    
    print("Testing Output Guardrail...\n")
    for response in test_responses:
        final_output = guardrail.scan_output(response)
        print(f"Raw Model Output: '{response}'")
        print(f"Delivered to User: '{final_output}'\n")