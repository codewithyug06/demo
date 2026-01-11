from rapidfuzz import fuzz

class IdentityReconciler:
    def __init__(self, threshold=85):
        self.threshold = threshold

    def match_identity(self, aadhaar_name, beneficiary_name):
        """
        Uses Token Sort Ratio to handle name swapping (e.g. "Kumar Amit").
        """
        score = fuzz.token_sort_ratio(aadhaar_name.lower(), beneficiary_name.lower())
        
        status = "REJECT"
        if score >= 90:
            status = "AUTO_APPROVE"
        elif score >= self.threshold:
            status = "MANUAL_REVIEW"
            
        return {
            "aadhaar_name": aadhaar_name,
            "bank_name": beneficiary_name,
            "similarity_score": score,
            "status": status
        }