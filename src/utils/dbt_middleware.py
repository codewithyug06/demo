from rapidfuzz import fuzz

class DBTMiddleware:
    @staticmethod
    def verify_beneficiary(aadhaar_name, bank_record_name, threshold=88):
        """
        Returns MATCH/NO_MATCH decision using Token Sort Ratio.
        """
        score = fuzz.token_sort_ratio(aadhaar_name.lower(), bank_record_name.lower())
        
        return {
            "aadhaar_entry": aadhaar_name,
            "bank_entry": bank_record_name,
            "match_score": score,
            "decision": "APPROVE" if score >= threshold else "MANUAL_VERIFY"
        }