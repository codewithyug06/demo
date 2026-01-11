from transformers import pipeline

class PolicyGenerator:
    def __init__(self):
        # We use DistilGPT2 because it's lightweight and runs on CPU quickly
        # For a hackathon, this proves you know GenAI without needing a massive GPU
        self.generator = pipeline('text-generation', model='distilgpt2')

    def generate_memo(self, hotspot_district, saturation_val, anomaly_count):
        """
        Constructs a prompt and uses the LLM to write a formal government memo.
        """
        prompt = f"""
        GOVERNMENT OF INDIA - OFFICIAL MEMO
        Subject: Critical Aadhaar Update Activity in {hotspot_district}
        
        Data Analysis Report:
        1. Saturation Index: {saturation_val:.2f}% (High Migration Detected)
        2. Security Anomalies: {anomaly_count} flagged transactions.
        
        Strategic Recommendation:
        Based on the data above, the administration requires immediate action to"""

        # Generate text (max_length limits output size)
        response = self.generator(prompt, max_length=150, num_return_sequences=1, truncation=True)
        return response[0]['generated_text']