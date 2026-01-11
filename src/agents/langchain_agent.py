import pandas as pd
import os

# --- SAFE IMPORTS FOR LATEST LANGCHAIN ---
try:
    from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
    from langchain_openai import ChatOpenAI  # <--- FIXED IMPORT HERE
    from langchain.agents.agent_types import AgentType
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

class DataAgent:
    def __init__(self, df, api_key=None):
        self.df = df
        self.api_key = api_key

    def ask_agent(self, question):
        """
        Uses LangChain if Key exists & Library is installed.
        Otherwise uses Rule-Based Logic (Fallback).
        """
        # 1. TRY AI MODE
        if self.api_key and LANGCHAIN_AVAILABLE:
            try:
                # Initialize the Chat Model
                llm = ChatOpenAI(
                    temperature=0, 
                    model="gpt-3.5-turbo", 
                    openai_api_key=self.api_key
                )
                
                # Create the Pandas Agent
                agent = create_pandas_dataframe_agent(
                    llm, 
                    self.df, 
                    verbose=True,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    allow_dangerous_code=True
                )
                return agent.run(question)
            except Exception as e:
                return f"⚠️ AI Error: {str(e)}. Switching to Offline Mode."
        
        # 2. OFFLINE FALLBACK (Rule Based - No API Key needed)
        q = question.lower()
        
        # Logic for "Highest"
        if "highest" in q and ("activity" in q or "update" in q):
            # Sort by total activity descending
            top = self.df.sort_values('total_activity', ascending=False).iloc[0]
            return f"Based on the offline data, **{top['district']} ({top['state']})** has the highest recorded activity with **{int(top['total_activity'])}** updates."
        
        # Logic for "Average"
        elif "average" in q:
            avg = self.df['total_activity'].mean()
            return f"The average update volume across all districts is **{avg:.0f}** per day."
            
        # Logic for "Total"
        elif "total" in q:
            total = self.df['total_activity'].sum()
            return f"The total number of Aadhaar updates in this dataset is **{int(total):,}**."
            
        # Default response
        else:
            return "🔴 **Offline Mode Active:** I can answer basic questions like 'Which district has the highest activity?' or 'What is the total volume?'. For complex queries, please enter an OpenAI API Key."