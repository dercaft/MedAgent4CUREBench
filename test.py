from langchain_openai import ChatOpenAI
import os
import dotenv
dotenv.load_dotenv()

reasoning = {
    "effort": "medium",  # 'low', 'medium', or 'high'
    "summary": "auto",  # 'detailed', 'auto', or None
}
model = ChatOpenAI(model = "moonshotai/Kimi-K2-Instruct", temperature = 0.1,
                    base_url = os.getenv("OPENAI_BASE_URL"),api_key = os.getenv("OPENAI_API_KEY"))
response = model.invoke("What is the capital of France?")
print(response)