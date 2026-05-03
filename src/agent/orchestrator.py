from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings

llm = ChatGoogleGenerativeAI(
    model=settings.gemini_model,
    google_api_key=settings.google_api_key,
    temperature=0.0,
)
