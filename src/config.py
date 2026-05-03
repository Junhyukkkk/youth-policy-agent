from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    google_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str = "youth-policy-index"
    youth_policy_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"


settings = Settings()
