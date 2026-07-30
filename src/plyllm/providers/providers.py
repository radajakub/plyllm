from plybench.utils.enums import ExtendedEnum


class Provider(str, ExtendedEnum):
    OPENAI = "openai"
    GEMINI = "gemini"
    METACENTRUM = "metacentrum"
    HUGGINGFACE = "huggingface"
