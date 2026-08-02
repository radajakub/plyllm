from plybench.utils.enums import ExtendedEnum


class Provider(str, ExtendedEnum):
    OPENAI = "openai"
    GEMINI = "gemini"
    GROK = "grok"
    METACENTRUM = "metacentrum"
    HUGGINGFACE = "huggingface"
