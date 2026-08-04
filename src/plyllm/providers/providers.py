from plybench.utils.enums import ExtendedEnum


class Provider(str, ExtendedEnum):
    OPENAI = "openai"
    GEMINI = "gemini"
    GROK = "grok"
    CLAUDE = "claude"
    METACENTRUM = "metacentrum"
    HUGGINGFACE = "huggingface"
