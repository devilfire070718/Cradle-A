from cradle.provider.llm.openai import OpenAIProvider
from cradle.provider.llm.restful_claude import RestfulClaudeProvider
from cradle.utils import Singleton
from cradle.provider.llm.kimi import KimiProvider

class LLMFactory(metaclass=Singleton):

    def __init__(self):
        self._builders = {}


    def create(self, llm_provider_config_path, embed_provider_config_path, **kwargs):

        llm_provider = None
        embed_provider = None

        key = llm_provider_config_path

        if "openai" in key:
            llm_provider = OpenAIProvider()
            llm_provider.init_provider(llm_provider_config_path)
            embed_provider = llm_provider
        elif "claude" in key:
            llm_provider = RestfulClaudeProvider()
            llm_provider.init_provider(llm_provider_config_path)
            #logger.warn(f"Claude do not support embedding, use OpenAI instead.")
            embed_provider = OpenAIProvider()
            embed_provider.init_provider(embed_provider_config_path)
        elif "kimi" in key:  # 新增 Kimi 支持
            llm_provider = KimiProvider()  
            llm_provider.init_provider(llm_provider_config_path)
            embed_provider = llm_provider
        elif "zhipu" in key:
            llm_provider = ZhipuProvider()
            llm_provider.init_provider(llm_provider_config_path)
            embed_provider = llm_provider  # 使用智谱作为 embedding 提供者

        if not llm_provider or not embed_provider:
            raise ValueError(key)

        return llm_provider, embed_provider
