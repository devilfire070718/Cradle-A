import os  
import json  
import asyncio  
from typing import Any, Dict, List, Tuple, Optional  
import backoff  
import requests  
  
from cradle import constants  
from cradle.provider.base import LLMProvider  
from cradle.config import Config  
from cradle.log import Logger  
from cradle.utils.json_utils import load_json  
from cradle.utils.file_utils import assemble_project_path  
  
config = Config()  
logger = Logger()  
  
PROVIDER_SETTING_KEY_VAR = "key_var"  
PROVIDER_SETTING_COMP_MODEL = "comp_model"  
PROVIDER_SETTING_BASE_URL = "base_url"  
  
class KimiProvider(LLMProvider):  
    """A class that wraps Kimi (Moonshot) model"""  
  
    client: Any = None  
    llm_model: str = ""  
    api_key: str = ""  
    base_url: str = "https://api.moonshot.cn/v1"  
  
    def __init__(self) -> None:  
        """Initialize a class instance"""  
        self.retries = 5

    def embed_query(self, text: str) -> List[float]:
        """Call out to Kimi's embedding endpoint for embedding query text."""
        logger.write(f"# KimiProvider # Calling embedding API for text: {text[:100]}...")

        # 智谱 API 配置
        zhipu_api_key = os.getenv("ZHIPU_API_KEY")
        zhipu_base_url = "https://open.bigmodel.cn/api/paas/v4"

        """使用 Kimi 的 embedding API"""
        headers = {
            "Authorization": f"Bearer {zhipu_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "embedding-3",
            "input": text
        }

        logger.write(f"# KimiProvider # Request URL: {zhipu_api_key}/embeddings")
        logger.write(f"# KimiProvider # Request payload: {payload}")

        try:
            response = requests.post(
                f"{zhipu_base_url}/embeddings",
                headers=headers,
                json=payload,
                timeout=60
            )

            logger.write(f"# KimiProvider # Response status: {response.status_code}")
            logger.write(f"# KimiProvider # Response headers: {dict(response.headers)}")

            if response.status_code != 200:
                logger.error(f"# KimiProvider # Kimi embedding API failed: {response.status_code}")
                logger.error(f"# KimiProvider # Response body: {response.text}")
                raise Exception(f"Embedding API request failed: {response.status_code}")

            response_data = response.json()
            logger.write(
                f"# KimiProvider # Successfully got embedding, dimension: {len(response_data['data'][0]['embedding'])}")
            return response_data["data"][0]["embedding"]

        except requests.exceptions.RequestException as e:
            logger.error(f"# KimiProvider # Network error during embedding request: {e}")
            raise
        except Exception as e:
            logger.error(f"# KimiProvider # Unexpected error during embedding: {e}")
            raise

    def get_embedding_dim(self) -> int:
        return 2048  # 智谱 embedding-3 模型的维度

    def init_provider(self, provider_cfg) -> None:  
        self.provider_cfg = self._parse_config(provider_cfg)  
  
    def _parse_config(self, provider_cfg) -> dict:  
        """Parse the config object"""  
        conf_dict = dict()  
  
        if isinstance(provider_cfg, dict):  
            conf_dict = provider_cfg  
        else:  
            path = assemble_project_path(provider_cfg)  
            conf_dict = load_json(path)  
  
        key_var_name = conf_dict[PROVIDER_SETTING_KEY_VAR]  
        self.api_key = os.getenv(key_var_name)  
          
        if not self.api_key:  
            raise ValueError(f"API key not found in environment variable: {key_var_name}")  
  
        self.llm_model = conf_dict[PROVIDER_SETTING_COMP_MODEL]  
          
        # Optional base URL override  
        if PROVIDER_SETTING_BASE_URL in conf_dict:  
            self.base_url = conf_dict[PROVIDER_SETTING_BASE_URL]  
  
        return conf_dict  
  
    def create_completion(  
        self,  
        messages: List[Dict[str, str]],  
        model: str | None = None,  
        temperature: float = config.temperature,  
        seed: int = config.seed,  
        max_tokens: int = config.max_tokens,  
    ) -> Tuple[str, Dict[str, int]]:  
        """Create a chat completion using the Kimi API"""  
  
        if model is None:  
            model = self.llm_model  
  
        if config.debug_mode:  
            logger.debug(f"Creating chat completion with model {model}, temperature {temperature}, max_tokens {max_tokens}")  
        else:  
            logger.write(f"Requesting {model} completion...")  
  
        @backoff.on_exception(  
            backoff.constant,  
            (requests.exceptions.RequestException, Exception),  
            max_tries=self.retries,  
            interval=10,  
        )  
        def _generate_response_with_retry(  
            messages: List[Dict[str, str]],  
            model: str,  
            temperature: float,  
            max_tokens: int = 512,  
        ) -> Tuple[str, Dict[str, int]]:  
            """Send a request to the Kimi API."""  
              
            headers = {  
                "Authorization": f"Bearer {self.api_key}",  
                "Content-Type": "application/json"  
            }  
  
            # Convert messages format to match Kimi API  
            formatted_messages = self._format_messages(messages)  
  
            payload = {  
                "model": model,  
                "messages": formatted_messages,  
                "temperature": temperature,  
                "max_tokens": max_tokens,  
                "stream": False  
            }  
  
            response = requests.post(  
                f"{self.base_url}/chat/completions",  
                headers=headers,  
                json=payload,  
                timeout=60  
            )  
  
            if response.status_code != 200:  
                logger.error(f"Kimi API request failed with status {response.status_code}: {response.text}")  
                raise Exception(f"API request failed: {response.status_code}")  
  
            response_data = response.json()  
  
            if not response_data.get("choices"):  
                logger.error("Failed to get a response from Kimi. Try again.")  
                raise Exception("No choices in response")  
  
            message = response_data["choices"][0]["message"]["content"]  
  
            info = {  
                "prompt_tokens": response_data.get("usage", {}).get("prompt_tokens", 0),  
                "completion_tokens": response_data.get("usage", {}).get("completion_tokens", 0),  
                "total_tokens": response_data.get("usage", {}).get("total_tokens", 0),  
            }  
  
            logger.write(f'Response received from {model}.')  
  
            return message, info  
  
        return _generate_response_with_retry(  
            messages,  
            model,  
            temperature,  
            max_tokens,  
        )  
  
    async def create_completion_async(  
        self,  
        messages: List[Dict[str, str]],  
        model: str | None = None,  
        temperature: float = config.temperature,  
        seed: int = config.seed,  
        max_tokens: int = config.max_tokens,  
    ) -> Tuple[str, Dict[str, int]]:  
        """Create an async chat completion using the Kimi API"""  
  
        if model is None:  
            model = self.llm_model  
  
        if config.debug_mode:  
            logger.debug(f"Creating async chat completion with model {model}, temperature {temperature}, max_tokens {max_tokens}")  
        else:  
            logger.write(f"Requesting {model} completion...")  
  
        @backoff.on_exception(  
            backoff.constant,  
            (Exception,),  
            max_tries=self.retries,  
            interval=10,  
        )  
        async def _generate_response_with_retry_async(  
            messages: List[Dict[str, str]],  
            model: str,  
            temperature: float,  
            max_tokens: int = 512,  
        ) -> Tuple[str, Dict[str, int]]:  
            """Send an async request to the Kimi API."""  
              
            # Use asyncio.to_thread to run the sync request in a thread  
            return await asyncio.to_thread(  
                self._sync_request,  
                messages,  
                model,  
                temperature,  
                max_tokens  
            )  
  
        return await _generate_response_with_retry_async(  
            messages,  
            model,  
            temperature,  
            max_tokens,  
        )  
  
    def _sync_request(self, messages, model, temperature, max_tokens):  
        """Helper method for sync request"""  
        headers = {  
            "Authorization": f"Bearer {self.api_key}",  
            "Content-Type": "application/json"  
        }  
  
        formatted_messages = self._format_messages(messages)  
  
        payload = {  
            "model": model,  
            "messages": formatted_messages,  
            "temperature": temperature,  
            "max_tokens": max_tokens,  
            "stream": False  
        }  
  
        response = requests.post(  
            f"{self.base_url}/chat/completions",  
            headers=headers,  
            json=payload,  
            timeout=60  
        )  
  
        if response.status_code != 200:  
            raise Exception(f"API request failed: {response.status_code}")  
  
        response_data = response.json()  
        message = response_data["choices"][0]["message"]["content"]  
  
        info = {  
            "prompt_tokens": response_data.get("usage", {}).get("prompt_tokens", 0),  
            "completion_tokens": response_data.get("usage", {}).get("completion_tokens", 0),  
            "total_tokens": response_data.get("usage", {}).get("total_tokens", 0),  
        }  
  
        return message, info  
  
    def _format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:  
        """Format messages for Kimi API"""  
        formatted_messages = []  
          
        for message in messages:  
            if message["role"] == "system":  
                # Extract text content from system message  
                if isinstance(message["content"], list):  
                    content = message["content"][0]["text"]  
                else:  
                    content = message["content"]  
                  
                formatted_messages.append({  
                    "role": "system",  
                    "content": content  
                })  
            elif message["role"] in ["user", "assistant"]:  
                # Handle user and assistant messages  
                if isinstance(message["content"], list):  
                    # Extract text content, ignore images for now  
                    text_content = ""  
                    for content_item in message["content"]:  
                        if content_item.get("type") == "text":  
                            text_content += content_item["text"]  
                      
                    formatted_messages.append({  
                        "role": message["role"],  
                        "content": text_content  
                    })  
                else:  
                    formatted_messages.append({  
                        "role": message["role"],  
                        "content": message["content"]  
                    })  
          
        return formatted_messages  
  
    def assemble_prompt(self, template_str: str = None, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:  
        """Assemble prompt using the default tripartite mode"""  
        if config.DEFAULT_MESSAGE_CONSTRUCTION_MODE == constants.MESSAGE_CONSTRUCTION_MODE_TRIPART:  
            return self.assemble_prompt_tripartite(template_str=template_str, params=params)  
        elif config.DEFAULT_MESSAGE_CONSTRUCTION_MODE == constants.MESSAGE_CONSTRUCTION_MODE_PARAGRAPH:  
            return self.assemble_prompt_paragraph(template_str=template_str, params=params)  
  
    def assemble_prompt_tripartite(self, template_str: str = None, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:  
        """Assemble prompt in tripartite format (simplified version)"""  
        # This is a simplified implementation  
        # You may need to adapt this based on your specific requirements  
          
        if not template_str or not params:  
            return []  
  
        # Basic implementation - you can enhance this based on your needs  
        system_message = {  
            "role": "system",  
            "content": [{"type": "text", "text": "You are a helpful assistant."}]  
        }  
  
        user_message = {  
            "role": "user",   
            "content": [{"type": "text", "text": template_str}]  
        }  
  
        return [system_message, user_message]  
  
    def assemble_prompt_paragraph(self, template_str: str = None, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:  
        """Assemble prompt in paragraph format"""  
        raise NotImplementedError("Paragraph mode not implemented for Kimi provider yet.")