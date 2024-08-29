from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Optional, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pydantic import Field

class HuggingFaceLLM(LLM):
    model_id: str = Field(..., description="Hugging Face model ID")
    model: Any = Field(default=None, exclude=True)
    tokenizer: Any = Field(default=None, exclude=True)
    device: str = Field(default="cuda" if torch.cuda.is_available() else "cpu")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: int = Field(default=256, description="Maximum number of tokens to generate")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_model()

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device)

    @property
    def _llm_type(self) -> str:
        return "custom_huggingface"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response[len(prompt):].strip()

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_id": self.model_id, "temperature": self.temperature, "max_tokens": self.max_tokens}
