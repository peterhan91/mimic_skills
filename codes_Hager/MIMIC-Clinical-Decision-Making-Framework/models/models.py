import os
import re
from os.path import join
from typing import Any, List, Mapping, Dict

import torch
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from transformers import GenerationConfig, StoppingCriteriaList
try:
    from auto_gptq import exllama_set_max_input_length
except ImportError:
    exllama_set_max_input_length = None
from langchain.llms.base import LLM
try:
    from exllamav2.generator import ExLlamaV2Sampler
except ImportError:
    ExLlamaV2Sampler = None
import tiktoken
try:
    import anthropic as anthropic_module
except ImportError:
    anthropic_module = None

from models.utils import create_stop_criteria, create_stop_criteria_exllama
from agents.agent import STOP_WORDS
from utils.nlp import extract_sections


# ── Qwen3.5 thinking block removal ──────────────────────────────────
# Qwen3.5 models generate <think>...</think> reasoning blocks by default.
# When using /v1/completions (not /v1/chat/completions), vLLM's
# --reasoning-parser has no effect and the raw <think> content appears
# in the output text.  This causes two problems:
#   1) The regex parser finds Action:/diagnosis: inside <think> blocks,
#      triggering false parses (e.g., thinking text parsed as lab names)
#   2) Thinking tokens accumulate in the agent scratchpad, wasting context
_RE_THINK_BLOCK = re.compile(r"<think>[\s\S]*?</think>", re.DOTALL)

# Configurable max_tokens for vLLM (override via VLLM_MAX_TOKENS env var)
VLLM_MAX_TOKENS = int(os.environ.get("VLLM_MAX_TOKENS", "4096"))

def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output."""
    return _RE_THINK_BLOCK.sub("", text).strip()


class CustomLLM(LLM):
    model_name: str
    max_context_length: int
    probabilities: torch.Tensor = None
    exllama: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    truncation_side: str = "left"
    model: Any
    generator: Any
    tokenizer: Any
    seed: int
    self_consistency: bool = False

    openai_api_key: str = None
    anthropic_api_key: str = None
    anthropic_client: Any = None
    vllm_base_url: str = None
    reasoning_effort: str = None
    tags: Dict[str, str] = None

    @property
    def _llm_type(self) -> Any:
        return "custom"

    @property
    def _llm_name(self) -> str:
        return self.model_name

    @property
    def _llm_device(self) -> str:
        return self.model.device

    @property
    def _llm_8bit(self) -> bool:
        return self.load_in_8bit

    @property
    def _llm_4bit(self) -> bool:
        return self.load_in_4bit

    @property
    def _llm_truncation_side(self) -> str:
        return self.truncation_side

    def load_model(self, base_models: str) -> None:
        torch.cuda.empty_cache()

        if self.model_name == "Human":
            return
        elif self.vllm_base_url:
            # vLLM serves model via OpenAI-compatible API — no local model loading,
            # but we still need the tokenizer for context length tracking.
            from transformers import AutoTokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            except OSError:
                # Fallback: model tokenizer not cached locally (e.g., gpt-oss uses tiktoken).
                # Use tiktoken for token counting — sufficient for context length tracking.
                self.tokenizer = tiktoken.get_encoding("o200k_base")
                print(f"  Tokenizer fallback: using tiktoken o200k_base for {self.model_name}")
            self.model = None
            print(f"Using vLLM server at {self.vllm_base_url} for {self.model_name}")
            return
        elif self.openai_api_key:
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                # Newer models (gpt-5.2, gpt-5-mini) not yet in tiktoken
                self.tokenizer = tiktoken.get_encoding("o200k_base")
            openai.api_key = self.openai_api_key
            return
        elif self.anthropic_api_key:
            if anthropic_module is None:
                raise ImportError("anthropic package required: pip install anthropic")
            self.anthropic_client = anthropic_module.Anthropic(api_key=self.anthropic_api_key)
            # No public Claude tokenizer; use cl100k_base as approximation.
            # With 200K context this is non-critical.
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            print(f"Using Anthropic API for {self.model_name}")
            return
        elif (
            self.model_name
            == "GeorgiaTechResearchInstitute/galactica-6.7b-evol-instruct-70k"
        ):
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(
                "GeorgiaTechResearchInstitute/galactica-6.7b-evol-instruct-70k"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                "GeorgiaTechResearchInstitute/galactica-6.7b-evol-instruct-70k",
                device_map="auto",
                torch_dtype=torch.float16,
            )

        elif "GPTQ" in self.model_name:
            if self.exllama:
                from exllamav2 import ExLlamaV2Cache
                from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer
                from models.exllamav2_generator_base_custom import (
                    ExLlamaV2BaseGenerator,
                )

                torch.cuda._lazy_init()
                config = ExLlamaV2Config()
                config.model_dir = join(base_models, self.model_name)
                config.prepare()
                config.max_seq_len = self.max_context_length
                config.scale_pos_emb = 1.0
                config.scale_alpha_value = 1.0
                config.no_flash_attn = False
                self.model = ExLlamaV2(config)
                self.model.load()
                self.tokenizer = ExLlamaV2Tokenizer(config)
                cache = ExLlamaV2Cache(self.model)
                self.generator = ExLlamaV2BaseGenerator(
                    self.model, cache, self.tokenizer
                )
                self.generator.warmup()

            else:
                from transformers import LlamaTokenizer, LlamaForCausalLM

                base_model = join(base_models, self.model_name)

                self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
                self.model = LlamaForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                self.model = exllama_set_max_input_length(
                    self.model, self.max_context_length
                )

        elif (
            self.model_name == "meta-llama/Meta-Llama-3-70B-Instruct"
            or self.model_name == "aaditya/OpenBioLLM-Llama3-70B"
            or self.model_name == "meta-llama/Meta-Llama-3.1-70B-Instruct"
            or self.model_name == "meta-llama/Llama-3.3-70B-Instruct"
        ):
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                BitsAndBytesConfig,
            )

            print(f"loading from {base_models}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=base_models,
            )

            eot = "<|eot_id|>"
            eot_id = self.tokenizer.convert_tokens_to_ids(eot)
            self.tokenizer.pad_token = eot
            self.tokenizer.pad_token_id = eot_id

            print("loaded tokenizer")
            bb_cfg = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=base_models,
                device_map="auto",
                quantization_config=bb_cfg,
            )
            print("loaded model")

        elif self.model_name == "google/medgemma-1.5-4b-it":
            from transformers import AutoTokenizer, AutoModelForCausalLM

            print(f"Loading MedGemma from {base_models}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=base_models,
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            print("Loaded tokenizer")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=base_models,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            print("Loaded MedGemma model")

        elif self.model_name == "Qwen/Qwen3-30B-A3B-Instruct-2507":
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                BitsAndBytesConfig,
            )

            print(f"Loading Qwen3-30B-A3B from {base_models}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=base_models,
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            print("Loaded tokenizer")

            bb_cfg = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=base_models,
                device_map="auto",
                quantization_config=bb_cfg,
            )
            print("Loaded Qwen3-30B-A3B model")

        elif self.model_name == "axiong/PMC_LLaMA_13B":
            from transformers import LlamaTokenizer, LlamaForCausalLM

            self.tokenizer = LlamaTokenizer.from_pretrained("axiong/PMC_LLaMA_13B")
            self.model = LlamaForCausalLM.from_pretrained(
                "axiong/PMC_LLaMA_13B", device_map="auto", torch_dtype=torch.float16
            )

        elif self.model_name == "google/flan-t5-xxl":
            from transformers import T5Tokenizer, T5ForConditionalGeneration

            self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
            self.model = T5ForConditionalGeneration.from_pretrained(
                "google/flan-t5-xxl", device_map="auto", torch_dtype=torch.float16
            )

        elif self.model_name == "bigscience/T0pp":
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            orig = os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION")
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

            self.tokenizer = AutoTokenizer.from_pretrained("bigscience/T0pp")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                "bigscience/T0pp", device_map="auto", torch_dtype=torch.float16
            )

            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = orig

        elif self.model_name.startswith("togethercomputer/RedPajama-INCITE"):
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, pad_token="[PAD]"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float16, device_map="auto"
            )

        elif self.model_name.startswith("tiiuae/falcon"):
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.load_in_8bit = "40b" in self.model_name
            if torch.cuda.device_count() > 1:
                self.load_in_8bit = False
                device_map = "balanced_low_0"
            else:
                device_map = "sequential"

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, pad_token="[PAD]"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                load_in_8bit=self.load_in_8bit,
                device_map=device_map,
            )

        else:
            raise ValueError("Model name not recognized")

        if not self.model_name.startswith("tiiuae/falcon") and not self.exllama:
            self.model.eval()
            if torch.__version__ >= "2" and "medgemma" not in self.model_name.lower() and "qwen3" not in self.model_name.lower():
                self.model = torch.compile(self.model)

        self.tokenizer.truncation_side = "left"

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def completion_with_backoff(self, **kwargs):
        return openai.ChatCompletion.create(**kwargs)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def anthropic_completion_with_backoff(self, **kwargs):
        return self.anthropic_client.messages.create(**kwargs)

    def _build_anthropic_messages(self, prompt):
        """Parse prompt into Anthropic system + messages format.

        extract_sections() returns a list of {role, content} dicts.
        Claude API requires system as a separate parameter, and the
        messages list must start with a user message and alternate
        user/assistant. We merge consecutive same-role messages.

        Claude 4.x does not support assistant prefilling (conversation must
        end with a user message). The framework appends an open assistant
        message like "Thought:" to steer output format. We move any trailing
        assistant content into the last user message as an instruction.
        """
        sections = extract_sections(prompt, self.tags)

        system_text = ""
        messages = []
        for sec in sections:
            if sec["role"] == "system":
                system_text += ("\n\n" + sec["content"]) if system_text else sec["content"]
            else:
                # Merge consecutive messages with the same role
                if messages and messages[-1]["role"] == sec["role"]:
                    messages[-1]["content"] += "\n" + sec["content"]
                else:
                    messages.append({"role": sec["role"], "content": sec["content"]})

        # Claude API requires messages to start with user role.
        if messages and messages[0]["role"] != "user":
            messages.insert(0, {"role": "user", "content": ""})

        # Claude 4.x: no assistant prefill. If last message is assistant
        # (e.g. "Thought:" from agent scratchpad), fold it into the
        # preceding user message as a format instruction.
        if messages and messages[-1]["role"] == "assistant":
            prefill = messages.pop()["content"].strip()
            if prefill:
                hint = f'\n\nBegin your response with exactly "{prefill}"'
                if messages and messages[-1]["role"] == "user":
                    messages[-1]["content"] += hint
                else:
                    messages.append({"role": "user", "content": hint.strip()})
            # If popping left an empty list or consecutive same-role,
            # ensure we still have a valid user message.
            if not messages:
                messages.append({"role": "user", "content": ""})

        return system_text, messages

    def remove_input_tokens(self, output_tokens, ids):
        # Truncate the larger tensor to match the size of the smaller one
        min_size = min(output_tokens.size(1), ids.size(1))
        truncated_output_tokens = output_tokens[:, :min_size]
        truncated_ids = ids[:, :min_size]

        # Element-wise comparison and cumulative product to count length of common prefix
        common_prefix = (
            (truncated_output_tokens == truncated_ids).cumprod(dim=0).sum().item()
        )

        return output_tokens[:, common_prefix:]

    def _call(
        self,
        prompt: str,
        stop: List[str],
        do_sample=True,
        temperature=0.01,
        top_k=1,
        top_p=0.95,
        num_beams=1,
        repetition_penalty=1.2,
        length_penalty=1.0,
        **kwargs,
    ) -> str:
        self.probabilities = None
        if self.model_name == "Human":
            output = input(prompt)

        elif self.vllm_base_url:
            import requests
            _effort = getattr(self, "reasoning_effort", None)
            if _effort and _effort != "none":
                # Reasoning model (e.g., gpt-oss): use chat completions API
                messages = extract_sections(prompt, self.tags)
                resp = requests.post(
                    f"{self.vllm_base_url}/chat/completions",
                    json={
                        "model": self.model_name,
                        "messages": messages,
                        "max_tokens": VLLM_MAX_TOKENS,
                        "temperature": 0.0,
                        "reasoning_effort": _effort,
                        "stop": STOP_WORDS + stop,
                        "seed": self.seed,
                    },
                    timeout=300,
                )
                resp.raise_for_status()
                output = resp.json()["choices"][0]["message"]["content"]
            else:
                resp = requests.post(
                    f"{self.vllm_base_url}/completions",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "max_tokens": VLLM_MAX_TOKENS,
                        "temperature": 0.0,
                        "stop": STOP_WORDS + stop,
                        "seed": self.seed,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                output = _strip_think_blocks(resp.json()["choices"][0]["text"])

        elif self.openai_api_key:
            messages = extract_sections(
                prompt,
                self.tags,
            )

            # GPT-5 family: no stop sequences; reasoning models need special handling
            _no_stop = self.model_name.startswith("gpt-5")
            _reasoning = getattr(self, "reasoning_effort", None)
            api_kwargs = dict(
                model=self.model_name,
                messages=messages,
            )
            # temperature/seed only supported when reasoning effort is "none" or absent
            if _reasoning and _reasoning != "none":
                api_kwargs["reasoning_effort"] = _reasoning
            else:
                if not self.model_name.startswith("gpt-5-mini"):
                    api_kwargs["temperature"] = 0.0
                    api_kwargs["seed"] = self.seed
            if STOP_WORDS and not _no_stop:
                api_kwargs["stop"] = STOP_WORDS

            response = self.completion_with_backoff(**api_kwargs)
            output = response["choices"][0]["message"]["content"]

        elif self.anthropic_api_key:
            system_text, messages = self._build_anthropic_messages(prompt)

            _effort = getattr(self, "reasoning_effort", None)
            if _effort and _effort != "none":
                # Extended thinking: no temperature, higher max_tokens
                api_kwargs = dict(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=16000,
                    thinking={"type": "adaptive"},
                    output_config={"effort": _effort},
                )
            else:
                api_kwargs = dict(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.0,
                )
            if system_text:
                api_kwargs["system"] = system_text
            if STOP_WORDS:
                api_kwargs["stop_sequences"] = STOP_WORDS

            response = self.anthropic_completion_with_backoff(**api_kwargs)
            # With thinking enabled, response may contain thinking blocks + text blocks
            output = None
            for block in response.content:
                if block.type == "text":
                    output = block.text
                    break
            if output is None:
                output = response.content[0].text

        elif self.exllama:
            with torch.inference_mode():
                ids = self.tokenizer.encode(prompt, encode_special_tokens=True)
                tokens_prompt = ids.shape[-1]

                settings = ExLlamaV2Sampler.Settings()
                if self.self_consistency:
                    settings = settings.clone()
                    settings.temperature = 0.7
                    seed = None
                else:
                    settings = settings.greedy_clone()
                    seed = self.seed

                stop_criteria = create_stop_criteria_exllama(
                    stop, self.tokenizer.eos_token_id, self.tokenizer
                )

                output_tokens, self.probabilities = self.generator.generate_simple(
                    prompt,
                    gen_settings=settings,
                    num_tokens=self.max_context_length - tokens_prompt,
                    seed=seed,
                    token_healing=True,
                    encode_special_tokens=True,
                    decode_special_tokens=False,
                    stop_criteria=stop_criteria,
                )

                output_tokens = self.remove_input_tokens(output_tokens, ids)
                output = self.tokenizer.decode(
                    output_tokens, decode_special_tokens=False
                )[0]
        else:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.max_context_length,
                truncation=True,
                padding=False,
            )
            input_ids = inputs["input_ids"].to(self.model.device)

            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )

            stop_criteria = create_stop_criteria(
                stop, self.tokenizer, self.model.device
            )

            with torch.no_grad():
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    stopping_criteria=StoppingCriteriaList([stop_criteria]),
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_length=self.max_context_length,
                )

            s = generation_output.sequences
            s_no_input = s[:, input_ids.shape[1] :]
            output = self.tokenizer.batch_decode(s_no_input, skip_special_tokens=True)[
                0
            ]

        # Remove observations strings from output if generated
        for stop_word in STOP_WORDS + stop:
            output = output.replace(stop_word, "")

        return output.strip()

    def generate_with_temperature(
        self,
        prompt: str,
        stop: List[str],
        temperature: float = 0.7,
    ) -> str:
        """Generate text with explicit temperature control for ToT sampling.

        Unlike _call() which uses hardcoded temperature=0.0 for most backends,
        this method respects the passed temperature across all backends.
        """
        self.probabilities = None

        if self.model_name == "Human":
            return input(prompt)

        elif self.vllm_base_url:
            import requests
            _effort = getattr(self, "reasoning_effort", None)
            if _effort and _effort != "none":
                messages = extract_sections(prompt, self.tags)
                resp = requests.post(
                    f"{self.vllm_base_url}/chat/completions",
                    json={
                        "model": self.model_name,
                        "messages": messages,
                        "max_tokens": VLLM_MAX_TOKENS,
                        "temperature": temperature,
                        "reasoning_effort": _effort,
                        "stop": STOP_WORDS + stop,
                        "seed": None if temperature > 0 else self.seed,
                    },
                    timeout=300,
                )
                resp.raise_for_status()
                output = resp.json()["choices"][0]["message"]["content"]
            else:
                resp = requests.post(
                    f"{self.vllm_base_url}/completions",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "max_tokens": VLLM_MAX_TOKENS,
                        "temperature": temperature,
                        "stop": STOP_WORDS + stop,
                        "seed": None if temperature > 0 else self.seed,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                output = _strip_think_blocks(resp.json()["choices"][0]["text"])

        elif self.openai_api_key:
            messages = extract_sections(prompt, self.tags)
            _no_stop = self.model_name.startswith("gpt-5")
            _reasoning = getattr(self, "reasoning_effort", None)
            api_kwargs = dict(
                model=self.model_name,
                messages=messages,
            )
            if _reasoning and _reasoning != "none":
                api_kwargs["reasoning_effort"] = _reasoning
            else:
                if not self.model_name.startswith("gpt-5-mini"):
                    api_kwargs["temperature"] = temperature
                    api_kwargs["seed"] = None if temperature > 0 else self.seed
            if STOP_WORDS and not _no_stop:
                api_kwargs["stop"] = STOP_WORDS

            response = self.completion_with_backoff(**api_kwargs)
            output = response["choices"][0]["message"]["content"]

        elif self.anthropic_api_key:
            system_text, messages = self._build_anthropic_messages(prompt)

            _effort = getattr(self, "reasoning_effort", None)
            if _effort and _effort != "none":
                api_kwargs = dict(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=16000,
                    thinking={"type": "adaptive"},
                    output_config={"effort": _effort},
                )
            else:
                api_kwargs = dict(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=temperature,
                )
            if system_text:
                api_kwargs["system"] = system_text
            if STOP_WORDS:
                api_kwargs["stop_sequences"] = STOP_WORDS

            response = self.anthropic_completion_with_backoff(**api_kwargs)
            output = None
            for block in response.content:
                if block.type == "text":
                    output = block.text
                    break
            if output is None:
                output = response.content[0].text

        elif self.exllama:
            with torch.inference_mode():
                ids = self.tokenizer.encode(prompt, encode_special_tokens=True)
                tokens_prompt = ids.shape[-1]

                settings = ExLlamaV2Sampler.Settings()
                if temperature > 0:
                    settings = settings.clone()
                    settings.temperature = temperature
                    seed = None
                else:
                    settings = settings.greedy_clone()
                    seed = self.seed

                stop_criteria = create_stop_criteria_exllama(
                    stop, self.tokenizer.eos_token_id, self.tokenizer
                )

                output_tokens, self.probabilities = self.generator.generate_simple(
                    prompt,
                    gen_settings=settings,
                    num_tokens=self.max_context_length - tokens_prompt,
                    seed=seed,
                    token_healing=True,
                    encode_special_tokens=True,
                    decode_special_tokens=False,
                    stop_criteria=stop_criteria,
                )

                output_tokens = self.remove_input_tokens(output_tokens, ids)
                output = self.tokenizer.decode(
                    output_tokens, decode_special_tokens=False
                )[0]

        else:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.max_context_length,
                truncation=True,
                padding=False,
            )
            input_ids = inputs["input_ids"].to(self.model.device)

            do_sample = temperature > 0
            generation_config = GenerationConfig(
                temperature=temperature if do_sample else 1.0,
                top_p=0.95 if do_sample else 1.0,
                top_k=50 if do_sample else 1,
                do_sample=do_sample,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            stop_criteria_obj = create_stop_criteria(
                stop, self.tokenizer, self.model.device
            )

            with torch.no_grad():
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    stopping_criteria=StoppingCriteriaList([stop_criteria_obj]),
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_length=self.max_context_length,
                )

            s = generation_output.sequences
            s_no_input = s[:, input_ids.shape[1]:]
            output = self.tokenizer.batch_decode(
                s_no_input, skip_special_tokens=True
            )[0]

        for stop_word in STOP_WORDS + stop:
            output = output.replace(stop_word, "")

        return output.strip()

    def generate_batch(
        self,
        prompt: str,
        stop: List[str],
        n: int,
        temperature: float = 0.7,
    ) -> List[str]:
        """Generate n completions in a single request (vLLM) or n sequential calls (fallback).

        vLLM's /v1/completions supports an `n` parameter that returns multiple
        completions sharing the prompt KV cache — ~3.5x faster than n separate calls.
        """
        if self.vllm_base_url:
            _effort = getattr(self, "reasoning_effort", None)
            if _effort and _effort != "none":
                # Reasoning models: sequential via chat completions
                return [
                    self.generate_with_temperature(prompt, stop=stop, temperature=temperature)
                    for _ in range(n)
                ]
            import requests
            resp = requests.post(
                f"{self.vllm_base_url}/completions",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": VLLM_MAX_TOKENS,
                    "temperature": temperature,
                    "stop": STOP_WORDS + stop,
                    "seed": None if temperature > 0 else self.seed,
                    "n": n,
                },
                timeout=300,
            )
            resp.raise_for_status()
            choices = resp.json()["choices"]
            outputs = []
            for choice in sorted(choices, key=lambda c: c["index"]):
                text = _strip_think_blocks(choice["text"])
                for stop_word in STOP_WORDS + stop:
                    text = text.replace(stop_word, "")
                outputs.append(text.strip())
            return outputs

        # Fallback: sequential calls for non-vLLM backends
        return [
            self.generate_with_temperature(prompt, stop=stop, temperature=temperature)
            for _ in range(n)
        ]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "load_in_8bit": self.load_in_8bit,
            "load_in_4bit": self.load_in_4bit,
        }
