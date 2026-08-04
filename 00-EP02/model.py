"""
Carga do VLM e interface única de geração para todos os experimentos.

Existem dois backends com a mesma interface `.ask()`:

  QwenBackend  Qwen2-VL de verdade. 4-bit NF4 quando há CUDA, float32 no
               fallback de CPU (bitsandbytes exige CUDA). Singleton, para não
               carregar o modelo duas vezes na VRAM.
  DryBackend   Modelo falso e determinístico. Não importa torch. Serve para
               validar a mecânica dos experimentos (prompt, checkpoint,
               parsing, voto de maioria, duas etapas da Reflexion) numa
               máquina sem GPU, antes de gastar horas de T4.

torch/transformers são importados só dentro do QwenBackend, então --dry-run
roda em ambiente sem nenhuma dessas dependências.
"""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# Limita a resolução/tokens de imagem para caber na VRAM de uma T4 (~15GB).
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28


def clear_vram() -> None:
    """Coleta lixo e esvazia o cache de CUDA. No-op sem torch."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_seed(seed: int) -> None:
    """Semeia o torch. Sem isso o self-consistency (que amostra) não repete."""
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Backend real
# --------------------------------------------------------------------------- #


def resolve_model_class():
    """Classe do VLM no transformers instalado; o nome mudou entre majors.

    `Qwen2VLForConditionalGeneration` é o nome histórico; nas versões novas o
    caminho genérico é `AutoModelForImageTextToText`. O lock pede >=5.14.1, onde
    isso não é garantido, então tenta em ordem em vez de importar direto.
    """
    import transformers

    for name in (
        "Qwen2VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise ImportError(
        "nenhuma classe de VLM encontrada no transformers instalado "
        "(tentei Qwen2VLForConditionalGeneration, AutoModelForImageTextToText, "
        "AutoModelForVision2Seq)"
    )


class QwenBackend:
    def __init__(self, model_id: str = MODEL_ID, four_bit: bool = True):
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        self.model_id = model_id

        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )

        model_cls = resolve_model_class()
        cuda = torch.cuda.is_available()

        if cuda and four_bit:
            from transformers import BitsAndBytesConfig

            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = model_cls.from_pretrained(
                model_id,
                quantization_config=quant,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            # bitsandbytes exige CUDA; em CPU cai para float32 e fica lento.
            self.model = model_cls.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if cuda else torch.float32,
                device_map="auto" if cuda else None,
            )
        self.model.eval()

        # Procedência, gravada em todo registro de checkpoint. 4-bit e float32
        # dão respostas diferentes; sem isso a mistura passa em silêncio.
        if cuda:
            self.tag = "cuda-4bit" if four_bit else "cuda-bf16"
        else:
            self.tag = "cpu-fp32"

        if not cuda:
            print(
                "AVISO: CUDA indisponível — rodando em CPU float32, ordem de "
                "minuto por item. No Colab: Runtime > Change runtime type > T4 GPU."
            )
        print(f"modelo carregado: {model_id} ({self.tag})")

    def ask(
        self,
        image_path: Path,
        system: str,
        user: str,
        max_new_tokens: int = 16,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> tuple[str, int]:
        """Uma geração. Devolve (texto cru, número de tokens novos)."""
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        conversation = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": user}],
            },
        ]
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(
            self.model.device
        )

        kwargs = {"max_new_tokens": max_new_tokens}
        if temperature is None:
            kwargs["do_sample"] = False
        else:
            kwargs.update(do_sample=True, temperature=temperature)
            if top_p is not None:
                kwargs["top_p"] = top_p

        with self.torch.no_grad():
            generated = self.model.generate(**inputs, **kwargs)

        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        raw = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return raw, int(trimmed.shape[1])


# --------------------------------------------------------------------------- #
# Backend falso, para validar a mecânica sem GPU
# --------------------------------------------------------------------------- #


class DryBackend:
    """Gera texto plausível e determinístico, sem olhar a imagem nem o label.

    Imita as duas formas de saída que os experimentos precisam parsear: resposta
    seca (baseline / prompt engineering) e raciocínio longo terminando em
    "Final Answer:" (CoT / self-consistency / Reflexion). Inclui números e
    booleanos distratores no meio do raciocínio, que é exatamente o que fazia o
    parser antigo errar.

    A acurácia resultante é aleatória — o objetivo é exercitar o encanamento,
    não medir qualidade.
    """

    tag = "dry"

    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def _pseudo(seed_text: str, modulo: int) -> int:
        digest = hashlib.md5(seed_text.encode("utf-8")).hexdigest()
        return int(digest, 16) % modulo

    def ask(
        self,
        image_path: Path,
        system: str,
        user: str,
        max_new_tokens: int = 16,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> tuple[str, int]:
        self.calls += 1
        # Com amostragem, varia entre chamadas para o voto de maioria ter o que
        # agregar; determinístico (greedy) repete a mesma saída.
        salt = f"|{self.calls}" if temperature is not None else ""
        is_count = "How many" in user

        if is_count:
            value = str(self._pseudo(user + salt, 5))
        else:
            value = "True" if self._pseudo(user + salt, 2) else "False"

        wants_reasoning = "step-by-step" in user or "corrected breakdown" in user.lower()
        if wants_reasoning:
            raw = (
                "Looking at the diagram, gate 1 is an OR and gate 2 is a NAND.\n"
                "Node x3 evaluates to True, node 7 evaluates to False.\n"
                f"Final Answer: {value}"
            )
        else:
            raw = value
        return raw, min(max_new_tokens, len(raw.split()))


# --------------------------------------------------------------------------- #
# Singleton
# --------------------------------------------------------------------------- #

_backend = None


def load_backend(dry_run: bool = False, model_id: str = MODEL_ID, four_bit: bool = True):
    """Carrega (uma vez) o backend de geração."""
    global _backend
    if _backend is None:
        _backend = DryBackend() if dry_run else QwenBackend(model_id, four_bit)
    return _backend
