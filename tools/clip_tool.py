import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from config.settings import settings


class ClipTool:
    """多语言 CLIP 图像/文本向量提取工具。

    文本：clip-ViT-B-32-multilingual-v1（多语言）
    图像：原版 CLIP ViT-B/32 图像编码器（共享 512 维空间）
    """

    def __init__(self):
        self.device = settings.device
        clip_model = settings.models["clip"]
        self.text_model = SentenceTransformer(clip_model, device=self.device)
        self.image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def get_image_embedding(self, image: Image.Image) -> list:
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.image_model.get_image_features(**inputs)
            if not isinstance(output, torch.Tensor):
                output = output.image_embeds if hasattr(output, 'image_embeds') else output.pooler_output
            features = output / output.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy()[0].tolist()

    def get_text_embedding(self, text: str) -> list:
        emb = self.text_model.encode(text, normalize_embeddings=True)
        return emb.tolist()


_clip_tool: ClipTool | None = None


def get_clip_tool() -> ClipTool:
    global _clip_tool
    if _clip_tool is None:
        _clip_tool = ClipTool()
    return _clip_tool
