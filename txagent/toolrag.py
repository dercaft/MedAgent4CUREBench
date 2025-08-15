import hashlib
import json
import os
import re
import requests
import torch
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingModel(ABC):
    """
    Abstract base class for embedding models.
    It defines the common interface for encoding text and identifying the model.
    """
    @abstractmethod
    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        """Encodes a list of texts into embeddings."""
        pass

    @abstractmethod
    def get_identifier(self) -> str:
        """
        Returns a unique string identifier for the model, used for caching.
        For local models, this might be the model name.
        For API models, this could be the endpoint URL or model name from the API.
        """
        pass

class LocalEmbeddingModel(EmbeddingModel):
    """Local embedding model powered by SentenceTransformers."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        # Configure model properties as in the original code
        self.model.max_seq_length = 4096
        self.model.tokenizer.padding_side = "right"

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        """
        Encodes texts using the local SentenceTransformer model.
        kwargs are passed directly to the model's encode method.
        """
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            **kwargs
        )
    
    def get_identifier(self) -> str:
        # Return a filesystem-friendly version of the model name
        return self.model_name.replace("/", "_")

class APIEmbeddingModel(EmbeddingModel):
    """API-based embedding model."""
    def __init__(self, api_url: str, model_name: str, api_key: str = None):
        if not api_url.endswith("/"):
            api_url += "/"
        self.api_url = f"{api_url}embed" # Assuming a common REST pattern like /embed
        self.model_name = model_name
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        """
        Encodes texts by calling a remote API endpoint.
        kwargs are ignored for the API model in this implementation.
        """
        payload = {
            "model": self.model_name,
            "texts": texts
        }
        try:
            response = requests.post(self.api_url, json=payload, headers=self.headers)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            result = response.json()
            
            # Assuming the API returns a dictionary with an 'embeddings' key
            if "embeddings" not in result:
                raise ValueError("API response did not contain 'embeddings' key")
            
            # Ensure embeddings are normalized client-side if not guaranteed by API
            embeddings = np.array(result["embeddings"], dtype=np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / norms
            return normalized_embeddings

        except requests.exceptions.RequestException as e:
            print(f"Error calling embedding API: {e}")
            return np.array([])

    def get_identifier(self) -> str:
        # Create an identifier from the API URL and model name
        clean_url = self.api_url.split('//')[-1].replace('/', '_').replace('.', '_')
        return f"api_{clean_url}_{self.model_name}"

class SiliconFlowEmbeddingModel(EmbeddingModel):
    """
    Embedding model client for the SiliconFlow API.
    
    This class handles communication with the SiliconFlow /v1/embeddings endpoint,
    supporting various models including the Qwen/Qwen3 series with customizable dimensions.
    """
    
    BASE_URL = "https://api.siliconflow.cn/v1/embeddings"

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None
    ):
        """
        Initializes the SiliconFlow embedding model client.

        Args:
            model_name (str): The name of the model to use, e.g., 'Qwen/Qwen3-Embedding-8B'.
            api_key (Optional[str]): Your SiliconFlow API key. If not provided, it will
                                     be read from the SILICONFLOW_API_KEY environment variable.
            dimensions (Optional[int]): The desired number of dimensions for the output
                                        embeddings. Only supported by certain models like
                                        the Qwen/Qwen3 series.
        """
        self.model_name = model_name
        
        # Validate dimensions for Qwen3 models based on documentation
        if "Qwen/Qwen3" in self.model_name and dimensions:
            self._validate_dimensions(dimensions)
        elif dimensions:
            print(f"Warning: 'dimensions' parameter is set but may not be supported by model '{self.model_name}'.")

        self.dimensions = dimensions
        
        api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            raise ValueError("SiliconFlow API key not provided. Please pass it as an argument or set the 'SILICONFLOW_API_KEY' environment variable.")

        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _validate_dimensions(self, dimensions: int):
        """Private method to validate dimension choices for Qwen3 models."""
        valid_dims = {
            "Qwen/Qwen3-Embedding-8B": [64, 128, 256, 512, 768, 1024, 2048, 4096],
            "Qwen/Qwen3-Embedding-4B": [64, 128, 256, 512, 768, 1024, 2048],
            "Qwen/Qwen3-Embedding-0.6B": [64, 128, 256, 512, 768, 1024],
        }
        supported_dims = valid_dims.get(self.model_name)
        if supported_dims and dimensions not in supported_dims:
            raise ValueError(f"Invalid dimensions '{dimensions}' for model '{self.model_name}'. "
                             f"Supported values are: {supported_dims}")
            
    def get_identifier(self) -> str:
        """
        Returns a unique, filesystem-friendly string identifier for the model configuration.
        """
        # Sanitize model name by replacing slashes
        safe_model_name = self.model_name.replace("/", "_")
        identifier = f"siliconflow_{safe_model_name}"
        if self.dimensions:
            identifier += f"_{self.dimensions}d"
        return identifier

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Encodes a list of texts into embeddings using the SiliconFlow API.

        Args:
            texts (List[str]): A list of strings to be embedded.

        Returns:
            np.ndarray: A numpy array of shape (n_texts, embedding_dim) containing
                        the normalized embeddings.
        """
        if not texts or not all(isinstance(t, str) for t in texts):
            raise ValueError("Input 'texts' must be a non-empty list of strings.")

        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float"
        }
        
        if self.dimensions:
            payload["dimensions"] = self.dimensions

        try:
            response = requests.post(self.BASE_URL, headers=self.headers, json=payload)
            # Raise an exception for HTTP error codes (4xx or 5xx)
            response.raise_for_status()
            
            result = response.json()
            
            # Sort the embeddings by their original index to ensure correct order
            data = sorted(result["data"], key=lambda item: item["index"])
            
            embeddings = np.array([item["embedding"] for item in data], dtype=np.float32)
            
            # It's best practice to normalize embeddings for cosine similarity calculations
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Avoid division by zero for zero-vectors
            norms[norms == 0] = 1e-12
            normalized_embeddings = embeddings / norms
            
            return normalized_embeddings

        except requests.exceptions.RequestException as e:
            print(f"Error calling SiliconFlow API: {e}")
            # Try to print more detailed error from API response if available
            if e.response is not None:
                print(f"API Response: {e.response.text}")
            return np.array([])
        except (KeyError, IndexError) as e:
            print(f"Error parsing API response: {e}. Received: {response.text}")
            return np.array([])
    
def get_md5(text: str) -> str:
    """Computes the MD5 hash of a given text."""
    return hashlib.md5(text.encode()).hexdigest()

class ToolRAGModel:
    def __init__(self, embedding_model: EmbeddingModel, device: str = None):
        """
        Initializes the ToolRAGModel with a dependency-injected embedding model.
        
        Args:
            embedding_model: An instance of a class that inherits from EmbeddingModel.
            device: The torch device to use for computations ('cuda', 'cpu', etc.).
        """
        self.embedding_model = embedding_model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"ToolRAGModel is using device: {self.device}")
        
        self.tool_desc_embedding = None
        self.tool_name = None
        self.tool_embedding_path = None

    def load_tool_desc_embedding(self, toolbox):
        """
        Generates or loads cached embeddings for the tools in the toolbox.
        The cache filename is now based on the embedding model's unique identifier.
        """
        self.tool_name, _ = toolbox.refresh_tool_name_desc(enable_full_desc=True)
        all_tools_str = [json.dumps(each) for each in toolbox.prepare_tool_prompts(toolbox.all_tools)]
        print("ToolRAGModel: all_tools_str number:", len(all_tools_str))
        md5_value = get_md5(str(all_tools_str))
        model_identifier = self.embedding_model.get_identifier()
        
        # Sanitize the identifier to be a valid filename
        safe_model_id = re.sub(r'[\\/*?:"<>|]', "", model_identifier)
        self.tool_embedding_path = f"{safe_model_id}_embedding_{md5_value}.pt"

        try:
            self.tool_desc_embedding = torch.load(self.tool_embedding_path, map_location=self.device)
            assert len(self.tool_desc_embedding) == len(toolbox.all_tools), \
                "Mismatch between cached embeddings and current tool count."
            print(f"Successfully loaded cached tool embeddings from {self.tool_embedding_path}")
        except FileNotFoundError:
            print("\033[92mNo cache found. Inferring tool embeddings.\033[0m")
            # Encode using the injected model
            embeddings_np = self.embedding_model.encode(all_tools_str)
            self.tool_desc_embedding = torch.from_numpy(embeddings_np).to(self.device)
            
            torch.save(self.tool_desc_embedding, self.tool_embedding_path)
            print(f"\033[92mSaved new tool embeddings to {self.tool_embedding_path}.\033[0m")
        except Exception as e:
            print(f"An error occurred while loading embeddings: {e}. Re-inferring.")
            # Handle other exceptions like assertion errors by re-inferring
            embeddings_np = self.embedding_model.encode(all_tools_str)
            self.tool_desc_embedding = torch.from_numpy(embeddings_np).to(self.device)
            torch.save(self.tool_desc_embedding, self.tool_embedding_path)

    def rag_infer(self, query: str, top_k: int = 5) -> list[str]:
        """Performs RAG inference to find the most relevant tools for a given query."""
        if self.tool_desc_embedding is None:
            raise RuntimeError("Tool descriptions are not loaded. Call load_tool_desc_embedding() first.")

        # Encode the query using the injected model
        query_embedding_np = self.embedding_model.encode([query])
        query_embedding = torch.from_numpy(query_embedding_np).to(self.device)
        
        # Calculate similarity (dot product of normalized embeddings)
        # Using torch.mm for efficient matrix multiplication
        scores = torch.mm(query_embedding, self.tool_desc_embedding.T)[0]
        
        top_k = min(top_k, len(self.tool_name))
        top_k_scores, top_k_indices = torch.topk(scores, top_k)
        
        top_k_tool_names = [self.tool_name[i] for i in top_k_indices]
        
        print("\n--- RAG Inference Results ---")
        for name, score in zip(top_k_tool_names, top_k_scores):
            print(f"Q: '{query}' -T: {name:<25} | S: {score.item():.4f}")
        
        return top_k_tool_names