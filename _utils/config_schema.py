"""
Type definitions and schema for configuration files using dataclasses.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from .constants import Config

@dataclass
class VolumeConfig:
    name: str
    path: str

@dataclass
class DatasetConfig:
    resolutions: List[int]
    enable_ar_bucket: bool
    min_ar: float
    max_ar: float
    num_ar_buckets: int
    frame_buckets: List[int]
    num_repeats: int

@dataclass
class TrainingConfig:
    resume: bool
    resume_folder: str
    dataset_dir: str
    epochs: int
    micro_batch_size: int
    gradient_accum_steps: int
    warmup_steps: int
    learning_rate: float
    weight_decay: float
    lora_rank: int
    lora_dtype: str
    save_every_n_epochs: int
    checkpoint_mode: str
    checkpoint_frequency: int
    pipeline_stages: int
    gradient_clipping: float
    activation_checkpointing: bool
    eval_every_n_epochs: int
    caching_batch_size: int
    steps_per_print: int
    dataset: DatasetConfig

@dataclass
class CoreConfig:
    gpu_type: str
    gpu_count: int
    timeout_hours: int
    gpu_type_test: str
    gpu_count_test: int
    timeout_hours_test: int
    volumes: List[VolumeConfig]

@dataclass
class InferenceConfig:
    api_endpoint: str
    workflow_path: str
    test_enabled: bool
    test_strength: float
    test_height: int
    test_width: int
    test_frames: int
    test_steps: int
    test_mode: str
    test_epochs: List[int]
    test_latest_n: int
    test_prompts: List[str]
    test_folder: str

@dataclass
class ModelSpec:
    link: str
    type: str
    filename: Optional[str] = None

@dataclass
class InferenceModelsConfig:
    nodes: List[str]
    specs: List[ModelSpec]

@dataclass
class ModelsConfig:
    training: Dict[str, Any]  # Dynamic based on model types
    inference: InferenceModelsConfig

@dataclass
class HFConfig:
    auto_upload: bool
    private_repo: bool
    upload_test_outputs: bool
    upload_tensorboard: bool
    force_redownload: bool
    skip_existing: bool

class AppConfig:
    def __init__(self, raw_config: Dict[str, Any]):
        self._raw = raw_config
        
        # Parse core config
        self.core = CoreConfig(
            gpu_type=self._get([Config.Sections.CORE, Config.Keys.GPU_TYPE]),
            gpu_count=self._get([Config.Sections.CORE, Config.Keys.GPU_COUNT]),
            timeout_hours=self._get([Config.Sections.CORE, Config.Keys.TIMEOUT]),
            gpu_type_test=self._get([Config.Sections.CORE, Config.Keys.GPU_TYPE_TEST]),
            gpu_count_test=self._get([Config.Sections.CORE, Config.Keys.GPU_COUNT_TEST]),
            timeout_hours_test=self._get([Config.Sections.CORE, Config.Keys.TIMEOUT_TEST]),
            volumes=[VolumeConfig(**v) for v in self._get([Config.Sections.CORE, Config.Keys.VOLUMES], [])]
        )
        
        # Parse dataset config
        dataset_config = DatasetConfig(**self._get([Config.Sections.TRAINING, "dataset"], {}))
        
        # Parse training config
        self.training = TrainingConfig(
            resume=self._get([Config.Sections.TRAINING, Config.Keys.RESUME]),
            resume_folder=self._get([Config.Sections.TRAINING, Config.Keys.RESUME_FOLDER]),
            dataset_dir=self._get([Config.Sections.TRAINING, Config.Keys.DATASET_DIR]),
            epochs=self._get([Config.Sections.TRAINING, Config.Keys.EPOCHS]),
            micro_batch_size=self._get([Config.Sections.TRAINING, Config.Keys.BATCH_SIZE]),
            gradient_accum_steps=self._get([Config.Sections.TRAINING, "gradient_accum_steps"]),
            warmup_steps=self._get([Config.Sections.TRAINING, "warmup_steps"]),
            learning_rate=self._get([Config.Sections.TRAINING, "learning_rate"]),
            weight_decay=self._get([Config.Sections.TRAINING, "weight_decay"]),
            lora_rank=self._get([Config.Sections.TRAINING, "lora_rank"]),
            lora_dtype=self._get([Config.Sections.TRAINING, "lora_dtype"]),
            save_every_n_epochs=self._get([Config.Sections.TRAINING, "save_every_n_epochs"]),
            checkpoint_mode=self._get([Config.Sections.TRAINING, "checkpoint_mode"]),
            checkpoint_frequency=self._get([Config.Sections.TRAINING, "checkpoint_frequency"]),
            pipeline_stages=self._get([Config.Sections.TRAINING, "pipeline_stages"]),
            gradient_clipping=self._get([Config.Sections.TRAINING, "gradient_clipping"]),
            activation_checkpointing=self._get([Config.Sections.TRAINING, "activation_checkpointing"]),
            eval_every_n_epochs=self._get([Config.Sections.TRAINING, "eval_every_n_epochs"]),
            caching_batch_size=self._get([Config.Sections.TRAINING, "caching_batch_size"]),
            steps_per_print=self._get([Config.Sections.TRAINING, "steps_per_print"]),
            dataset=dataset_config
        )
        
        # Parse inference config
        self.inference = InferenceConfig(
            api_endpoint=self._get([Config.Sections.INFERENCE, "api_endpoint"]),
            workflow_path=self._get([Config.Sections.INFERENCE, "workflow_path"]),
            test_enabled=self._get([Config.Sections.INFERENCE, "test_enabled"]),
            test_strength=self._get([Config.Sections.INFERENCE, "test_strength"]),
            test_height=self._get([Config.Sections.INFERENCE, "test_height"]),
            test_width=self._get([Config.Sections.INFERENCE, "test_width"]),
            test_frames=self._get([Config.Sections.INFERENCE, "test_frames"]),
            test_steps=self._get([Config.Sections.INFERENCE, "test_steps"]),
            test_mode=self._get([Config.Sections.INFERENCE, "test_mode"]),
            test_epochs=self._get([Config.Sections.INFERENCE, "test_epochs"]),
            test_latest_n=self._get([Config.Sections.INFERENCE, "test_latest_n"]),
            test_prompts=self._get([Config.Sections.INFERENCE, "test_prompts"]),
            test_folder=self._get([Config.Sections.INFERENCE, "test_folder"])
        )
        
        # Parse models config
        inference_models = InferenceModelsConfig(
            nodes=self._get([Config.Sections.MODELS, "inference", "nodes"]),
            specs=[ModelSpec(**spec) for spec in self._get([Config.Sections.MODELS, "inference", "specs"])]
        )
        
        self.models = ModelsConfig(
            training=self._get([Config.Sections.MODELS, "training"]),
            inference=inference_models
        )
        
        # Parse HF config
        self.hf = HFConfig(
            auto_upload=self._get([Config.Sections.HF, "auto_upload"]),
            private_repo=self._get([Config.Sections.HF, "private_repo"]),
            upload_test_outputs=self._get([Config.Sections.HF, "upload_test_outputs"]),
            upload_tensorboard=self._get([Config.Sections.HF, "upload_tensorboard"]),
            force_redownload=self._get([Config.Sections.HF, "force_redownload"]),
            skip_existing=self._get([Config.Sections.HF, "skip_existing"])
        )

    def _get(self, keys: list, default: Any = None) -> Any:
        """Helper to navigate nested config"""
        value = self._raw
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value

    @property
    def dataset_path(self) -> str:
        """Get full dataset path"""
        from .constants import Paths
        return f"{Paths.ROOT}/{self.training.dataset_dir}" 