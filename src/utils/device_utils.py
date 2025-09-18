import torch
import logging
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)

class DeviceType(Enum):
    """Supported device types"""
    AUTO = "auto"           # Automatic selection
    CPU = "cpu"            # CPU only
    CUDA = "cuda"          # NVIDIA GPU
    MPS = "mps"            # Apple Silicon GPU (M1/M2/M3)

class DeviceInfo:
    """Device info"""
    def __init__(self,device: torch.device, device_type: DeviceType, device_name: str):
        self.device = device
        self.device_type = device_type
        self.device_name = device_name
        self.memory_gb = self._get_memory_info()

    def _get_memory_info(self) -> Optional[float]:
        """Get memory info in GB"""

        try:
            if self.device.type == DeviceType.CUDA.value:
                return torch.cuda.get_device_properties(self.device).total_memory
            elif self.device.type == DeviceType.MPS.value:
                return self._estimate_mps_memory()

    def _estimate_mps_memory(self) -> float:
        """Estimate MPS memory based on Mac model"""

        try:
            import platform
            import subprocess

            # Get Mac model info
            result = subprocess.run(['sysctl', '-n', 'hw.model'],
                                    capture_output=True, text=True)
            model = result.stdout.strip()

            # Rough estimates based on common Mac models
            if 'MacBookAir' in model:
                return 8.0  # Most MBA have 8GB unified memory
            elif 'MacBookPro' in model or 'iMac' in model or 'Mac' in model:
                return 16.0  # Conservative estimate
            else:
                return 8.0  # Default estimate
        except Exception:
            return 8.0  # Default fallback

    def detect_best_device(preferred_device: Optional[str] = None) -> DeviceInfo:
        






