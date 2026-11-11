"""Optional VGGT-Omega depth backend for offline BLASt3R inference."""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download


VGGT_OMEGA_REPO = 'facebook/VGGT-Omega'
VGGT_OMEGA_FILENAME = 'vggt_omega_1b_512.pt'


def _checkpoint_path(checkpoint):
    path = Path(checkpoint).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        for filename in (VGGT_OMEGA_FILENAME, 'model.pt'):
            candidate = path / filename
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f'no VGGT-Omega checkpoint found under {path}; expected '
            f'{VGGT_OMEGA_FILENAME} or model.pt')

    try:
        return Path(hf_hub_download(repo_id=str(checkpoint), filename=VGGT_OMEGA_FILENAME))
    except Exception as exc:
        raise RuntimeError(
            f'could not download gated checkpoint {checkpoint}/{VGGT_OMEGA_FILENAME}. '
            'Request access at https://huggingface.co/facebook/VGGT-Omega and '
            'authenticate with `hf auth login`, or pass a local checkpoint path.'
        ) from exc


class VGGTOmegaDepth:
    """Load VGGT-Omega and expose depth, confidence and intrinsics only.

    The BLASt3R adapter invokes :meth:`predict` once per image to enforce the
    monocular-depth experiment; this class itself does not combine views.
    """

    depth_backend = 'vggt_omega'
    force_dtype = torch.float32

    def __init__(self, checkpoint=VGGT_OMEGA_REPO, device='cuda'):
        try:
            from vggt_omega.models import VGGTOmega
            from vggt_omega.utils.pose_enc import encoding_to_camera
        except ImportError as exc:
            raise ImportError(
                'VGGT-Omega is optional. Install the official package first, e.g. '
                '`pip install --no-deps git+https://github.com/facebookresearch/vggt-omega.git` '
                'and ensure its dependency `einops` is installed.'
            ) from exc

        self.device = torch.device(device)
        self._encoding_to_camera = encoding_to_camera
        checkpoint = _checkpoint_path(checkpoint)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if isinstance(state, dict) and 'model' in state and not any(
                key.startswith('aggregator.') for key in state):
            state = state['model']
        self.model = VGGTOmega(autocast=self.device.type == 'cuda')
        self.model.load_state_dict(state, strict=True)
        del state
        self.model = self.model.to(self.device).eval()
        print(f'>> Loaded VGGT-Omega depth backend from {checkpoint}')

    @torch.inference_mode()
    def predict(self, images):
        """Predict on ``images`` shaped (N, 3, H, W), in RGB [0, 1]."""
        predictions = self.model(images.to(self.device))
        _, intrinsics = self._encoding_to_camera(
            predictions['pose_enc'], predictions['images'].shape[-2:])
        return predictions['depth'], predictions['depth_conf'], intrinsics
