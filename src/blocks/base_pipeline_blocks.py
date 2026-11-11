import cv2
import numpy as np
import torch

from src.blocks import PipelineBlock
from src.blocks.utils import create_black_level_mask


class UndistortBlock(PipelineBlock):
    def __init__(self, previous_block: PipelineBlock, intrinsic_matrix: np.ndarray, distortion_coefficients: np.ndarray):
        super().__init__(previous_block)

        self.intrinsic_matrix = intrinsic_matrix
        self.distortion_coefficients = distortion_coefficients

    def __getitem__(self, index):
        previous_block_result = self.previous_block[index]

        image = previous_block_result["image"]

        undistorted_image = cv2.undistort(image, self.intrinsic_matrix, self.distortion_coefficients, None)

        previous_block_result["image"] = undistorted_image

        return previous_block_result


class BlackLevelCorrectionBlock(PipelineBlock):
    def __getitem__(self, index):
        previous_block_result = self.previous_block[index]

        image = previous_block_result["image"]
        black_level = previous_block_result["black_level"]

        black_level_mask = create_black_level_mask(image.shape, black_level)

        corrected_image = image.astype(np.float32) - black_level_mask
        corrected_image[corrected_image < 0] = 0

        previous_block_result["image"] = corrected_image

        return previous_block_result


class NormalizationBlock(PipelineBlock):
    def __getitem__(self, index):
        previous_block_result = self.previous_block[index]

        image = previous_block_result["image"]
        white_level = previous_block_result["white_level"]
        black_level = previous_block_result["black_level"]

        black_level_mask = create_black_level_mask(image.shape, black_level)
        normalized_image = image.astype(np.float32) / (white_level - black_level_mask)

        previous_block_result["image"] = normalized_image

        return previous_block_result


class DemosaicBlock(PipelineBlock):
    def __getitem__(self, index):
        previous_block_result = self.previous_block[index]

        image = previous_block_result["image"]
        cfa_pattern = previous_block_result["cfa_pattern"]

        raw_colors = np.asarray(cfa_pattern).reshape((2, 2))
        demosaiced_image = np.zeros((image.shape[0] // 2, image.shape[1] // 2, 3))
        for i in range(2):
            for j in range(2):
                ch = raw_colors[i, j]
                if ch == 1:
                    demosaiced_image[:, :, ch] += image[i::2, j::2] / 2
                else:
                    demosaiced_image[:, :, ch] = image[i::2, j::2]

        previous_block_result["image"] = demosaiced_image

        return previous_block_result


class AutoWhiteBalanceBlock(PipelineBlock):
    def __init__(
        self,
        previous_block: PipelineBlock,
        illumination_estimation_option: str | None = None,
    ):
        super().__init__(previous_block)

        self.illumination_estimation_option = illumination_estimation_option

    def __getitem__(self, index):
        previous_block_result = self.previous_block[index]

        image = previous_block_result["image"]
        as_shot_neutral = previous_block_result["as_shot_neutral"]

        if self.illumination_estimation_option is None:
            white_balance_params = as_shot_neutral
        else:
            white_balance_params = self.illumination_parameters_estimation(
                image, self.illumination_estimation_option
            )
        white_balance_params = np.asarray(white_balance_params)

        if white_balance_params.shape == (3,):
            white_balance_params = np.diag(1.0 / white_balance_params)

        if white_balance_params.shape != (3, 3):
            raise ValueError(
                "White balance parameters must be either a list of 3 values or a 3x3 matrix."
            )

        white_balanced_image = np.dot(image, white_balance_params.T)
        white_balanced_image = np.clip(white_balanced_image, 0.0, 1.0)

        previous_block_result["image"] = white_balanced_image

        return previous_block_result

    def illumination_parameters_estimation(
        self, image: np.ndarray, illumination_estimation_option: str
    ):
        ie_method = illumination_estimation_option.lower()
        if ie_method == "gray_world":
            ie = np.mean(image, axis=(0, 1))
            ie /= ie[1]
            return ie
        if ie_method == "shades_of_gray":
            sog_p = 4.0
            ie = np.mean(image**sog_p, axis=(0, 1)) ** (1 / sog_p)
            ie /= ie[1]
            return ie
        if ie_method == "white_patch":
            ie = np.max(image, axis=(0, 1))
            ie /= ie[1]
            return ie
        if ie_method == "iterative_white_patch":
            samples_count = 20
            sample_size = 20
            rows, cols = image.shape[:2]
            data = np.reshape(image, (rows * cols, 3))
            maxima = np.zeros((samples_count, 3))
            for i in range(samples_count):
                maxima[i, :] = np.max(
                    data[
                        np.random.randint(low=0, high=rows * cols, size=(sample_size)),
                        :,
                    ],
                    axis=0,
                )
            ie = np.mean(maxima, axis=0)
            ie /= ie[1]
            return ie

        raise ValueError(
            'Bad illumination_estimation_option value! Use the following options: "gray_world", "white_patch", "shades_of_gray", "iterative_white_patch"'
        )


class XYZTransformBlock(PipelineBlock):
    def __init__(self, previous_block: PipelineBlock, color_matrix: np.ndarray):
        super().__init__(previous_block)

        if color_matrix.shape != (3, 3):
            raise ValueError("Color matrix must be a 3x3 matrix.")

        normalized_matrix = color_matrix / np.sum(color_matrix, axis=1, keepdims=True)
        self.cam2xyz_matrix = np.ascontiguousarray(
            np.linalg.inv(normalized_matrix).astype(np.float32)
        )

    def __getitem__(self, index):
        previous_block_result = self.previous_block[index]

        image = previous_block_result["image"]

        xyz_transformed_image = image @ self.cam2xyz_matrix.T
        xyz_transformed_image = np.clip(xyz_transformed_image, 0.0, 1.0)

        previous_block_result["image"] = xyz_transformed_image

        return previous_block_result


class SRGBTransformBlock(PipelineBlock):
    def __init__(self, previous_block: PipelineBlock, xyz_to_srgb_matrix: np.ndarray):
        super().__init__(previous_block)

        if xyz_to_srgb_matrix.shape != (3, 3):
            raise ValueError("XYZ to sRGB matrix must be a 3x3 matrix.")

        normalized_matrix = xyz_to_srgb_matrix / np.sum(
            xyz_to_srgb_matrix, axis=-1, keepdims=True
        )
        self.xyz_to_srgb_matrix = np.ascontiguousarray(
            normalized_matrix.astype(np.float32)
        )

    def __getitem__(self, index):
        previous_block_result = self.previous_block[index]

        image = previous_block_result["image"]

        srgb_transformed_image = image @ self.xyz_to_srgb_matrix.T
        srgb_transformed_image = np.clip(srgb_transformed_image, 0.0, 1.0)

        previous_block_result["image"] = srgb_transformed_image

        return previous_block_result


class GammaCorrectionBlock(PipelineBlock):
    def __getitem__(self, index):
        previous_block_result = self.previous_block[index]

        image = previous_block_result["image"]

        gamma_corrected_image = image.copy()
        idx = gamma_corrected_image <= 0.0031308
        gamma_corrected_image[idx] *= 12.92
        gamma_corrected_image[idx == False] = (
            gamma_corrected_image[idx == False] ** (1.0 / 2.4)
        ) * 1.055 - 0.055

        previous_block_result["image"] = gamma_corrected_image

        return previous_block_result


class SpatialTransformBlock(PipelineBlock):
    def __init__(
        self,
        previous_block: PipelineBlock,
        transform_matrix: np.ndarray,
        orientation_str2id: dict | None = None,
        pretransform_resize_factor: int = 4,
        resize_shape_before_transform: tuple[int, int] = (3072, 4096),
        resize_shape_before_crop: tuple[int, int] = (2654, 3538),
        intrinsic_matrix: np.ndarray | None = None,
        distortion_coefficients: np.ndarray | None = None,
    ):
        super().__init__(previous_block)

        if transform_matrix.shape != (3, 3):
            raise ValueError("Transform matrix must be a 3x3 matrix.")
        if pretransform_resize_factor <= 0:
            raise ValueError("Pretransform resize factor must be a positive number.")

        self.transform_matrix = transform_matrix

        if orientation_str2id is not None:
            self.orientation_str2id = orientation_str2id
        else:
            self.orientation_str2id = {
                "Horizontal (normal)": 1,
                "Mirror horizontal": 2,
                "Rotate 180": 3,
                "Mirror vertical": 4,
                "Mirror horizontal and rotate 270 CW": 5,
                "Rotate 90 CW": 6,
                "Mirror horizontal and rotate 90 CW": 7,
                "Rotate 270 CW": 8,
            }

        self.pretransform_resize_factor = pretransform_resize_factor
        self.resize_shape_before_transform = resize_shape_before_transform
        self.resize_shape_before_crop = resize_shape_before_crop
        self.intrinsic_matrix = intrinsic_matrix
        self.distortion_coefficients = distortion_coefficients

    def __getitem__(self, index):
        previous_block_result = self.previous_block[index]

        image = previous_block_result["image"]
        orientation = previous_block_result["orientation"]
        crop_bounds = previous_block_result["huawei_bounds"]

        orientation_corrected_image = self.fix_orientation(image, orientation)
        perspective_transformed_image = self.projective_transform(
            orientation_corrected_image
        )
        flipped_image = cv2.flip(perspective_transformed_image, 1)

        if self.intrinsic_matrix is not None and self.distortion_coefficients is not None:
            flipped_image = cv2.undistort(flipped_image, self.intrinsic_matrix, self.distortion_coefficients, None)

        cropped_image = self.crop_resize(flipped_image, crop_bounds)

        previous_block_result["image"] = cropped_image

        return previous_block_result

    def fix_orientation(self, image, orientation):
        # 1 = Horizontal(normal)
        # 2 = Mirror horizontal
        # 3 = Rotate 180
        # 4 = Mirror vertical
        # 5 = Mirror horizontal and rotate 270 CW
        # 6 = Rotate 90 CW
        # 7 = Mirror horizontal and rotate 90 CW
        # 8 = Rotate 270 CW

        if type(orientation) is str:
            orientation = self.orientation_str2id[orientation]
        elif type(orientation) is list:
            orientation = orientation[0]

        if orientation == 1:
            pass
        elif orientation == 2:
            image = cv2.flip(image, 0)
        elif orientation == 3:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif orientation == 4:
            image = cv2.flip(image, 1)
        elif orientation == 5:
            image = cv2.flip(image, 0)
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif orientation == 6:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == 7:
            image = cv2.flip(image, 0)
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == 8:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return image

    def projective_transform(self, image):
        S = np.array(
            [
                [1 / self.pretransform_resize_factor, 0, 0],
                [0, 1 / self.pretransform_resize_factor, 0],
                [0, 0, 1],
            ]
        )
        scaled_transform_matrix = self.transform_matrix @ S
        transformed_image = cv2.warpPerspective(
            image, scaled_transform_matrix, self.resize_shape_before_transform
        )
        return transformed_image

    def crop_resize(self, image, crop_bounds, upper_bound=200):
        resized_image = cv2.resize(image, self.resize_shape_before_crop)

        # Crop by bounds
        h_start, h_end, w_start, w_end = crop_bounds
        cropped_image = resized_image[h_start:h_end, w_start:w_end]

        # Upper crop
        crop_mid = cropped_image.shape[1] // 2
        cropped_image = cropped_image[
            upper_bound : 2000 + upper_bound, crop_mid - 1000 : crop_mid + 1000
        ]
        return cropped_image


class JDDBlock(PipelineBlock):
    def __init__(self, previous_block: PipelineBlock, model: torch.nn.Module, device: str = "cpu"):
        super().__init__(previous_block)

        self.device = device
        self.model = model.to(self.device)

    def _release_memory(self) -> None:
        if torch.device(self.device).type == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.device(self.device).type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def __getitem__(self, index):
        previous_block_result = self.previous_block[index]

        image = previous_block_result["image"]
        input_tensor = torch.from_numpy(image).float().to(self.device)
        r = input_tensor[0::2, 0::2]
        gr = input_tensor[0::2, 1::2]
        gb = input_tensor[1::2, 0::2]
        b = input_tensor[1::2, 1::2]
        input_tensor = torch.stack((r, gr, gb, b), dim=0)

        out = self.model(input_tensor.unsqueeze(0))
        output_image = (
            out.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
        )
        del out
        self._release_memory()

        previous_block_result["image"] = output_image

        return previous_block_result


class RawSpatialTransformBlock(PipelineBlock):
    """
    SpatialTransformBlock applied to a Bayer mosaic instead of an RGB image.

    Runs the same operations and produces the same (2000, 2000) output size as
    SpatialTransformBlock, but the geometry is applied to each CFA plane
    separately, so INTER_LINEAR never averages samples of different colors.

    The geometry moves the CFA pattern, so the output layout is not always
    textbook RGGB: a 90 degree rotation swaps the two greens. R stays in the
    top-left slot and B in the bottom-right for every orientation, and the
    layout depends only on the orientation, so JDDBlock stays consistent as
    long as it is trained on this same layout.
    """

    CFA_PHASES = ((0, 0), (0, 1), (1, 0), (1, 1))

    def __init__(
        self,
        previous_block: PipelineBlock,
        transform_matrix: np.ndarray,
        orientation_str2id: dict | None = None,
        pretransform_resize_factor: int = 4,
        resize_shape_before_transform: tuple[int, int] = (3072, 4096),
        resize_shape_before_crop: tuple[int, int] = (2654, 3538),
        intrinsic_matrix: np.ndarray | None = None,
        distortion_coefficients: np.ndarray | None = None,
        output_size: int = 2000,
        upper_bound: int = 200,
        align_cfa_phase: bool = True,
    ):
        super().__init__(previous_block)

        if transform_matrix.shape != (3, 3):
            raise ValueError("Transform matrix must be a 3x3 matrix.")
        if pretransform_resize_factor <= 0:
            raise ValueError("Pretransform resize factor must be a positive number.")
        if output_size <= 0:
            raise ValueError("Output size must be a positive number.")
        if resize_shape_before_crop[0] % 2 or resize_shape_before_crop[1] % 2:
            raise ValueError("resize_shape_before_crop must have even dimensions.")

        self.transform_matrix = transform_matrix

        if orientation_str2id is not None:
            self.orientation_str2id = orientation_str2id
        else:
            self.orientation_str2id = {
                "Horizontal (normal)": 1,
                "Mirror horizontal": 2,
                "Rotate 180": 3,
                "Mirror vertical": 4,
                "Mirror horizontal and rotate 270 CW": 5,
                "Rotate 90 CW": 6,
                "Mirror horizontal and rotate 90 CW": 7,
                "Rotate 270 CW": 8,
            }

        self.pretransform_resize_factor = pretransform_resize_factor
        self.resize_shape_before_transform = resize_shape_before_transform
        self.resize_shape_before_crop = resize_shape_before_crop
        self.intrinsic_matrix = intrinsic_matrix
        self.distortion_coefficients = distortion_coefficients
        self.output_size = output_size
        self.upper_bound = upper_bound
        self.align_cfa_phase = align_cfa_phase

    def __getitem__(self, index):
        previous_block_result = self.previous_block[index]

        image = previous_block_result["image"]
        orientation = previous_block_result["orientation"]
        crop_bounds = previous_block_result["huawei_bounds"]

        if image.ndim != 2:
            raise ValueError("RawSpatialTransformBlock expects a 2D Bayer mosaic.")
        if image.shape[0] % 2 or image.shape[1] % 2:
            raise ValueError("Bayer mosaic must have even height and width.")

        # Planes are keyed by their CFA color (the original phase). The current
        # phase of each plane is tracked separately because the geometry moves
        # the CFA phase around.
        planes = {phase: image[phase[0] :: 2, phase[1] :: 2].copy() for phase in self.CFA_PHASES}
        phases = {phase: phase for phase in self.CFA_PHASES}

        planes, phases = self.fix_orientation(planes, phases, orientation)
        planes = self.projective_transform(planes, phases)
        planes, phases = self.flip(planes, phases)

        if self.intrinsic_matrix is not None and self.distortion_coefficients is not None:
            planes = {
                color: cv2.undistort(
                    plane, self.intrinsic_matrix, self.distortion_coefficients, None
                )
                for color, plane in planes.items()
            }

        planes, crop_phases = self.crop_resize(planes, phases, crop_bounds)

        previous_block_result["image"] = self.repack(planes, crop_phases)

        return previous_block_result

    def _orientation_operation(self, orientation):
        # 1 = Horizontal(normal)
        # 2 = Mirror horizontal
        # 3 = Rotate 180
        # 4 = Mirror vertical
        # 5 = Mirror horizontal and rotate 270 CW
        # 6 = Rotate 90 CW
        # 7 = Mirror horizontal and rotate 90 CW
        # 8 = Rotate 270 CW

        if type(orientation) is str:
            orientation = self.orientation_str2id[orientation]
        elif type(orientation) is list:
            orientation = orientation[0]

        if orientation == 1:
            return "none"
        if orientation == 2:
            return "flip_vertical"
        if orientation == 3:
            return "rotate_180"
        if orientation == 4:
            return "flip_horizontal"
        if orientation == 5:
            return "flip_vertical_rotate_90_ccw"
        if orientation == 6:
            return "rotate_90_cw"
        if orientation == 7:
            return "flip_vertical_rotate_90_cw"
        if orientation == 8:
            return "rotate_90_ccw"

        raise ValueError(f"Unknown orientation: {orientation}")

    @staticmethod
    def _apply_operation(array, operation):
        if operation == "none":
            return array
        if operation == "flip_vertical":
            return cv2.flip(array, 0)
        if operation == "flip_horizontal":
            return cv2.flip(array, 1)
        if operation == "rotate_180":
            return cv2.rotate(array, cv2.ROTATE_180)
        if operation == "rotate_90_cw":
            return cv2.rotate(array, cv2.ROTATE_90_CLOCKWISE)
        if operation == "rotate_90_ccw":
            return cv2.rotate(array, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if operation == "flip_vertical_rotate_90_cw":
            return cv2.rotate(cv2.flip(array, 0), cv2.ROTATE_90_CLOCKWISE)
        if operation == "flip_vertical_rotate_90_ccw":
            return cv2.rotate(cv2.flip(array, 0), cv2.ROTATE_90_COUNTERCLOCKWISE)

        raise ValueError(f"Unknown operation: {operation}")

    def fix_orientation(self, planes, phases, orientation):
        operation = self._orientation_operation(orientation)

        # The new phase of every color is derived by running the same operation
        # on a 2x2 mosaic of color labels, so all 8 orientations stay correct.
        labels = np.arange(4, dtype=np.int32).reshape(2, 2)
        moved = self._apply_operation(labels, operation)

        new_phases = {}
        for dy in range(2):
            for dx in range(2):
                color_index = int(moved[dy, dx])
                new_phases[(color_index // 2, color_index % 2)] = (dy, dx)

        planes = {
            color: self._apply_operation(plane, operation) for color, plane in planes.items()
        }

        return planes, new_phases

    def projective_transform(self, planes, phases):
        S = np.array(
            [
                [1 / self.pretransform_resize_factor, 0, 0],
                [0, 1 / self.pretransform_resize_factor, 0],
                [0, 0, 1],
            ]
        )
        scaled_transform_matrix = self.transform_matrix @ S

        transformed = {}
        for phase, plane in planes.items():
            dy, dx = phases[phase]
            # Plane sample (i, j) sits at mosaic (2i + dy, 2j + dx), so the
            # 1/2 subsampling is undone inside the matrix per plane.
            offset = np.array(
                [[2, 0, dx], [0, 2, dy], [0, 0, 1]],
                dtype=float,
            )
            transformed[phase] = cv2.warpPerspective(
                plane,
                scaled_transform_matrix @ offset,
                self.resize_shape_before_transform,
            )

        return transformed

    def flip(self, planes, phases):
        flipped = {}
        new_phases = {}

        for phase, plane in planes.items():
            dy, dx = phases[phase]
            new_phases[phase] = (dy, 1 - dx)
            flipped[phase] = cv2.flip(plane, 1)

        return flipped, new_phases

    def crop_resize(self, planes, phases, crop_bounds):
        resize_shape = (self.resize_shape_before_crop[0] // 2, self.resize_shape_before_crop[1] // 2)

        h_start, h_end, w_start, w_end = crop_bounds
        crop_mid = (w_end - w_start) // 2

        r0 = h_start + self.upper_bound
        c0 = w_start + crop_mid - self.output_size // 2

        if self.align_cfa_phase:
            # Align the crop origin with the current phase of R, so R lands in
            # the top-left slot of the packed RGGB mosaic regardless of how the
            # geometry moved the CFA pattern. One pixel of 2000 at most.
            r_dy, r_dx = phases[self.CFA_PHASES[0]]
            r0 += (r_dy - r0) % 2
            c0 += (r_dx - c0) % 2

        r1, c1 = r0 + self.output_size, c0 + self.output_size

        cropped = {}
        crop_phases = {}

        for color, plane in planes.items():
            resized = cv2.resize(plane, resize_shape)
            dy, dx = phases[color]

            row_start = int(np.ceil((r0 - dy) / 2))
            row_end = int(np.ceil((r1 - dy) / 2))
            col_start = int(np.ceil((c0 - dx) / 2))
            col_end = int(np.ceil((c1 - dx) / 2))

            patch = resized[row_start:row_end, col_start:col_end]

            if patch.shape != (self.output_size // 2, self.output_size // 2):
                raise ValueError(
                    f"Crop window {crop_bounds} with upper_bound={self.upper_bound} "
                    f"does not fit inside the resized mosaic for color {color}: "
                    f"got {patch.shape}."
                )

            cropped[color] = patch
            crop_phases[color] = ((dy - r0) % 2, (dx - c0) % 2)

        return cropped, crop_phases

    @staticmethod
    def repack(planes, phases):
        output_size = next(iter(planes.values())).shape[0] * 2
        mosaic = np.empty((output_size, output_size), dtype=next(iter(planes.values())).dtype)

        for color, plane in planes.items():
            dy, dx = phases[color]
            mosaic[dy::2, dx::2] = plane

        return mosaic
