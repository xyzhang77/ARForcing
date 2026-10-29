import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scripts.diversity import eval_diversity as evaluation


class DiversityMetricTest(unittest.TestCase):
    def test_unique_pairs_normalization_and_range(self):
        features = np.array([[[2, 0]], [[0, 3]], [[-4, 0]]])
        result = evaluation.compute_diversity(features)
        self.assertAlmostEqual(result["avg_cosine_similarity"], -1 / 3)
        self.assertAlmostEqual(result["diversity"], 4 / 3)
        self.assertAlmostEqual(evaluation.compute_diversity(features[[0, 2]])["diversity"], 2)

    def test_same_videos_do_not_count_temporal_change_as_diversity(self):
        video = [[1, 0], [0, 1]]
        result = evaluation.compute_diversity([video, video])
        self.assertEqual(result["diversity"], 0)
        self.assertEqual(result["per_frame_diversity"], [0, 0])

    def test_compare_matching_frames_not_time_pooled_features(self):
        result = evaluation.compute_diversity([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])
        self.assertEqual(result["diversity"], 1)
        result = evaluation.compute_diversity([[[1, 0], [0, 1]], [[1, 0], [1, 0]]])
        self.assertEqual(result["per_frame_diversity"], [0, 1])
        self.assertEqual(result["diversity"], 0.5)

    def test_reject_invalid_features_instead_of_reporting_full_diversity(self):
        invalid = [np.ones((1, 2, 3)), np.ones((0, 2, 3)), np.ones((2, 0, 3)),
                   np.ones((2, 1, 0)), np.ones((2, 3)), np.zeros((2, 1, 3)),
                   np.full((2, 1, 3), np.nan), np.full((2, 1, 3), np.inf)]
        for features in invalid:
            with self.subTest(shape=features.shape), self.assertRaises(ValueError):
                evaluation.compute_diversity(features)

    def test_prompt_macro_average(self):
        records = [{"clip": {"avg_cosine_similarity": value},
                    "dino": {"avg_cosine_similarity": value - 1}} for value in (0, 1)]
        summary = evaluation.aggregate_scores(records)
        self.assertEqual(summary["clip"]["diversity"], 0.5)
        self.assertEqual(summary["dino"]["diversity"], 1.5)
        with self.assertRaises(ValueError):
            evaluation.aggregate_scores([])

    def test_reference_feature_selection(self):
        hidden = np.arange(12).reshape(2, 2, 3)
        pooled = np.ones((2, 3))
        outputs = SimpleNamespace(last_hidden_state=hidden, pooler_output=pooled)
        np.testing.assert_array_equal(evaluation.select_features(outputs, "clip"), hidden[:, 0])
        self.assertIs(evaluation.select_features(outputs, "dino"), pooled)
        hidden = Mock()
        for outputs in (SimpleNamespace(last_hidden_state=hidden),
                        SimpleNamespace(last_hidden_state=hidden, pooler_output=None)):
            hidden.reset_mock()
            self.assertIs(evaluation.select_features(outputs, "dino"), hidden.mean.return_value)
            hidden.mean.assert_called_once_with(dim=1)


class DiversityInputTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2])
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def videos(self, names):
        for name in names:
            (self.root / name).touch()
        return evaluation.discover_videos(self.root)

    def test_prompt_hyphens_and_numeric_sample_order(self):
        groups = self.videos(["a-cat-10.mp4", "a-cat-2.mp4", "a-dog-2.mp4", "a-dog-10.mp4"])
        self.assertEqual(list(groups), ["a-cat", "a-dog"])
        self.assertEqual(list(groups["a-cat"]), [2, 10])

    def test_indexed_inference_names(self):
        groups = self.videos(["42-0_regular.mp4", "42-1_regular.mp4"])
        self.assertEqual(list(groups["42"]), [0, 1])

    def test_reject_singletons_mixed_weights_duplicates_and_partial_sets(self):
        cases = [[], ["cat-0.mp4"], ["cat-0.mp4", "cat-00.mp4"],
                 ["42-0_regular.mp4", "42-1_ema.mp4"],
                 ["cat-0.mp4", "cat-1.mp4", "dog-0.mp4", "dog-2.mp4"],
                 ["cat-0.tmp.mp4"], ["unlabelled.mp4"]]
        for index, names in enumerate(cases):
            directory = self.root / str(index)
            directory.mkdir()
            for name in names:
                (directory / name).touch()
            with self.subTest(names=names), self.assertRaises(ValueError):
                evaluation.discover_videos(directory)

    def test_refuse_overwrite_before_loading_models(self):
        self.videos(["cat-0.mp4", "cat-1.mp4"])
        output = self.root / "result.json"
        output.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            evaluation.main(["--video-dir", str(self.root), "--output-path", str(output)])
        self.assertEqual(output.read_text(), "keep")

    def test_uniform_frames_include_endpoints_and_never_repeat(self):
        self.assertEqual(evaluation.uniform_frame_indices(9, 3), [0, 4, 8])
        self.assertEqual(evaluation.uniform_frame_indices(9, 1), [0])
        self.assertEqual(evaluation.uniform_frame_indices(3, 3), [0, 1, 2])
        for total, count in ((0, 1), (5, 0), (2, 3)):
            with self.assertRaises(ValueError):
                evaluation.uniform_frame_indices(total, count)

    def test_sequential_rgb_decode_and_release_on_failure(self):
        cap = Mock()
        cap.get.side_effect = lambda key: {1: 9, 2: 16}[key]
        cap.read.side_effect = [(True, np.array([[[i, i + 1, i + 2]]], dtype=np.uint8)) for i in range(9)]
        cv2 = SimpleNamespace(VideoCapture=Mock(return_value=cap), CAP_PROP_FRAME_COUNT=1,
                              CAP_PROP_FPS=2, COLOR_BGR2RGB=3,
                              cvtColor=lambda frame, code: frame[..., ::-1].copy())
        with patch.dict("sys.modules", {"cv2": cv2}):
            frames, metadata = evaluation.read_video_frames("dummy.mp4", 3)
            self.assertEqual([frame.getpixel((0, 0)) for frame in frames], [(2, 1, 0), (6, 5, 4), (10, 9, 8)])
            self.assertEqual(metadata["frame_indices"], [0, 4, 8])
            for frame in frames:
                frame.close()
            cap.release.assert_called_once()
            cap.release.reset_mock()
            cap.read.side_effect = [(False, None)]
            with self.assertRaisesRegex(ValueError, "Cannot decode"):
                evaluation.read_video_frames("broken.mp4", 3)
            cap.release.assert_called_once()

    def test_prompt_scoring_and_mismatched_video_rejection(self):
        samples = {0: Path("cat-0.mp4"), 1: Path("cat-1.mp4")}
        info = {"total_frames": 3, "fps": 16, "frame_indices": [0, 2]}
        for second_info in (info, {**info, "fps": 24}, {**info, "total_frames": 4}):
            images = [Mock() for _ in range(4)]
            reader = [(images[:2], info), (images[2:], second_info)]
            features = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
            with patch.object(evaluation, "read_video_frames", side_effect=reader), \
                    patch.object(evaluation, "extract_embeddings", return_value=features) as encode:
                scorers = {"clip": (None, None), "dino": (None, None)}
                if second_info != info:
                    with self.assertRaisesRegex(ValueError, "matching lengths and FPS"):
                        evaluation.evaluate_prompt("cat", samples, scorers, 2, 32, "cpu")
                    encode.assert_not_called()
                else:
                    result = evaluation.evaluate_prompt("cat", samples, scorers, 2, 32, "cpu")
                    self.assertEqual(result["clip"]["diversity"], 1)
                    self.assertEqual(result["dino"]["diversity"], 1)
                    self.assertEqual(result["num_pairs"], 1)
            for image in images:
                image.close.assert_called_once()


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is not installed")
class DiversityEncoderTest(unittest.TestCase):
    def test_batching_dtype_and_reference_features(self):
        import torch

        def forward(pixel_values, **kwargs):
            self.assertFalse(torch.is_grad_enabled())
            features = torch.cat([pixel_values, torch.ones_like(pixel_values)], dim=1)
            return SimpleNamespace(last_hidden_state=features[:, None, :], pooler_output=-features)

        model = Mock(side_effect=forward, dtype=torch.float32)
        processor = Mock(side_effect=lambda images, **kwargs: {
            "pixel_values": torch.tensor(images).reshape(-1, 1)})
        expected = np.array([[i, 1] for i in range(5)], dtype=np.float32)
        for kind, sign in (("clip", 1), ("dino", -1)):
            processor.reset_mock()
            result = evaluation.extract_embeddings(list(range(5)), model, processor, kind, 2, "cpu")
            np.testing.assert_array_equal(result, sign * expected)
            self.assertEqual(result.dtype, np.float32)
            self.assertEqual([len(call.kwargs["images"]) for call in processor.call_args_list], [2, 2, 1])


if __name__ == "__main__":
    unittest.main()
