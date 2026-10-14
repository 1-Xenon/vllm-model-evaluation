import json
import unittest

from vllm_evaluation.bounding_boxes import extract_bounding_boxes


class BoundingBoxTests(unittest.TestCase):
    def test_aliases_numeric_strings_and_subject_types(self):
        for key in (
            "bbox",
            "bbox_2d",
            "boundingBox",
            "bounding_box",
            "Bounding Box",
            "bouding_box",
            "box2D",
        ):
            with self.subTest(key=key):
                result = extract_bounding_boxes(
                    json.dumps({"who": [{"type": "person", key: "[0.1, 0.2, 0.6, 0.9]"}]})
                )
                self.assertEqual(len(result["boxes"]), 1)
                self.assertEqual(result["boxes"][0]["label"], "person")
                self.assertEqual(result["boxes"][0]["coordinates"], [0.1, 0.2, 0.6, 0.9])
                self.assertEqual(result["warnings"], [])

    def test_markdown_embedded_json_and_loose_lines(self):
        examples = [
            '```json\n{"type":"equipment","bbox_2d":[10,20,30,40]}\n```',
            'Here is the result: {"type":"equipment","bbox_2d":[10,20,30,40]} Done.',
            "{'type': 'equipment', 'bbox_2d': (10, 20, 30, 40)}",
            'type: equipment, bbox_2d: "[10, 20, 30, 40]"',
        ]
        for answer in examples:
            with self.subTest(answer=answer):
                result = extract_bounding_boxes(answer)
                self.assertEqual(len(result["boxes"]), 1)
                self.assertEqual(result["boxes"][0]["label"], "equipment")
                self.assertEqual(result["boxes"][0]["coordinates"], [10, 20, 30, 40])

    def test_named_coordinates_formats_units_and_image_references(self):
        result = extract_bounding_boxes(
            json.dumps(
                {
                    "images": [
                        {
                            "image_index": 0,
                            "coordinate_system": "normalized",
                            "type": "animal",
                            "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.8, "y_max": 0.9},
                        },
                        {
                            "image_number": 2,
                            "type": "equipment",
                            "bbox": {"x": 10, "y": 20, "width": 20, "height": 30},
                        },
                        {
                            "image_path": "media/third.png",
                            "type": "person",
                            "bbox_format": "cxcywh",
                            "coordinate_scale": "0-1000",
                            "bbox": [500, 500, 200, 400],
                        },
                    ]
                }
            )
        )
        a, b, c = result["boxes"]
        self.assertEqual(a["unit"], "normalized")
        self.assertEqual(a["image"], {"index": 0})
        self.assertEqual(b["coordinates"], [10, 20, 30, 50])
        self.assertEqual(b["image"], {"index": 1})
        self.assertEqual(c["coordinates"], [400, 300, 600, 700])
        self.assertEqual(c["unit"], "1000")
        self.assertEqual(c["image"], {"path": "media/third.png"})

    def test_nested_box_metadata_and_multiple_boxes(self):
        result = extract_bounding_boxes(
            json.dumps(
                {
                    "type": "person",
                    "bboxes": [
                        [1, 2, 3, 4],
                        [10, 20, 30, 40],
                        {
                            "type": "equipment",
                            "bbox": {"coordinates": [100, 200, 300, 400], "units": "pixels"},
                        },
                    ],
                }
            )
        )
        self.assertEqual(
            [box["label"] for box in result["boxes"]], ["person", "person", "equipment"]
        )
        self.assertEqual(result["boxes"][2]["unit"], "pixels")

    def test_invalid_boxes_are_skipped_without_inventing_coordinates(self):
        result = extract_bounding_boxes(
            json.dumps(
                {
                    "who": [
                        {"bbox": None},
                        {"bbox": ""},
                        {"bbox": [1, 2]},
                        {"bbox": [4, 3, 2, 1]},
                        {"bbox": [True, 0, 1, 1]},
                        {"bbox": "[NaN, 0, 1, 1]"},
                        {"bbox": "unknown"},
                    ]
                }
            )
        )
        self.assertEqual(result["boxes"], [])
        self.assertIn("Skipped 5", result["warnings"][0])
        self.assertEqual(extract_bounding_boxes("No boxes.")["boxes"], [])
        self.assertEqual(extract_bounding_boxes(None)["boxes"], [])

    def test_list_of_named_boxes_preserves_each_label(self):
        result = extract_bounding_boxes(
            json.dumps(
                {
                    "bounding_boxes": [
                        {"type": "person", "x_min": 10, "y_min": 20, "x_max": 30, "y_max": 40},
                        {"type": "animal", "x_min": 40, "y_min": 50, "x_max": 60, "y_max": 70},
                    ]
                }
            )
        )
        self.assertEqual([box["label"] for box in result["boxes"]], ["person", "animal"])

    def test_does_not_treat_unrelated_arrays_as_boxes_or_execute_text(self):
        answer = '{"scores":[0,0,1,1],"coordinates":[1,2,3,4],"bbox_format":"xyxy"}'
        self.assertEqual(extract_bounding_boxes(answer)["boxes"], [])
        self.assertEqual(extract_bounding_boxes('__import__("os").system("false")')["boxes"], [])

    def test_box_count_and_nesting_are_bounded(self):
        result = extract_bounding_boxes(json.dumps([{"bbox": [0, 0, 1, 1]}] * 250))
        self.assertEqual(len(result["boxes"]), 200)
        self.assertIn("Showing at most 200 boxes.", result["warnings"])
        self.assertEqual(extract_bounding_boxes("[" * 2000 + "]" * 2000)["boxes"], [])
