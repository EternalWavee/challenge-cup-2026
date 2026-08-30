from __future__ import annotations

import unittest

from app.expert_router import route_segmentation, route_segmentation_token


class ExpertRouterTest(unittest.TestCase):
    def test_targeted_segmentation(self) -> None:
        route = route_segmentation("请把图中的建筑和道路分割出来")
        self.assertIsNotNone(route)
        self.assertEqual(route.mode, "targeted")
        self.assertEqual(route.targets, ("building", "road"))
        self.assertEqual(route.unsupported, ())

    def test_all_class_segmentation(self) -> None:
        for query in (
            "请对整张图进行全类别语义分割",
            "请对这张遥感影像进行语义分割",
            "Generate semantic segmentation for this image.",
        ):
            with self.subTest(query=query):
                route = route_segmentation(query)
                self.assertIsNotNone(route)
                self.assertEqual(route.mode, "all_classes")
                self.assertEqual(route.targets, ())

    def test_unsupported_target(self) -> None:
        route = route_segmentation("请生成车辆的二值掩膜")
        self.assertIsNotNone(route)
        self.assertEqual(route.targets, ())
        self.assertEqual(route.unsupported, ("vehicle",))

    def test_non_segmentation_falls_through(self) -> None:
        for query in (
            "图中一共有多少辆汽车？",
            "请定位图中的建筑。",
            "请描述这张遥感图像。",
            "不要分割建筑，请描述图像。",
            "请分割建筑并统计有多少栋。",
        ):
            with self.subTest(query=query):
                self.assertIsNone(route_segmentation(query))

    def test_model_targeted_token_uses_original_query(self) -> None:
        route = route_segmentation_token("<seg>", "麻烦处理一下图里的建筑")
        self.assertIsNotNone(route)
        self.assertEqual(route.mode, "targeted")
        self.assertEqual(route.targets, ("building",))

    def test_model_all_class_token(self) -> None:
        route = route_segmentation_token("<seg_all>。", "帮我处理这张图")
        self.assertIsNotNone(route)
        self.assertEqual(route.mode, "all_classes")

    def test_model_token_must_be_exact(self) -> None:
        self.assertIsNone(
            route_segmentation_token("建议调用 <seg> 完成任务", "分割建筑")
        )


if __name__ == "__main__":
    unittest.main()
