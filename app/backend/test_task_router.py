from __future__ import annotations

import unittest

from app.expert_router import route_tasks


SINGLE_TASK_CASES = {
    "semantic_segmentation": (
        "请把建筑物分割出来",
        "生成道路区域的二值掩膜",
        "将水体从背景中分离出来",
        "抠出图中的森林区域",
        "给农田做像素级标注",
        "请输出裸地的语义图",
        "把所有地物类别逐像素分开",
        "完成整幅影像的多类别分割",
        "Segment the buildings in this image.",
        "Generate a binary mask for roads.",
        "Delineate the water bodies.",
        "Return a full-scene multiclass segmentation.",
    ),
    "object_counting": (
        "图中一共有多少架飞机？",
        "这里有几栋建筑物？",
        "帮我数一下停车场里的小汽车",
        "统计图中船舶的数量",
        "车辆总数是多少？",
        "请数一数画面里的储油罐",
        "清点所有能看到的卡车",
        "这些房屋的个数是多少？",
        "How many ships are visible?",
        "Count the aircraft in the image.",
        "What is the number of storage tanks?",
        "Tally all visible vehicles.",
    ),
    "grounding": (
        "大型机场在哪里？",
        "请定位图中的灰色体育场",
        "框出左侧的船只",
        "把红色油罐车圈出来",
        "指出高速公路收费站的位置",
        "找出建筑旁边的白色车辆",
        "标记画面中所有飞机",
        "距离港口最近的船是哪一艘？",
        "给我返回棒球场的坐标",
        "Locate the train station.",
        "Where is the baseball field?",
        "Draw a bounding box around the nearest ship.",
        "Find all red vehicles.",
    ),
    "change_detection": (
        "描述两个时相之间的变化",
        "这两张影像有什么不同？",
        "比较前后影像中的变化区域",
        "哪些建筑是新建的？",
        "找出灾前灾后的差异",
        "检测图中新增和拆除的房屋",
        "两期遥感影像进行比较",
        "前一张和后一张相比发生了什么？",
        "Describe the changes between the two images.",
        "What has changed between these scenes?",
        "Compare the before and after images.",
        "Identify newly built structures.",
    ),
    "scene_classification": (
        "这张图属于什么场景？",
        "判断遥感影像的场景类型",
        "这里主要是哪类区域？",
        "识别图中的土地利用类型",
        "这是城市还是农村？",
        "判断该区域的地貌类型",
        "这是什么类型的遥感场景？",
        "What type of landscape is shown?",
        "Classify the scene type.",
        "Is this area urban or rural?",
        "What kind of area surrounds the airport?",
        "Determine the land-use type.",
    ),
    "image_captioning": (
        "请详细描述这张遥感影像",
        "描述一下图中的主要地物",
        "介绍这幅卫星图的内容",
        "概述图中可以看到的东西",
        "图中有什么？",
        "这张图包含哪些地物？",
        "说说这幅航拍影像",
        "讲讲图片里的主要场景",
        "Describe this remote-sensing image in detail.",
        "Caption this image.",
        "What can you see in the aerial image?",
        "Give an overview of this satellite image.",
    ),
}


class TaskRouterSingleIntentTest(unittest.TestCase):
    def test_single_task_phrasings(self) -> None:
        for expected, queries in SINGLE_TASK_CASES.items():
            for query in queries:
                with self.subTest(expected=expected, query=query):
                    decision = route_tasks(query)
                    self.assertEqual(decision.tasks, (expected,))
                    self.assertEqual(decision.plans[0].prompt, query)

    def test_general_vqa_falls_through(self) -> None:
        queries = (
            "道路是什么颜色？",
            "图中是否存在飞机？",
            "白色车辆旁边是什么？",
            "这栋建筑大概有多高？",
            "河流的流向是什么？",
            "Is there a ship in the image?",
            "What color is the roof?",
            "Are the vehicles parked in a row?",
            "Which object is larger?",
            "回答我一个关于遥感的问题",
        )
        for query in queries:
            with self.subTest(query=query):
                decision = route_tasks(query)
                self.assertEqual(decision.tasks, ())
                self.assertIn("no_rule_match", decision.warnings)


class TaskRouterCompoundIntentTest(unittest.TestCase):
    def test_compound_queries(self) -> None:
        cases = (
            (
                "请分割图中汽车并且数一下有多少个",
                ("semantic_segmentation", "object_counting"),
            ),
            (
                "先数一下飞机，然后定位最靠右的那一架",
                ("object_counting", "grounding"),
            ),
            (
                "定位所有船只并统计它们的数量",
                ("grounding", "object_counting"),
            ),
            (
                "描述整幅影像，同时判断场景类型",
                ("image_captioning", "scene_classification"),
            ),
            (
                "比较两个时相的变化，并统计新增建筑数量",
                ("change_detection", "object_counting"),
            ),
            (
                "Segment all buildings and then count them.",
                ("semantic_segmentation", "object_counting"),
            ),
            (
                "Locate every aircraft; also tell me how many there are.",
                ("grounding", "object_counting"),
            ),
        )
        for query, expected in cases:
            with self.subTest(query=query):
                decision = route_tasks(query)
                self.assertEqual(decision.tasks, expected)
                self.assertTrue(decision.is_compound)
                self.assertIn("compound_execution_required", decision.warnings)
                self.assertEqual(len(decision.plans), len(expected))

    def test_targetless_second_clause_keeps_original_request(self) -> None:
        query = "先定位所有飞机，然后数一下有多少个"
        decision = route_tasks(query)
        counting = next(plan for plan in decision.plans if plan.task == "object_counting")
        self.assertIn(query, counting.prompt)
        self.assertIn("计数任务", counting.prompt)
        self.assertIn("aircraft", counting.targets)


class TaskRouterNegationAndConstraintTest(unittest.TestCase):
    def test_negated_intent_does_not_route(self) -> None:
        cases = (
            ("不要分割建筑，请详细描述这张图", ("image_captioning",)),
            ("不用计数，只要定位车辆", ("grounding",)),
            ("不要定位，只统计飞机数量", ("object_counting",)),
            ("不要描述，判断场景类型", ("scene_classification",)),
            ("无需比较变化，直接描述这张影像", ("image_captioning",)),
            ("Don't count the cars; describe the image.", ("image_captioning",)),
        )
        for query, expected in cases:
            with self.subTest(query=query):
                self.assertEqual(route_tasks(query).tasks, expected)

    def test_image_count_warnings(self) -> None:
        change = route_tasks("比较两个时相之间的变化", image_count=1)
        self.assertIn("change_detection_expected_two_images", change.warnings)
        self.assertNotIn(
            "change_detection_expected_two_images",
            route_tasks("比较两个时相之间的变化", image_count=2).warnings,
        )
        segmentation = route_tasks("分割图中的建筑", image_count=2)
        self.assertIn("segmentation_expert_requires_one_image", segmentation.warnings)

    def test_dense_counting_subtype(self) -> None:
        dense_queries = (
            "精确统计图中所有建筑物的数量",
            "这片密集建筑群一共有多少栋？",
            "Count every building in this crowded area.",
        )
        for query in dense_queries:
            with self.subTest(query=query):
                plan = route_tasks(query).plans[0]
                self.assertEqual(plan.task, "object_counting")
                self.assertEqual(plan.subtype, "dense_candidate")
        ordinary = route_tasks("图中有多少架飞机？").plans[0]
        self.assertEqual(ordinary.subtype, "standard")

    def test_segmentation_details(self) -> None:
        targeted = route_tasks("请分割建筑、道路和水体")
        self.assertEqual(targeted.plans[0].targets, ("building", "road", "water"))
        self.assertEqual(targeted.plans[0].subtype, "targeted")

        all_classes = route_tasks("对整幅图进行所有类别语义分割")
        self.assertEqual(all_classes.plans[0].subtype, "all_classes")

        unsupported = route_tasks("请分割车辆和飞机")
        self.assertIn("segmentation_unsupported_target:vehicle", unsupported.warnings)
        self.assertIn("segmentation_unsupported_target:aircraft", unsupported.warnings)

    def test_fine_grained_remote_sensing_targets(self) -> None:
        cases = (
            ("定位图中的小汽车", ("small_vehicle",)),
            ("找出所有红色油罐车", ("large_vehicle",)),
            ("大型高速公路收费站在哪里", ("expressway_toll_station",)),
            ("请框出机场跑道", ("runway",)),
            ("定位左侧的棒球场", ("baseball_field",)),
            ("高尔夫球场位于哪里", ("golf_course",)),
            ("找出图中的火车站", ("train_station",)),
            ("标记所有风力发电机", ("windmill",)),
        )
        for query, expected in cases:
            with self.subTest(query=query):
                self.assertEqual(route_tasks(query).plans[0].targets, expected)


if __name__ == "__main__":
    unittest.main()
