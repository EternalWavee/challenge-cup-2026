from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SUPPORTED_TARGETS = {
    "background": ["背景", "背景区域", "背景掩码", "背景掩膜", "background", "background region", "background mask"],
    "building": ["建筑", "建筑物", "建筑区域", "建筑群", "房屋", "楼房", "厂房", "厂区建筑", "住宅", "民居", "屋顶", "楼顶", "building", "buildings", "house", "houses", "roof", "roofs", "rooftop", "rooftops"],
    "road": ["道路", "公路", "高速公路", "路面", "街道", "车行道", "道路区域", "road", "roads", "street", "streets", "highway", "highways", "road surface"],
    "water": ["水体", "河流", "河道", "湖泊", "湖面", "池塘", "水塘", "水库", "水域", "水面", "海面", "water", "river", "rivers", "lake", "lakes", "pond", "ponds", "reservoir", "water body", "water bodies"],
    "barren": ["裸地", "裸土", "裸露地面", "裸露土壤", "荒地", "未利用地", "bare land", "bare soil", "bare ground", "barren", "wasteland"],
    "forest": ["林地", "林区", "树林", "森林", "树木覆盖区", "树冠", "forest", "forests", "woodland", "woods", "tree cover", "tree canopy"],
    "agricultural": ["农田", "田地", "耕地", "农业用地", "庄稼地", "种植区", "farmland", "agricultural land", "cropland", "cultivated land", "crop field", "crop fields"],
}

UNSUPPORTED_TARGETS = {
    "vegetation": ["植被", "植物覆盖区", "草地", "草坪", "绿地", "灌木", "vegetation", "grassland", "lawn", "shrub", "shrubs"],
    "parking_lot": ["停车场", "停车区域", "停车位", "停车区", "parking lot", "parking lots", "parking area", "car park"],
    "runway": ["机场跑道", "飞行跑道", "跑道", "runway", "runways", "airstrip"],
    "ship": ["船", "船舶", "船只", "舰船", "船艇", "轮船", "货船", "游艇", "ship", "ships", "boat", "boats", "vessel", "vessels"],
    "vehicle": ["车辆", "汽车", "机动车", "vehicle", "vehicles", "automobile", "automobiles"],
    "small_vehicle": ["小型车辆", "小车辆", "小汽车", "轿车", "面包车", "small vehicle", "small vehicles", "car", "cars", "van", "vans"],
    "large_vehicle": ["大型车辆", "大车辆", "卡车", "货车", "油罐车", "巴士", "公交车", "客车", "large vehicle", "large vehicles", "truck", "trucks", "tanker truck", "tanker trucks", "bus", "buses", "coach", "coaches"],
    "bridge": ["桥", "桥梁", "桥面", "bridge", "bridges"],
    "overpass": ["高架桥", "立交桥", "跨线桥", "overpass", "overpasses", "viaduct", "viaducts"],
    "aircraft": ["飞机", "飞行器", "客机", "战斗机", "直升机", "aircraft", "airplane", "airplanes", "plane", "planes", "helicopter", "helicopters"],
    "storage_tank": ["储罐", "储油罐", "油罐", "蓄水罐", "storage tank", "storage tanks", "tank farm"],
    "sports_field": ["体育场地", "运动场地", "球场", "sports field", "sports fields"],
    "baseball_field": ["棒球场", "垒球场", "baseball field", "baseball fields", "softball field"],
    "basketball_court": ["篮球场", "basketball court", "basketball courts"],
    "ground_track_field": ["田径场", "跑道田径场", "ground track field", "ground-track field", "athletics track"],
    "golf_course": ["高尔夫球场", "高尔夫场", "golf course", "golf courses", "golf field"],
    "stadium": ["体育场", "露天体育场", "stadium", "stadiums"],
    "tennis_court": ["网球场", "tennis court", "tennis courts"],
    "airport": ["机场", "航站区", "airport", "airfield"],
    "harbor": ["港口", "港区", "码头", "港湾", "harbor", "harbour", "port", "dock", "pier"],
    "chimney": ["烟囱", "烟筒", "chimney", "chimneys", "smokestack", "smokestacks"],
    "dam": ["水坝", "大坝", "堤坝", "dam", "dams"],
    "expressway_service_area": ["高速公路服务区", "高速服务区", "服务区", "expressway service area", "highway service area"],
    "expressway_toll_station": ["高速公路收费站", "高速收费站", "收费站", "收费口", "expressway toll station", "highway toll station", "toll station", "toll plaza"],
    "train_station": ["火车站", "铁路车站", "列车站", "train station", "railway station"],
    "windmill": ["风车", "风力发电机", "风机", "风电机组", "windmill", "windmills", "wind turbine", "wind turbines"],
}

TARGET_LABELS = {
    "background": "背景",
    "building": "建筑",
    "road": "道路",
    "water": "水体",
    "barren": "裸地",
    "forest": "林地",
    "agricultural": "农田",
    "vegetation": "植被",
    "parking_lot": "停车场",
    "runway": "跑道",
    "ship": "船舶",
    "vehicle": "车辆",
    "small_vehicle": "小型车辆",
    "large_vehicle": "大型车辆",
    "bridge": "桥梁",
    "overpass": "高架桥",
    "aircraft": "飞机",
    "storage_tank": "储罐",
    "sports_field": "运动场",
    "baseball_field": "棒球场",
    "basketball_court": "篮球场",
    "ground_track_field": "田径场",
    "golf_course": "高尔夫球场",
    "stadium": "体育场",
    "tennis_court": "网球场",
    "airport": "机场",
    "harbor": "港口",
    "chimney": "烟囱",
    "dam": "大坝",
    "expressway_service_area": "高速公路服务区",
    "expressway_toll_station": "高速公路收费站",
    "train_station": "火车站",
    "windmill": "风机",
}

def _intent(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


SEGMENTATION_INTENT = _intent(
    r"分割|分离出|语义切分|语义分区|语义图|标签图|类别图|掩码|掩膜|蒙版|"
    r"抠出|抠取|提取.{0,8}(?:区域|轮廓)|像素级|逐像素|区域标注|"
    r"\bsegmentation\b|\bsemantic masks?\b|\bbinary masks?\b|"
    r"\bsemantic maps?\b|\bclass maps?\b|\blabel maps?\b|"
    r"^(?:please\s+)?segment\b|\bisolate\b|\bdelineate\b|\bextract .{0,30} regions?\b"
)
ALL_CLASS_INTENT = _intent(
    r"全类别|所有类别|各个类别|每种类别|所有地物类别|所有语义类别|全部地物类别|"
    r"全部地表覆盖类别|每个土地覆盖类别|所有土地覆盖类别|不用指定类别|不指定类别|"
    r"全景语义分割|全语义分割|多类别分割|完整语义|完整标签图|完整类别图|"
    r"整(?:张|幅|个)?图(?:像|影像)?.{0,18}分割|全图.{0,12}分割|"
    r"all[- ]class|all classes|all semantic classes|all land[- ]cover classes|"
    r"every land[- ]cover class|full semantic segmentation|full[- ]scene multiclass|"
    r"complete semantic map|segment (?:the )?(?:whole|entire|full) image"
)
GENERIC_FULL_SCENE_INTENT = _intent(
    r"(?:对|给)?(?:这|该|当前)?(?:张|幅|个)?(?:遥感)?(?:图|图像|影像)"
    r"(?:进行|做|生成|完成)?(?:一次|一下)?(?:完整)?(?:语义|地物)?分割|"
    r"(?:生成|输出|给出)?(?:这|该|当前)?(?:张|幅)?(?:遥感)?(?:图|图像|影像)的?"
    r"(?:语义分割|地物分割)(?:结果|图|掩码|掩膜)?|"
    r"(?:run|perform|generate) (?:a )?(?:semantic )?segmentation "
    r"(?:of|for|on) (?:this|the) (?:image|scene)"
)
COUNTING_INTENT = _intent(
    r"多少|有几(?:个|辆|艘|架|栋|座|处|片|条|台|列|块|根|口)|"
    r"一共(?:有)?几|总共有几|数一数|数一下|帮我数|请数|数数|计数|清点|统计.{0,12}(?:数量|数目|个数)|"
    r"数量(?:是多少|有多少|为多少)|数目(?:是多少|有多少)|总数|个数|"
    r"\bhow many\b|\bcount\b|\bcounting\b|\bnumber of\b|\btotal number\b|"
    r"\bquantity of\b|\btally\b|\bquantify\b"
)
GROUNDING_INTENT = _intent(
    r"定位|位置|位于哪里|在哪里|在哪儿|何处|坐标|方位|框出|框选|画框|圈出|"
    r"标出|标记|标注出|目标框|指出|找出|寻找|识别并标注|最靠近|最近的|距离.{0,16}最近|"
    r"\blocate\b|\blocation\b|\bwhere (?:is|are|can)\b|\bbounding box\b|"
    r"\bbox (?:the|all)\b|\bmark\b|\boutline\b|\bpoint out\b|\bfind (?:the|all)\b|"
    r"\bidentify (?:the )?location\b|\bnearest\b|\bclosest\b"
)
CHANGE_INTENT = _intent(
    r"时相|前后变化|发生.{0,8}变化|变化检测|变化区域|有什么变化|有何变化|"
    r"有什么不同|哪里不同|"
    r"对比.{0,16}(?:变化|差异|不同)|比较.{0,16}(?:变化|差异|不同)|"
    r"两(?:张|幅|期).{0,16}(?:对比|比较)|前一张.{0,18}后一张|"
    r"差异|不同之处|新增|新建|建成|拆除|消失|扩建|扩张|缩减|减少|增加|"
    r"灾前灾后|震前震后|洪水前后|\bchange detection\b|\bchanges? between\b|"
    r"\bwhat (?:has )?changed\b|\bdifferences? between\b|\bcompare .{0,30} images?\b|"
    r"\bbefore and after\b|\bnewly built\b|\bdisappeared\b|\bdemolished\b"
)
SCENE_INTENT = _intent(
    r"场景类型|场景类别|什么场景|什么类型的.{0,8}场景|哪种场景|属于.{0,8}场景|区域类型|区域类别|"
    r"什么区域|哪类区域|地貌类型|地表类型|土地利用类型|土地覆盖类型|"
    r"主要用途|功能区类型|城乡类型|城市还是农村|"
    r"\bscene (?:type|class|category)\b|\btype of (?:scene|area|landscape)\b|"
    r"\bkind of (?:scene|area|landscape)\b|\bwhat (?:scene|area) is\b|"
    r"\burban or rural\b|\bland[- ]use type\b|\bland[- ]cover type\b"
)
CAPTION_INTENT = _intent(
    r"详细描述|描述这|描述一下|描述图|描述影像|描述.{0,8}(?:图|影像)|"
    r"介绍这|介绍图|概述这|概述图|"
    r"整体描述|总体描述|说说.{0,10}(?:图|影像)|讲讲.{0,10}(?:图|影像)|"
    r"图中有什么|图里有什么|包含哪些地物|看到了什么|解读这|"
    r"影像内容|图片内容|生成.{0,6}(?:描述|说明)|"
    r"\bdescribe (?:this|the) .{0,18}(?:image|photo|scene)\b|"
    r"\bdescribe .{0,18}(?:remote[- ]sensing|satellite|aerial) image\b|"
    r"\bimage caption\b|\bcaption (?:this|the) image\b|\bsummarize (?:this|the) image\b|"
    r"\bwhat (?:can you see|is shown|does the image contain)\b|\bgive an overview\b"
)

NEGATED_INTENTS = {
    "semantic_segmentation": _intent(r"(?:不要|不用|无需|(?:^|[，,；;。！？\s])别)\s*(?:进行|做|生成)?\s*(?:分割|分离|掩[码膜])|(?:do not|don't|without)\s+(?:segment|segmentation|mask)"),
    "object_counting": _intent(r"(?:不要|不用|无需|(?:^|[，,；;。！？\s])别)\s*(?:进行|做)?\s*(?:计数|统计|数)|(?:do not|don't)\s+(?:count|tally)"),
    "grounding": _intent(r"(?:不要|不用|无需|(?:^|[，,；;。！？\s])别)\s*(?:进行|做)?\s*(?:定位|找出|标出|框出)|(?:do not|don't)\s+(?:locate|mark|find|draw a box)"),
    "change_detection": _intent(r"(?:不要|不用|无需|(?:^|[，,；;。！？\s])别)\s*(?:比较|对比|分析)?\s*(?:变化|差异)|(?:do not|don't)\s+(?:compare|analy[sz]e changes?)"),
    "scene_classification": _intent(r"(?:不要|不用|无需|(?:^|[，,；;。！？\s])别)\s*(?:判断|识别)?\s*(?:场景|区域类型)|(?:do not|don't)\s+classify (?:the )?scene"),
    "image_captioning": _intent(r"(?:不要|不用|无需|(?:^|[，,；;。！？\s])别)\s*(?:详细)?描述|(?:do not|don't)\s+(?:describe|caption)"),
}

TASK_PATTERNS = {
    "semantic_segmentation": SEGMENTATION_INTENT,
    "object_counting": COUNTING_INTENT,
    "grounding": GROUNDING_INTENT,
    "change_detection": CHANGE_INTENT,
    "scene_classification": SCENE_INTENT,
    "image_captioning": CAPTION_INTENT,
}
TASK_PRIORITY = (
    "semantic_segmentation",
    "grounding",
    "object_counting",
    "change_detection",
    "scene_classification",
    "image_captioning",
)
DENSE_COUNTING_HINT = _intent(
    r"密集|密密麻麻|大量|成片|成群|建筑群|全部建筑|所有建筑|逐个数|精确统计|"
    r"dense|crowded|large number of|all buildings|every building|count each"
)
COMPOUND_SPLIT = _intent(
    r"\s*(?:，|,|；|;|。|\?|？|！|!|并且|同时|然后|接着|随后|再帮我|再给我|"
    r"并分别|以及还要|并(?=(?:帮我|给我|数|统计|定位|找|描述|判断|比较|分割))|"
    r"and then|as well as|also|after that)\s*"
)

FOCUSED_PROMPTS = {
    "semantic_segmentation": "请只完成原始请求中的语义分割任务，并保留原始请求指定的类别。原始请求：{query}",
    "grounding": "请只完成原始请求中的目标定位任务，返回目标坐标。原始请求：{query}",
    "object_counting": "请只完成原始请求中的目标计数任务，直接回答数量。原始请求：{query}",
    "change_detection": "请只完成原始请求中的时相变化分析任务。原始请求：{query}",
    "scene_classification": "请只完成原始请求中的场景分类任务。原始请求：{query}",
    "image_captioning": "请只完成原始请求中的遥感影像描述任务。原始请求：{query}",
}


@dataclass(frozen=True)
class PlannedTask:
    task: str
    prompt: str
    matched_text: tuple[str, ...]
    targets: tuple[str, ...] = ()
    subtype: str | None = None


@dataclass(frozen=True)
class RouteDecision:
    query: str
    tasks: tuple[str, ...]
    plans: tuple[PlannedTask, ...]
    warnings: tuple[str, ...] = ()

    @property
    def primary_task(self) -> str | None:
        return self.tasks[0] if self.tasks else None

    @property
    def is_compound(self) -> bool:
        return len(self.tasks) > 1

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "tasks": list(self.tasks),
            "primary_task": self.primary_task,
            "is_compound": self.is_compound,
            "warnings": list(self.warnings),
            "plans": [
                {
                    "task": plan.task,
                    "prompt": plan.prompt,
                    "matched_text": list(plan.matched_text),
                    "targets": list(plan.targets),
                    "subtype": plan.subtype,
                }
                for plan in self.plans
            ],
        }


@dataclass(frozen=True)
class SegmentationRoute:
    mode: str
    targets: tuple[str, ...]
    unsupported: tuple[str, ...]


def _contains(text: str, alias: str) -> bool:
    alias = alias.lower()
    if re.fullmatch(r"[a-z0-9 _-]+", alias):
        return re.search(
            rf"(?<![a-z0-9_]){re.escape(alias)}(?![a-z0-9_])",
            text,
        ) is not None
    if alias == "油罐":
        return re.search(r"油罐(?!车)", text) is not None
    return alias in text


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _matched_text(pattern: re.Pattern[str], text: str) -> tuple[str, ...]:
    return _unique(match.group(0).strip() for match in pattern.finditer(text))


def _all_targets(query: str) -> tuple[str, ...]:
    text = query.strip().lower()
    aliases = {**SUPPORTED_TARGETS, **UNSUPPORTED_TARGETS}
    return _prune_targets(
        name
        for name, values in aliases.items()
        if any(_contains(text, alias) for alias in values)
    )


def _prune_targets(values: Iterable[str]) -> tuple[str, ...]:
    targets = list(dict.fromkeys(values))
    groups = (
        ("vehicle", {"small_vehicle", "large_vehicle"}),
        (
            "sports_field",
            {
                "baseball_field",
                "basketball_court",
                "ground_track_field",
                "golf_course",
                "stadium",
                "tennis_court",
            },
        ),
        ("bridge", {"overpass"}),
        ("airport", {"runway"}),
        (
            "road",
            {"expressway_service_area", "expressway_toll_station"},
        ),
    )
    for generic, specifics in groups:
        if generic in targets and any(target in targets for target in specifics):
            targets.remove(generic)
    return tuple(targets)


def _detect_clause(clause: str) -> dict[str, tuple[str, ...]]:
    candidates: list[tuple[int, int, str, tuple[str, ...]]] = []
    for task, pattern in TASK_PATTERNS.items():
        matches = _matched_text(pattern, clause)
        if matches and not NEGATED_INTENTS[task].search(clause):
            first = pattern.search(clause)
            candidates.append(
                (first.start() if first else len(clause), TASK_PRIORITY.index(task), task, matches)
            )
    detected = {
        task: matches
        for _, _, task, matches in sorted(candidates)
    }

    # “描述图像的场景类型”和“描述两期变化”里的“描述”不是额外的描述任务。
    if "image_captioning" in detected and (
        "scene_classification" in detected or "change_detection" in detected
    ):
        detected.pop("image_captioning")
    if "grounding" in detected and "change_detection" in detected:
        strong_grounding = re.search(
            r"定位|位置|在哪里|在哪儿|何处|坐标|框出|框选|画框|圈出|标出|标记|"
            r"\blocate\b|\blocation\b|\bwhere\b|\bbounding box\b|\bpoint out\b",
            clause,
            re.IGNORECASE,
        )
        if not strong_grounding:
            detected.pop("grounding")
    return detected


def _plan_subtype(task: str, query: str) -> str | None:
    if task == "semantic_segmentation":
        return (
            "all_classes"
            if ALL_CLASS_INTENT.search(query) or GENERIC_FULL_SCENE_INTENT.search(query)
            else "targeted"
        )
    if task == "object_counting":
        return "dense_candidate" if DENSE_COUNTING_HINT.search(query) else "standard"
    if task == "grounding":
        return (
            "multiple_targets"
            if re.search(r"所有|全部|每个|all|every", query, re.IGNORECASE)
            else "single_target"
        )
    return None


def route_tasks(query: str, image_count: int | None = None) -> RouteDecision:
    """Plan known remote-sensing tasks without invoking a language model.

    Rules intentionally return no task for general VQA. Multiple matched tasks are
    preserved in priority order so a future executor can run them sequentially.
    """
    text = query.strip()
    if not text:
        return RouteDecision(query=query, tasks=(), plans=(), warnings=("empty_query",))

    clauses = tuple(part.strip() for part in COMPOUND_SPLIT.split(text) if part.strip())
    if not clauses:
        clauses = (text,)
    by_task: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    for clause in clauses:
        for task, matches in _detect_clause(clause).items():
            by_task.setdefault(task, []).append((clause, matches))

    tasks = tuple(by_task)
    targets = _all_targets(text)
    plans: list[PlannedTask] = []
    is_compound = len(tasks) > 1
    for task in tasks:
        task_clauses = by_task[task]
        clause, matches = task_clauses[0]
        clause_tasks = _detect_clause(clause)
        clause_has_required_target = (
            task not in {"semantic_segmentation", "grounding", "object_counting"}
            or bool(_all_targets(clause))
        )
        if not is_compound:
            prompt = text
        elif len(clause_tasks) == 1 and clause != text and clause_has_required_target:
            prompt = clause
        else:
            prompt = FOCUSED_PROMPTS[task].format(query=text)
        plans.append(
            PlannedTask(
                task=task,
                prompt=prompt,
                matched_text=_unique(
                    value
                    for _, clause_matches in task_clauses
                    for value in clause_matches
                ),
                targets=targets,
                subtype=_plan_subtype(task, text),
            )
        )

    warnings: list[str] = []
    if not tasks:
        warnings.append("no_rule_match")
    if is_compound:
        warnings.append("compound_execution_required")
    if "change_detection" in tasks and image_count is not None and image_count != 2:
        warnings.append("change_detection_expected_two_images")
    if "semantic_segmentation" in tasks and image_count is not None and image_count != 1:
        warnings.append("segmentation_expert_requires_one_image")
    if tasks == ("semantic_segmentation",):
        segmentation = (
            SegmentationRoute("all_classes", (), ())
            if ALL_CLASS_INTENT.search(text) or GENERIC_FULL_SCENE_INTENT.search(text)
            else resolve_segmentation_targets(text)
        )
        if segmentation.mode == "targeted" and not segmentation.targets and not segmentation.unsupported:
            warnings.append("segmentation_target_unresolved")
        if segmentation.unsupported:
            warnings.extend(
                f"segmentation_unsupported_target:{target}"
                for target in segmentation.unsupported
            )
    return RouteDecision(
        query=text,
        tasks=tasks,
        plans=tuple(plans),
        warnings=tuple(warnings),
    )


def route_segmentation(query: str) -> SegmentationRoute | None:
    decision = route_tasks(query)
    if decision.tasks != ("semantic_segmentation",):
        return None
    if ALL_CLASS_INTENT.search(query) or GENERIC_FULL_SCENE_INTENT.search(query):
        return SegmentationRoute("all_classes", (), ())
    return resolve_segmentation_targets(query)


def resolve_segmentation_targets(query: str) -> SegmentationRoute:
    """Resolve class names after segmentation intent is already known."""
    text = query.strip().lower()
    targets = _prune_targets(
        name
        for name, aliases in SUPPORTED_TARGETS.items()
        if any(_contains(text, alias) for alias in aliases)
    )
    unsupported = _prune_targets(
        name
        for name, aliases in UNSUPPORTED_TARGETS.items()
        if any(_contains(text, alias) for alias in aliases)
    )
    return SegmentationRoute("targeted", targets, unsupported)


def route_segmentation_token(
    answer: str,
    original_query: str,
) -> SegmentationRoute | None:
    """Accept only the two exact route tokens learned during SFT."""
    token = answer.strip().rstrip("。.! ").lower()
    if token == "<seg_all>":
        return SegmentationRoute("all_classes", (), ())
    if token == "<seg>":
        return resolve_segmentation_targets(original_query)
    return None


def call_segmentation_expert(
    endpoint: str,
    *,
    image_path: Path,
    route: SegmentationRoute,
    request_id: str,
    timeout_seconds: int,
) -> dict:
    payload = json.dumps(
        {
            "image_path": str(image_path),
            "mode": route.mode,
            "targets": list(route.targets),
            "request_id": request_id,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"分割专家返回 HTTP {exc.code}：{detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"分割专家不可用：{exc.reason}") from exc
