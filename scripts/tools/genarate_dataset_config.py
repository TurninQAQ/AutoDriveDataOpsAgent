#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import io
import os
import yaml
from collections import OrderedDict

try:
    text_type = unicode
except NameError:
    text_type = str


# =============================================================================
# 手工生成 YAML 时优先修改这里
# =============================================================================

# 输入 record 路径列表。每个 record 目录下面应该直接包含 clip_* 子目录。
# 可以填一个，也可以填多个；多个 record 会合并进同一个任务 YAML。
# 如果命令行传了 record 路径，会覆盖这里。
TARGET_DIRS = [
    "/home/cfy/project/two/test/record_multitask_full_a",
]

# 生成出来的任务 YAML 保存路径。
# 如果命令行传了 -o/--output，会覆盖这里。
OUTPUT_YAML = "datasets_config.yaml"

# 可选。任务类型会去 config/task_types.yaml 读取默认优先级。
# 如果同时写了 PRIORITY，PRIORITY 会覆盖 task_type 默认优先级。
# None 表示不写入 YAML，提交时使用 config/task_types.yaml 的 default_priority。
TASK_TYPE = None
PRIORITY = None

# 每个 clip 的默认资源标签和 Airflow pool。
TIER = "small"
POOL = "default_pool"

# 每个阶段拿到 GPU/开始运行后的超时时间，单位分钟；GPU 等待时间不计入。
TIMEOUT_MIN = 60

# 【常改】同一个任务 DAG 内最多同时跑多少个 clip。
MAX_ACTIVE_RUNS = 5

# 【一般不改】任务级独占锁。True 表示一批任务跑完后，下一批任务才开始。
TASK_EXCLUSIVE = True
TASK_LOCK_WAIT_INTERVAL_SEC = 10
PREEMPT_GRACE_TIMEOUT_MIN = 60

# 【常改】流程顺序，写成什么顺序，平台就按什么顺序跑。
# 字符串表示串行阶段；列表表示并行阶段。
# 例1：全串行
#   ["precheck", "parser", "segment", "map", "od", "coloration", "occ"]
# 例2：OD 和 OCC 并行
#   ["precheck", "parser", "segment", "map", ["od", "occ"], "coloration"]
PIPELINE_STAGES = [
    "precheck",
    "parser",
    "segment",
    "map",
    "od",
    "coloration",
    "occ",
]

# GPU 调度配置。precheck/parser/map/coloration 默认不用 GPU，不要写进 GPU_STAGES。
GPU_IDS = "5,6,7,8,9"
GPU_STAGES = "segment,od,occ"

# 【常改】需要独占一张接近空闲 GPU 的阶段。
# 当前 OD 按 segment 级别处理，默认 segment 和 od 都独占；设为 "" 表示关闭独占。
EXCLUSIVE_GPU_STAGES = "segment,od"
EXCLUSIVE_GPU_IDLE_USED_MAX_MB = 512
GPU_STAGE_MEMORY_MB = OrderedDict([
    ("segment", 24000),
    ("od", 24000),
    ("occ", 4000),
])
GPU_WAIT_INTERVAL_SEC = 10
GPU_RESERVATION_PENDING_SEC = 60

# 各算法镜像。precheck 是本地检查脚本，不需要镜像。
DEFAULT_IMAGES = OrderedDict([
    ("image_parser", "172.16.201.100:5000/data_parser:v1.1.1_cidi_07-15_16_12_27"),
    ("image_segment", "172.16.201.100:5000/sam31:v1.1.2_cfy_07-15_14_25_15"),
    ("image_map", "172.16.201.100:5000/offline_mapping:v1.1.3_cidi_07-15_17_52_07"),
    ("image_od", "172.16.201.100:5000/label_od:v1.1.14_nty_07-07_17_54_58"),
    ("image_coloration", "172.16.201.100:5000/pointcloud_coloration:v1.1.0_cidi_07-15_17_44_25"),
    ("image_occ", "172.16.201.100:5000/label_occ:v1.0.27_cidi_07-21_10_15_45"),
])

# 【不要改】给命令行默认提示用。
PIPELINE_STAGES_ARG = ",".join(
    "+".join(group) if isinstance(group, list) else group
    for group in PIPELINE_STAGES
)


class QuotedString(str):
    pass


def represent_ordered_dict(dumper, data):
    return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())


def represent_quoted_string(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')


yaml.add_representer(OrderedDict, represent_ordered_dict)
yaml.add_representer(QuotedString, represent_quoted_string)


def parse_csv(value):
    if value is None:
        return []
    return [item.strip() for item in str(value).replace("，", ",").split(",") if item.strip()]


def csv_text(value):
    if isinstance(value, (list, tuple)):
        return ",".join(parse_csv(",".join(str(item) for item in value)))
    return ",".join(parse_csv(value))


def parse_pipeline_stages(value):
    groups = []
    for item in parse_csv(value):
        stages = [stage.strip() for stage in item.split("+") if stage.strip()]
        if not stages:
            continue
        groups.append(stages if len(stages) > 1 else stages[0])
    return groups


def flatten_pipeline_stages(stage_groups):
    result = []
    for item in stage_groups:
        result.extend(item if isinstance(item, list) else [item])
    return result


def parse_stage_memory(value):
    if value is None:
        return None
    stage_memory = OrderedDict()
    for item in parse_csv(value):
        if ":" not in item:
            raise ValueError("stage memory item must be stage:mb, got {}".format(item))
        stage, memory = item.split(":", 1)
        stage = stage.strip()
        memory = int(memory.strip())
        if not stage or memory <= 0:
            raise ValueError("invalid stage memory item: {}".format(item))
        stage_memory[stage] = memory
    return stage_memory


def load_base_yaml(path):
    if not path:
        return None
    base_path = os.path.abspath(os.path.expanduser(path))
    with io.open(base_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("base yaml root must be a mapping: {}".format(base_path))
    return data


def first_dataset_config(base_config):
    if not base_config:
        return {}
    datasets = base_config.get("datasets") or []
    for item in datasets:
        if isinstance(item, dict):
            return item
    return {}


def ordered_mapping(value, fallback):
    source = value if isinstance(value, dict) else fallback
    return OrderedDict((key, source[key]) for key in source)


def base_images(base_config):
    images = OrderedDict(DEFAULT_IMAGES)
    base_dataset = first_dataset_config(base_config)
    for key, value in base_dataset.items():
        if str(key).startswith("image_") and value:
            images[key] = value
    return images


def build_images(args, base_config=None):
    images = base_images(base_config) if base_config else OrderedDict(DEFAULT_IMAGES)
    for key in DEFAULT_IMAGES:
        value = getattr(args, key, None)
        if value:
            images[key] = value
    return images


def generate_dataset_configs(
    target_dirs,
    output_file="datasets.yaml",
    pipeline_stages=None,
    task_type=None,
    priority=None,
    max_active_runs=MAX_ACTIVE_RUNS,
    task_exclusive=TASK_EXCLUSIVE,
    task_lock_wait_interval_sec=TASK_LOCK_WAIT_INTERVAL_SEC,
    preempt_grace_timeout_min=PREEMPT_GRACE_TIMEOUT_MIN,
    gpu_ids=GPU_IDS,
    gpu_stages=GPU_STAGES,
    exclusive_gpu_stages=None,
    exclusive_gpu_idle_used_max_mb=None,
    gpu_stage_memory_mb=None,
    gpu_wait_interval_sec=GPU_WAIT_INTERVAL_SEC,
    gpu_reservation_pending_sec=GPU_RESERVATION_PENDING_SEC,
    images=None,
    tier=None,
    pool=None,
    timeout_min=TIMEOUT_MIN,
    base_config=None,
):
    """
    扫描多个 root_dir 下所有以 'clip_' 开头的子目录，
    生成包含 'datasets:' 根键的 YAML 配置文件。
    """
    if not isinstance(target_dirs, list):
        target_dirs = [target_dirs]

    all_clip_dirs = []

    for root_dir in target_dirs:
        root_dir = os.path.abspath(os.path.expanduser(root_dir))
        one_record_dirs = []
        print(root_dir)
        if not os.path.exists(root_dir):
            print("警告: 目录 {} 不存在，跳过".format(root_dir))
            continue

        try:
            for item in os.listdir(root_dir):
                full_path = os.path.join(root_dir, item)
                if os.path.isdir(full_path) and item.startswith("clip_"):
                    one_record_dirs.append((root_dir, item))
        except OSError:
            print("警告: 没有权限访问目录 {}，跳过".format(root_dir))
            continue

        one_record_dirs.sort(key=lambda x: x[1])
        all_clip_dirs.extend(one_record_dirs)

    if not all_clip_dirs:
        raise RuntimeError("在所有目标目录中未找到以 'clip_' 开头的子目录")

    print("找到 {} 个 clip 目录，正在生成配置...".format(len(all_clip_dirs)))

    base_config = base_config or {}
    base_dataset = first_dataset_config(base_config)

    pipeline_stages = (
        pipeline_stages
        if pipeline_stages is not None
        else base_config.get("pipeline_stages") or list(PIPELINE_STAGES)
    )
    selected_stages = set(flatten_pipeline_stages(pipeline_stages))
    task_type = (
        task_type
        if task_type is not None
        else base_config.get("task_type", TASK_TYPE)
    )
    priority = (
        priority
        if priority is not None
        else base_config.get("priority", PRIORITY)
    )
    max_active_runs = (
        max_active_runs
        if max_active_runs is not None
        else base_config.get("max_active_runs", MAX_ACTIVE_RUNS)
    )
    task_exclusive = (
        task_exclusive
        if task_exclusive is not None
        else base_config.get("task_exclusive", TASK_EXCLUSIVE)
    )
    task_lock_wait_interval_sec = (
        task_lock_wait_interval_sec
        if task_lock_wait_interval_sec is not None
        else base_config.get("task_lock_wait_interval_sec", TASK_LOCK_WAIT_INTERVAL_SEC)
    )
    preempt_grace_timeout_min = (
        preempt_grace_timeout_min
        if preempt_grace_timeout_min is not None
        else base_config.get("preempt_grace_timeout_min", PREEMPT_GRACE_TIMEOUT_MIN)
    )
    gpu_ids = gpu_ids if gpu_ids is not None else base_config.get("gpu_ids", GPU_IDS)
    gpu_stages = gpu_stages if gpu_stages is not None else base_config.get("gpu_stages", GPU_STAGES)
    exclusive_gpu_stages = (
        exclusive_gpu_stages
        if exclusive_gpu_stages is not None
        else base_config.get("exclusive_gpu_stages", EXCLUSIVE_GPU_STAGES)
    )
    exclusive_gpu_idle_used_max_mb = (
        exclusive_gpu_idle_used_max_mb
        if exclusive_gpu_idle_used_max_mb is not None
        else base_config.get(
            "exclusive_gpu_idle_used_max_mb",
            EXCLUSIVE_GPU_IDLE_USED_MAX_MB,
        )
    )
    gpu_stage_memory_mb = (
        gpu_stage_memory_mb
        if gpu_stage_memory_mb is not None
        else ordered_mapping(base_config.get("gpu_stage_memory_mb"), GPU_STAGE_MEMORY_MB)
    )
    gpu_wait_interval_sec = (
        gpu_wait_interval_sec
        if gpu_wait_interval_sec is not None
        else base_config.get("gpu_wait_interval_sec", GPU_WAIT_INTERVAL_SEC)
    )
    gpu_reservation_pending_sec = (
        gpu_reservation_pending_sec
        if gpu_reservation_pending_sec is not None
        else base_config.get("gpu_reservation_pending_sec", GPU_RESERVATION_PENDING_SEC)
    )
    images = images or base_images(base_config)
    timeout_min = (
        timeout_min
        if timeout_min is not None
        else base_dataset.get("timeout_min", TIMEOUT_MIN)
    )
    tier = tier if tier is not None else base_dataset.get("tier", TIER)
    pool = pool if pool is not None else base_dataset.get("pool", POOL)
    configs_list = []

    for root_dir, clip_name in all_clip_dirs:
        single_config = OrderedDict([
            ("dataset_name", clip_name),
            ("tier", tier),
            ("pool", pool),
            ("dataset_path", root_dir),
        ])
        for key, value in images.items():
            stage_name = key.replace("image_", "")
            if stage_name in selected_stages:
                single_config[key] = value
        single_config["timeout_min"] = timeout_min
        configs_list.append(single_config)

    final_output = OrderedDict()
    if task_type not in (None, ""):
        final_output["task_type"] = task_type
    if priority not in (None, ""):
        final_output["priority"] = int(priority)
    final_output["pipeline_stages"] = pipeline_stages
    final_output["max_active_runs"] = max_active_runs
    final_output["task_exclusive"] = task_exclusive
    final_output["task_lock_wait_interval_sec"] = task_lock_wait_interval_sec
    final_output["preempt_grace_timeout_min"] = preempt_grace_timeout_min
    gpu_ids_text = csv_text(gpu_ids)
    gpu_stages_text = csv_text(gpu_stages)
    exclusive_gpu_stages_text = csv_text(exclusive_gpu_stages)
    if gpu_stages_text:
        final_output["gpu_ids"] = QuotedString(gpu_ids_text)
    final_output["gpu_stages"] = QuotedString(gpu_stages_text)
    if gpu_stages_text:
        final_output["exclusive_gpu_stages"] = QuotedString(exclusive_gpu_stages_text)
        final_output["exclusive_gpu_idle_used_max_mb"] = exclusive_gpu_idle_used_max_mb
        final_output["gpu_stage_memory_mb"] = gpu_stage_memory_mb
        final_output["gpu_wait_interval_sec"] = gpu_wait_interval_sec
        final_output["gpu_reservation_pending_sec"] = gpu_reservation_pending_sec
    final_output["datasets"] = configs_list

    yaml_text = yaml.dump(final_output, allow_unicode=True, default_flow_style=False)
    if not isinstance(yaml_text, text_type):
        yaml_text = yaml_text.decode("utf-8")
    with io.open(output_file, 'w', encoding='utf-8') as f:
        f.write(yaml_text)
    print("成功生成配置文件: {}".format(output_file))
    print("共包含 {} 个数据集配置".format(len(configs_list)))
    return output_file


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Generate task submit YAML from record dirs")
    parser.add_argument(
        "target_dirs",
        nargs="*",
        help="record dirs containing clip_* subdirs; empty means use TARGET_DIRS in this file",
    )
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument(
        "--base-yaml",
        default=None,
        help="inherit stage/image/gpu/default dataset settings from an existing stable YAML",
    )
    parser.add_argument(
        "--pipeline-stages",
        default=None,
        help="comma-separated groups; use + for parallel stages, e.g. od+occ; default: {}".format(PIPELINE_STAGES_ARG),
    )
    parser.add_argument(
        "--task-type",
        default=None,
        help="task type defined in config/task_types.yaml",
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=None,
        help="explicit task priority; smaller number means higher priority",
    )
    parser.add_argument("--max-active-runs", type=int, default=None)
    task_exclusive_group = parser.add_mutually_exclusive_group()
    task_exclusive_group.add_argument("--task-exclusive", dest="task_exclusive", action="store_true")
    task_exclusive_group.add_argument("--no-task-exclusive", dest="task_exclusive", action="store_false")
    parser.set_defaults(task_exclusive=None)
    parser.add_argument(
        "--task-lock-wait-interval-sec",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--preempt-grace-timeout-min",
        type=int,
        default=None,
    )
    parser.add_argument("--gpu-ids", default=None)
    parser.add_argument("--gpu-stages", default=None)
    parser.add_argument(
        "--exclusive-gpu-stages",
        default=None,
        help='需要独占 GPU 的阶段；不传则使用 base-yaml 或文件顶部 EXCLUSIVE_GPU_STAGES，设为 "" 表示关闭',
    )
    parser.add_argument(
        "--exclusive-gpu-idle-used-max-mb",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--gpu-stage-memory-mb",
        default=None,
        help="comma separated stage:memory_mb list",
    )
    parser.add_argument("--gpu-wait-interval-sec", type=int, default=None)
    parser.add_argument(
        "--gpu-reservation-pending-sec",
        type=int,
        default=None,
    )
    parser.add_argument("--timeout-min", type=int, default=None)
    parser.add_argument("--tier", default=None)
    parser.add_argument("--pool", default=None)
    for image_key, default_value in DEFAULT_IMAGES.items():
        parser.add_argument("--{}".format(image_key.replace("_", "-")), default=None)
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        base_config = load_base_yaml(args.base_yaml)
        target_dirs = args.target_dirs or list(TARGET_DIRS)
        if not target_dirs:
            raise ValueError("未提供 record 路径。请修改 TARGET_DIRS，或在命令行传入 record 目录。")

        output_file = args.output or OUTPUT_YAML
        generated_yaml = generate_dataset_configs(
            target_dirs,
            output_file=output_file,
            pipeline_stages=parse_pipeline_stages(args.pipeline_stages) if args.pipeline_stages else None,
            task_type=args.task_type,
            priority=args.priority,
            max_active_runs=args.max_active_runs,
            task_exclusive=args.task_exclusive,
            task_lock_wait_interval_sec=args.task_lock_wait_interval_sec,
            preempt_grace_timeout_min=args.preempt_grace_timeout_min,
            gpu_ids=args.gpu_ids,
            gpu_stages=args.gpu_stages,
            exclusive_gpu_stages=args.exclusive_gpu_stages,
            exclusive_gpu_idle_used_max_mb=args.exclusive_gpu_idle_used_max_mb,
            gpu_stage_memory_mb=parse_stage_memory(args.gpu_stage_memory_mb),
            gpu_wait_interval_sec=args.gpu_wait_interval_sec,
            gpu_reservation_pending_sec=args.gpu_reservation_pending_sec,
            images=build_images(args, base_config=base_config),
            tier=args.tier,
            pool=args.pool,
            timeout_min=args.timeout_min,
            base_config=base_config,
        )
        print("")
        print("下一步提交命令示例，请在提交时填写任务名前缀:")
        print(
            "/opt/airflow/scripts/manage_task.sh submit --name <任务名前缀> --yaml {}".format(
                os.path.abspath(generated_yaml),
            )
        )
    except Exception as exc:
        print("生成配置失败: {}".format(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
