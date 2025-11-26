import os
from pathlib import Path
import tempfile

import hydra
from nuplan.planning.script.run_training import main as main_train
from nuplan.planning.script.run_simulation import main as main_simulation
from nuplan.planning.script.run_nuboard import main as main_nuboard

# -----------------------------
# 通用路径配置
# -----------------------------
# training config
TRAIN_CONFIG_PATH = '../nuplan/planning/script/config/training'
TRAIN_CONFIG_NAME = 'default_training'

# simulation config
SIM_CONFIG_PATH = '../nuplan/planning/script/config/simulation'
SIM_CONFIG_NAME = 'default_simulation'

# nuboard config（现在先不用，可以先注释掉）
NUBOARD_CONFIG_PATH = '../nuplan/planning/script/config/nuboard'
NUBOARD_CONFIG_NAME = 'default_nuboard'

# 保存目录（临时目录，可以换成你自己的持久路径）
SAVE_DIR = Path(tempfile.gettempdir()) / 'tutorial_nuplan_framework'

EXPERIMENT_TRAIN = 'training_raster_experiment'
JOB_NAME = 'train_default_raster'
JOB_NAME_SIM = 'sim_ml_debug'  
LOG_DIR = SAVE_DIR / EXPERIMENT_TRAIN / JOB_NAME


def find_ckpt(root: Path) -> Path:
    """
    在 root 下面递归搜索 .ckpt
    优先选包含 'last' 的，其次选按名字排序最后一个
    """
    ckpts = list(root.rglob('*.ckpt'))
    if not ckpts:
        return None

    last_ckpts = [p for p in ckpts if 'last' in p.name]
    if last_ckpts:
        return sorted(last_ckpts)[-1]
    return sorted(ckpts)[-1]


# --------- 1. 训练（只在没有 ckpt 时跑一次） ---------
# 先在 SAVE_DIR 下面找有没有现成的 ckpt
existing_ckpt = find_ckpt(SAVE_DIR)

if existing_ckpt is not None:
    CHECKPOINT_PATH = str(existing_ckpt)
    print(f'发现已有 checkpoint，跳过训练：{CHECKPOINT_PATH}')
else:
    print('未发现 checkpoint，开始训练一次 raster 模型...')
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=TRAIN_CONFIG_PATH)

    cfg_train = hydra.compose(
        config_name=TRAIN_CONFIG_NAME,
        overrides=[
            f'group={str(SAVE_DIR)}',
            f'cache.cache_path={str(SAVE_DIR)}/cache',
            f'experiment_name={EXPERIMENT_TRAIN}',
            f'job_name={JOB_NAME}',
            # 把 Lightning 的输出根目录锁到 LOG_DIR，方便找 ckpt
            f'lightning.trainer.params.default_root_dir={str(LOG_DIR)}',

            'py_func=train',
            '+training=training_raster_model',
            'scenario_builder=nuplan_mini',

            # 注意：这里不要太小，否则 val 集为 0，会触发 AssertionError
            'scenario_filter.limit_total_scenarios=20',

            # 为了在你这台机器上轻量一点
            'lightning.trainer.params.accelerator=ddp_spawn',
            'lightning.trainer.params.max_epochs=1',
            'data_loader.params.batch_size=2',
            'data_loader.params.num_workers=0',
        ],
    )

    # 跑训练
    main_train(cfg_train)

    # 训练完再搜一次 ckpt
    new_ckpt = find_ckpt(SAVE_DIR)
    if new_ckpt is None:
        raise RuntimeError(
            f'训练结束但在 {SAVE_DIR} 下没有找到 .ckpt 文件，请用 find /tmp -name \"*.ckpt\" 手动看一下实际存到哪里了。'
        )

    CHECKPOINT_PATH = str(new_ckpt)
    print(f'训练完成，找到 checkpoint：{CHECKPOINT_PATH}')

# --------- 2. 用 ml_planner 跑 simulation（这里是你要 debug 的部分） ---------
EXPERIMENT_SIM = 'simulation_ml_planner_experiment'
PLANNER = 'ml_planner'
CHALLENGE = 'closed_loop_nonreactive_agents'  # 也可以先用 open_loop_boxes，收敛快一点

DATASET_PARAMS = [
    'scenario_builder=nuplan_mini',
    'scenario_filter=all_scenarios',
    # 为了 debug + 内存友好，先只选一小部分
    'scenario_filter.scenario_types=[near_multiple_vehicles]',
    'scenario_filter.num_scenarios_per_type=1',
    'scenario_filter.limit_total_scenarios=5',
]

hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize(config_path=SIM_CONFIG_PATH)

cfg_sim = hydra.compose(
    config_name=SIM_CONFIG_NAME,
    overrides=[
        f'experiment_name={EXPERIMENT_SIM}',
        f'group={SAVE_DIR}',
        f'job_name={JOB_NAME_SIM}',
        # 关键：指定 ml_planner + 对应模型和 ckpt
        'model=raster_model',
        f'planner={PLANNER}',
        'planner.ml_planner.model_config=${model}',
        f'planner.ml_planner.checkpoint_path="{CHECKPOINT_PATH}"',

        f'+simulation={CHALLENGE}',
        *DATASET_PARAMS,
    ],
)

# 👉 你要 debug 的话，可以在 ml_planner 的 compute_trajectory 之类地方打断点
# 或者在这里加一行：
# import pdb; pdb.set_trace()
main_simulation(cfg_sim)

ml_planner_simulation_folder = cfg_sim.output_dir

# --------- 3. nuBoard（现在你先注释着也行）---------
# hydra.core.global_hydra.GlobalHydra.instance().clear()
# hydra.initialize(config_path=NUBOARD_CONFIG_PATH)
#
# cfg_nuboard = hydra.compose(
#     config_name=NUBOARD_CONFIG_NAME,
#     overrides=[
#         'scenario_builder=nuplan_mini',
#         f'simulation_path={[ml_planner_simulation_folder]}',
#     ],
# )
#
# main_nuboard(cfg_nuboard)
