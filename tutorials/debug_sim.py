# tutorials/debug_sim.py

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from nuplan.planning.script.run_simulation import CONFIG_NAME, run_simulation
from nuplan.planning.script.builders.planner_builder import build_planners
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario

# 直接作为脚本跑
if __name__ == "__main__":
    # 1. 把之前 hydra 状态清一下（你前面跑过 training）
    GlobalHydra.instance().clear()

    # 2. 用绝对路径指向 simulation 的 config 目录
    repo_root = Path(__file__).resolve().parents[1]  # nuplan-devkit 根目录
    config_dir = str(repo_root / "nuplan/planning/script/config/simulation")
    print(f"Using config_dir = {config_dir}")

    with initialize_config_dir(config_dir=config_dir):
        # 3. 组 simulation 的 cfg
        cfg = compose(
            config_name=CONFIG_NAME,  # 一般就是 "default_simulation"
            overrides=[
                # 相当于 SkeletonTestSimulation.default_overrides 里的一部分
                "group=debug_sim",
                "experiment_name=debug_simulation",

                # 跑 mini 数据集（你可以按自己情况改）
                "scenario_builder=nuplan_mini",
                "scenario_filter=nuplan_mini",
                "scenario_filter.num_scenarios_per_type=1",

                # 观测和控制器（和 unittest 里一样）
                "observation=box_observation",
                "ego_controller=log_play_back_controller",

                # ⭐⭐ 关键：给 planner 选一个实现，不然 cfg.planner 不会存在
                "planner=simple_planner",   # 想用 ML 的话改成 ml_planner / imitation_planner
            ],
        )

        # 打个调试信息看一眼
        print("Top-level keys:", list(cfg.keys()))

        # 这里如果不再报错，说明 cfg.planner 已经有了
        print("planner config:\n", OmegaConf.to_yaml(cfg.planner))

        # 4. 手动构造 planner（跟 test_run_simulation 同一个套路）
        planners = build_planners(cfg.planner, MockAbstractScenario())
        planner = planners[0] if isinstance(planners, (list, tuple)) else planners

        # 5. 删除 cfg.planner，避免 run_simulation 再自己 build 一遍
        OmegaConf.set_struct(cfg, False)
        cfg.pop("planner")
        OmegaConf.set_struct(cfg, True)

        # ✅ 6. 在你 planner 的 step / compute_trajectory 里打断点，然后跑这一行
        run_simulation(cfg, planner)
