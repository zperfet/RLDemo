"""
ART 2048 训练脚本
使用 OpenPipe ART 框架训练 Qwen 3 14B 模型玩 2048 游戏
"""

import asyncio
import math
import os
import random
import string
import xml.etree.ElementTree as ET
from typing import Literal, Optional, TypedDict

import art
import requests
import weave
from art.serverless.backend import ServerlessBackend
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

# ============================================================================
# 配置和常量
# ============================================================================

# 加载环境变量
load_dotenv()

# 设置 WANDB API Key（可以从环境变量或 .env 文件读取）
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "5c15d7b3b0b4e432f23b799599edf9125439e358")
os.environ["WANDB_API_KEY"] = WANDB_API_KEY

if not os.environ.get("WANDB_API_KEY"):
    raise ValueError("WANDB_API_KEY is required for inference, training, and logging to Weights & Biases.")

# 游戏配置
WINNING_VALUE = 64  # 为了加快训练，从 2048 降低到 64
BOARD_LENGTH = 4

# 训练配置
MODEL_NAME = "agent-001"
PROJECT_NAME = "2048"
BASE_MODEL = "OpenPipe/Qwen3-14B-Instruct"
TRAINING_STEPS = 20
GAMES_PER_STEP = 18
LEARNING_RATE = 1e-5

# 随机种子
random.seed(42)


# ============================================================================
# 游戏数据结构
# ============================================================================

class TwentyFortyEightGame(TypedDict):
    """2048 游戏状态"""
    id: str
    board: list[list[Optional[int]]]


# ============================================================================
# 游戏逻辑函数
# ============================================================================

def populate_random_cell(game: TwentyFortyEightGame) -> None:
    """在棋盘上随机填充一个 2 或 4"""
    all_clear_coordinates = [
        (i, j)
        for i in range(len(game["board"]))
        for j in range(len(game["board"][i]))
        if game["board"][i][j] is None
    ]
    if not all_clear_coordinates:
        return
    
    random_clear_coordinates = random.choice(all_clear_coordinates)
    # 90% 概率填充 2，10% 概率填充 4
    game["board"][random_clear_coordinates[0]][random_clear_coordinates[1]] = (
        2 if random.random() < 0.9 else 4
    )


def generate_game(board_length: int = BOARD_LENGTH) -> TwentyFortyEightGame:
    """生成新的 2048 游戏"""
    # 生成随机 6 位字符串作为游戏 ID
    game_id = "".join(random.choices(string.ascii_letters + string.digits, k=6))
    game = {
        "id": game_id,
        "board": [[None for _ in range(board_length)] for _ in range(board_length)],
    }

    # 填充两个随机格子
    populate_random_cell(game)
    populate_random_cell(game)

    return game


def render_board(game: TwentyFortyEightGame) -> str:
    """
    将棋盘渲染为人类可读的格式
    
    示例输出:
    _    | 2    | _    | 4
    4    | 8    | 2    | 16
    16   | 32   | 64   | 128
    _    | 2    | 2    | 4
    """
    board = game["board"]
    
    # 计算最大单元格宽度以对齐
    if any(cell is not None for row in board for cell in row):
        max_cell_width = max(
            [len(str(cell)) for row in board for cell in row if cell is not None]
        )
    else:
        max_cell_width = 1

    board_str = ""
    for row in board:
        # 用空格填充单元格使其宽度相同
        board_str += "|".join(
            [
                str(cell).rjust(max_cell_width)
                if cell is not None
                else "_".rjust(max_cell_width)
                for cell in row
            ]
        )
        board_str += "\n"
    return board_str


def condense_sequence(sequence: list[Optional[int]]) -> list[Optional[int]]:
    """
    压缩序列，优先匹配序列开头的元素
    序列应该从最远的方向开始传递，以便在压缩棋盘时使用
    """
    condensed_sequence = []
    gapless_sequence = [cell for cell in sequence if cell is not None]

    i = 0
    while i < len(gapless_sequence):
        if (
            i + 1 < len(gapless_sequence)
            and gapless_sequence[i] == gapless_sequence[i + 1]
        ):
            # 合并相同数字
            condensed_sequence.append(gapless_sequence[i] * 2)
            i += 2
        else:
            condensed_sequence.append(gapless_sequence[i])
            i += 1

    # 用 None 填充序列到固定长度
    return condensed_sequence + [None] * (BOARD_LENGTH - len(condensed_sequence))


def condense_board(
    game: TwentyFortyEightGame, direction: Literal["left", "right", "up", "down"]
) -> None:
    """按指定方向压缩棋盘"""
    if direction == "left":
        for row in game["board"]:
            condensed_row = condense_sequence(row)
            for i in range(len(row)):
                row[i] = condensed_row[i]

    elif direction == "right":
        for row in game["board"]:
            reversed_row = row[::-1]
            # 压缩前后都要反转
            condensed_row = condense_sequence(reversed_row)[::-1]
            for i in range(len(row)):
                row[i] = condensed_row[i]

    elif direction == "up":
        for col_index in range(len(game["board"][0])):
            column = [row[col_index] for row in game["board"]]
            condensed_column = condense_sequence(column)
            for row_index in range(len(column)):
                game["board"][row_index][col_index] = condensed_column[row_index]

    elif direction == "down":
        for col_index in range(len(game["board"][0])):
            column = [row[col_index] for row in game["board"]]
            reversed_column = column[::-1]
            condensed_column = condense_sequence(reversed_column)[::-1]
            for row_index in range(len(column)):
                game["board"][row_index][col_index] = condensed_column[row_index]


def apply_agent_move(game: TwentyFortyEightGame, move_xml: str) -> None:
    """将智能体的移动应用到游戏棋盘"""
    try:
        root = ET.fromstring(move_xml)
        direction = root.text
    except Exception:
        raise ValueError("Invalid xml")

    if direction not in ["left", "right", "up", "down"]:
        raise ValueError(f"Invalid direction: {direction}")

    condense_board(game, direction)
    populate_random_cell(game)


def max_cell_value(game: TwentyFortyEightGame) -> int:
    """返回棋盘上的最大单元格值"""
    return max([cell for row in game["board"] for cell in row if cell is not None])


def check_game_finished(game: TwentyFortyEightGame) -> bool:
    """检查游戏是否结束"""
    # 如果达到获胜值，游戏结束
    if max_cell_value(game) >= WINNING_VALUE:
        return True

    # 如果还有空单元格，游戏继续
    if any(cell is None for row in game["board"] for cell in row):
        return False

    # 棋盘已满，游戏结束
    return True


def total_board_value(game: TwentyFortyEightGame) -> int:
    """返回棋盘上所有单元格值的总和"""
    return sum([cell for row in game["board"] for cell in row if cell is not None])


# ============================================================================
# 强化学习相关代码
# ============================================================================

class Scenario2048(BaseModel):
    """2048 训练场景配置"""
    step: int


@weave.op
@art.retry(exceptions=(requests.ReadTimeout))
async def rollout(model: art.Model, scenario: Scenario2048) -> art.Trajectory:
    """
    执行一次 rollout（一个完整的游戏回合）
    生成轨迹用于训练模型
    """
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    game = generate_game()
    move_number = 0

    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": (
                    "You are an excellent 2048 player. Always choose the move most likely "
                    "to lead to combine cells to eventually reach the number 2048. "
                    "Optional moves are 'left', 'right', 'up', 'down'. "
                    "Return your move as an XML object with a single property 'move', "
                    "like so: <move>left</move>"
                ),
            }
        ],
        metadata={
            "game_id": game["id"],
            "notebook-id": "2048",
            "step": scenario.step,
        },
        reward=0,
    )

    while True:
        # 将当前棋盘状态发送给模型
        trajectory.messages_and_choices.append(
            {"role": "user", "content": render_board(game)}
        )

        try:
            messages = trajectory.messages()
            chat_completion = await client.chat.completions.create(
                max_completion_tokens=128,
                messages=messages,
                model=model.get_inference_name(),
            )
        except Exception as e:
            print(f"Error generating chat completion: {e}")
            raise e

        choice = chat_completion.choices[0]
        content = choice.message.content
        assert isinstance(content, str)
        trajectory.messages_and_choices.append(choice)

        # 应用模型选择的移动
        try:
            apply_agent_move(game, content)
            move_number += 1
        except ValueError:
            # 无效移动，给予负奖励
            trajectory.reward = -1
            break

        # 检查游戏是否结束
        if check_game_finished(game):
            max_value = max_cell_value(game)
            board_value = total_board_value(game)
            trajectory.metrics["max_value"] = max_value
            trajectory.metrics["board_value"] = board_value
            trajectory.metrics["move_number"] = move_number

            # 计算奖励
            # 优先接近获胜值，其次最大化棋盘总值，最重要的是赢得游戏！
            if max_value < WINNING_VALUE:
                # 将最大值按对数缩放到 0-1 之间
                max_value_reward = (math.log(max_value, 2) - 1) / (
                    math.log(WINNING_VALUE, 2) - 1
                )
                # 将棋盘总值按对数缩放到 0-1 之间
                board_value_reward = (math.log(board_value, 2) - 1) / (
                    math.log(WINNING_VALUE * 16, 2) - 1
                )
                # 组合两个奖励，最大值权重更高
                trajectory.reward = max_value_reward + (board_value_reward * 0.2)
            else:
                # 如果智能体获胜，给予双倍奖励
                trajectory.reward = 2
            break

    return trajectory


async def train_model():
    """训练模型的主函数"""
    # 初始化 Weave
    weave.init(PROJECT_NAME, settings={"print_call_link": False})

    # 声明模型
    model = art.TrainableModel(
        name=MODEL_NAME,
        project=PROJECT_NAME,
        base_model=BASE_MODEL,
    )

    # 初始化服务器后端
    # 训练和推理将在 Weights & Biases 服务器上运行
    backend = ServerlessBackend()

    # 注册模型到 Serverless Backend（设置日志、推理和训练）
    await model.register(backend)

    # 训练循环
    current_step = await model.get_step()
    print(f"Starting training from step {current_step} to {TRAINING_STEPS}")

    for i in range(current_step, TRAINING_STEPS):
        print(f"\n=== Training Step {i} ===")
        
        # 收集轨迹
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, Scenario2048(step=i)) for _ in range(GAMES_PER_STEP)
                )
                for _ in range(1)
            ),
            pbar_desc="gather",
            max_exceptions=GAMES_PER_STEP,
        )
        
        # 删除表现差的检查点
        await model.delete_checkpoints('train/reward')
        
        # 训练模型
        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=LEARNING_RATE),
        )
        
        print(f"Step {i} completed")

    print("\nTraining completed!")
    return model


async def evaluate_model(model: art.Model):
    """评估训练好的模型"""
    last_step = await model.get_step()
    deployed_inference_model_name = f"{model.get_inference_name()}:step{last_step}"

    print(f"\n=== Evaluating Model ===")
    print(f"Model: {deployed_inference_model_name}")

    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    game = generate_game()
    move_number = 0

    messages = [
        {
            "role": "system",
            "content": (
                "You are an excellent 2048 player. Always choose the move most likely "
                "to lead to combine cells to eventually reach the number 2048. "
                "Optional moves are 'left', 'right', 'up', 'down'. "
                "Return your move as an XML object with a single property 'move', "
                "like so: <move>left</move>"
            ),
        },
    ]

    while not check_game_finished(game):
        rendered_board = render_board(game)
        messages.append({"role": "user", "content": rendered_board})

        try:
            response = await client.chat.completions.create(
                model=deployed_inference_model_name,
                messages=messages,
            )
            content = response.choices[0].message.content
        except Exception as e:
            print(f"Error generating chat completion: {e}")
            raise e

        messages.append({"role": "assistant", "content": content})

        try:
            apply_agent_move(game, content)
            move_number += 1
        except ValueError:
            raise ValueError(f"Invalid move on move {move_number}: {content}")

        # 每 10 步打印一次棋盘
        if move_number % 10 == 0:
            print(f"\n--- Move {move_number} ---")
            print(f"Board:\n{rendered_board}")
            print(f"Agent move: {content}")
            print(f"Updated board:\n{render_board(game)}")

    # 游戏结束，打印结果
    print(f"\n=== Game Finished ===")
    print(f"Total moves: {move_number}")

    max_value = max_cell_value(game)
    board_value = total_board_value(game)

    if max_value >= WINNING_VALUE:
        print("🎉 Game won! 💪")
    else:
        print("😢 Game lost!")

    print(f"\nFinal board:\n{render_board(game)}")
    print(f"Max value: {max_value}")
    print(f"Board value: {board_value}")


# ============================================================================
# 主函数
# ============================================================================

async def main():
    """主函数"""
    print("=" * 60)
    print("ART 2048 Training Script")
    print("=" * 60)
    
    # 训练模型
    model = await train_model()
    
    # 评估模型
    await evaluate_model(model)
    
    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
