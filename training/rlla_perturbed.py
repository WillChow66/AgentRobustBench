"""
Modified ToolRL reward function for perturbed training data.

Drop-in replacement for verl/utils/reward_score/rlla.py
Only change: response_only samples get correctness signal (not 0.0).

Usage:
  Copy this file to ToolRL/verl/utils/reward_score/rlla.py
  OR import and override in main_ppo.py
"""

import re
import json
import os
from collections import Counter


def match_score(list1, list2):
    """Compute a similarity score considering element frequency, ignoring order."""
    if list1 == list2:
        return 1.0

    if os.getenv("REFINEDREWARD", 0) == "1":
        if list1 != list2:
            return 0.0

    if not list1 or not list2:
        return 0.0

    count1 = Counter(list1)
    count2 = Counter(list2)

    intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
    max_possible = len(list1) + len(list2) - intersection

    return intersection / max_possible if max_possible > 0 else 0.0


def customize_format_reward_func(completions, answer, step, max_possible_reward, min_possible_reward, **kwargs):
    """Format reward: checks <think>, <tool_call>, <response> structure. Unchanged from ToolRL."""
    if str(os.getenv("MAX1STEP30MAX3", 0)) == "1":
        if step >= 30:
            max_possible_reward = max_possible_reward / 2
            min_possible_reward = min_possible_reward / 2

    if str(os.getenv("SCHEDULEREWARD", 0)) == "1":
        max_possible_reward = 2 - (2 - max_possible_reward) * step / 150
        min_possible_reward = -2 + (2 + min_possible_reward) * step / 150
        if max_possible_reward < 1.0:
            max_possible_reward = 1.0
        if min_possible_reward > -1.0:
            min_possible_reward = -1.0

    rewards = []
    responses = [completion[0]['content'] for completion in completions]

    for response, ans in zip(responses, answer):
        reward = min_possible_reward
        if "<response>" in ans and "<tool_call>" not in ans:
            pattern = r"^<think>.*?</think>\n<response>.*?</response>$"
            if re.search(pattern, response, re.DOTALL) and response.count("<response>") == 1 and response.count("</response>") == 1:
                reward = max_possible_reward
        elif "<response>" not in ans and "<tool_call>" in ans:
            pattern = r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>$"
            if re.search(pattern, response, re.DOTALL) and response.count("<tool_call>") == 1 and response.count("</tool_call>") == 1:
                reward = max_possible_reward
        elif "<response>" in ans and "<tool_call>" in ans:
            pattern = r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>\n<response>.*?</response>$"
            if re.search(pattern, response, re.DOTALL) and response.count("<tool_call>") == 1 and response.count("</tool_call>") == 1 and response.count("<response>") == 1 and response.count("</response>") == 1:
                reward = max_possible_reward
        else:
            pattern = r"^<think>.*?</think>$"
            if re.search(pattern, response, re.DOTALL):
                reward = max_possible_reward

        rewards.append(reward)

    return rewards


def customize_length_reward_func(completions, answer, step, max_possible_reward, min_possible_reward, **kwargs):
    """Length reward for <think> section. Unchanged from ToolRL."""
    if os.getenv("SCHEDULELENGTH", 0) == "1":
        max_reward_len = (640 - 384) * step / 105 + 384
    else:
        max_reward_len = 512

    responses = [completion[0]['content'] for completion in completions]
    rewards = []

    for response, ans in zip(responses, answer):
        if "<think>" not in response or "</think>" not in response:
            rewards.append(min_possible_reward)
            continue
        think_responses = response.split("<think>")[-1].split("</think>")[0].strip()
        reward = round(len(think_responses.split()) / max_reward_len, 2)
        if reward > 1.0:
            reward = 1.0

        final_reward = reward * (max_possible_reward - min_possible_reward) + min_possible_reward
        rewards.append(final_reward)

    return rewards


def compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward, min_possible_reward):
    """Tool call scoring with fuzzy matching. Unchanged from ToolRL."""
    if gt_tools == pd_tools:
        return max_possible_reward

    if os.getenv("COARSEREWARD", 0) == "1":
        if gt_tools != pd_tools:
            return min_possible_reward

    gt_names = [tool["name"] for tool in gt_tools]
    pd_names = [tool["name"] for tool in pd_tools]
    score = match_score(list(gt_names), list(pd_names))

    local_max_possible = 1.0
    used_pd_indices = set()

    for gt_tool in gt_tools:
        gt_name = gt_tool["name"]
        gt_params = gt_tool["parameters"]

        if str(os.getenv("INTERMEDIATEREWARD", 0)) == "1":
            local_max_possible += 1.0
        else:
            local_max_possible += 1.0 + len(gt_params)

        best_match = None
        best_match_score = 0.0
        best_match_index = -1

        for i, pd_tool in enumerate(pd_tools):
            if i in used_pd_indices or pd_tool["name"] != gt_name:
                continue

            if str(os.getenv("INTERMEDIATEREWARD", 0)) == "1":
                if gt_tool == pd_tool:
                    best_match = pd_tool
                    best_match_index = i
                    best_match_score = 1.0
                    break
                else:
                    continue

            pd_params = pd_tool["parameters"]
            param_score = match_score(list(gt_params.keys()), list(pd_params.keys()))
            correctness_score = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)
            total_score = param_score + correctness_score

            if total_score > best_match_score:
                best_match_score = total_score
                best_match = pd_tool
                best_match_index = i

        if best_match:
            used_pd_indices.add(best_match_index)
            score += best_match_score

    return (max_possible_reward - min_possible_reward) * score / local_max_possible + min_possible_reward


def customize_correctness_reward_tool(completions, answer, step, max_possible_reward, min_possible_reward, **kwargs):
    """
    Tool call correctness reward.

    *** MODIFIED from ToolRL ***
    Change: response_only samples now receive correctness signal.
    - Original ToolRL: response_only → 0.0 (no signal)
    - Modified: response_only correct (no tool_call in response) → +max/3
    - Modified: response_only wrong (has tool_call in response) → +min/3
    """
    if str(os.getenv("MAX1STEP30MAX3", 0)) == "1":
        if step < 30:
            max_possible_reward = max_possible_reward / 3
            min_possible_reward = min_possible_reward / 3

    if str(os.getenv("SCHEDULEREWARD", 0)) == "1":
        max_possible_reward = (max_possible_reward - 2) * step / 150 + 2
        min_possible_reward = (min_possible_reward + 2) * step / 150 - 2
        if max_possible_reward > 3.0:
            max_possible_reward = 3.0
        if min_possible_reward < -3.0:
            min_possible_reward = -3.0

    responses = [completion[0]['content'] for completion in completions]
    rewards = []

    for response, ans in zip(responses, answer):
        reward = 0.0

        if "<tool_call>" not in ans:
            # ═══ MODIFIED: response_only now gets signal ═══
            if "<tool_call>" not in response and "</tool_call>" not in response:
                reward = max_possible_reward / 3  # Correct: didn't call tool
            else:
                reward = min_possible_reward / 3  # Wrong: called tool when shouldn't
            rewards.append(reward)
            continue

        gt_tool_call = ans.split("<tool_call>")[1].split("</tool_call>")[0].strip()
        gt_tools = gt_tool_call.split("\n")
        gt_tools = [json.loads(tool) for tool in gt_tools]

        try:
            assert "<tool_call>" in response
            assert "</tool_call>" in response
            pd_tools = response.split("<tool_call>")[1].split("</tool_call>")[0].strip().split("\n")
            pd_tools = [json.loads(tool) for tool in pd_tools]
            reward = compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward, min_possible_reward)
        except:
            reward = min_possible_reward

        rewards.append(reward)

    return rewards


def compute_score(solution_str, ground_truth, step=0):
    """Main scoring function. Compatible with ToolRL trainer."""
    exp_name = str(os.getenv("EXPERIMENT_NAME", ""))
    if "llama" in exp_name:
        predict_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()
    elif "qwen" in exp_name:
        predict_str = solution_str.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
    else:
        raise NotImplementedError(f"Unknown model name: {exp_name}")

    if str(os.getenv("CORRECTMAX1", 0)) == "1":
        tool_max_possible = 1.0
        tool_min_possible = -1.0
    else:
        tool_max_possible = 3.0
        tool_min_possible = -3.0

    format_max_possible = 1.0
    format_min_possible = 0.0

    length_max_possible = 1.0
    length_min_possible = 0.0

    completions = [[{"role": "assistant", "content": predict_str}]]
    answer = [ground_truth]

    fomrat_score = customize_format_reward_func(completions, answer, step, format_max_possible, format_min_possible)[0]
    correctness_score = customize_correctness_reward_tool(completions, answer, step, tool_max_possible, tool_min_possible)[0]

    if str(os.getenv("WITHLENGTH", 0)) == "1":
        length_score = customize_length_reward_func(completions, answer, step, length_max_possible, length_min_possible)[0]
    else:
        length_score = 0

    score = fomrat_score + correctness_score + length_score

    return score, fomrat_score, correctness_score, length_score
