import os

import requests
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

MAX_RETRIES = 3
BASE_DELAY = 10
MAX_WORKERS = 16
TIMEOUT = 600

load_dotenv()

def get_reward_score(
    instruction: str,
    response: str,
    ground_truth: str,
    agents_to_force_enable: list[str] = None,
    agents_to_force_disable: list[str] = None,
    extra_info=None,
) -> float:
    """
    Call external reward API to get reward score.

    Args:
        prompt (str): The input prompt.
        response (str): The generated response.
        api_url (str): The reward API endpoint.
        metadata (dict, optional): Additional metadata to send in the request.

    Returns:
        float: Reward score returned by
    """
    
    payload = {
        "instruction": instruction,
        "response": response,
        "ground_truth": ground_truth,
        "langsmith_tracing": {
            "run_name": "agentic-reward-system-run",
            "run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "tags": ["string"],
            "metadata": {"additionalProp1": {}},
        },
        "agents_to_force_enable": agents_to_force_enable,
        "agents_to_force_disable": agents_to_force_disable,
    }

    API_URL = os.getenv("API_URL")

    try:
        res = requests.post(API_URL, json=payload, timeout=TIMEOUT)
        res.raise_for_status()
        data = res.json()
        reward = data.get("final_eval_score")
        if reward is None:
            raise ValueError("Missing final_eval_score in API response")
        return reward
    except Exception as e:
        print(f"[RewardAPI Error] {e}")
        return 5.0  # Fallback reward

def get_reward_score_batch(
    instructions: list[str],
    responses: list[str],
    ground_truths: list[str],
    agents_to_force_enable_list: list[list[str]] = None,
    agents_to_force_disable_list: list[list[str]] = None,
    extra_infos: list = None,
) -> list[float]:
    """
    Batch version with threading for parallel calls.
    """
    print("in get_reward_score_batch")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i, instruction in enumerate(instructions):
            agents_enable = (
                agents_to_force_enable_list[i] if agents_to_force_enable_list else None
            )
            agents_disable = (
                agents_to_force_disable_list[i] if agents_to_force_disable_list else None
            )
            extra_info = extra_infos[i] if extra_infos else None

            futures.append(
                executor.submit(
                    get_reward_score,
                    instruction,
                    responses[i],
                    ground_truths[i],
                    agents_enable,
                    agents_disable,
                    extra_info,
                )
            )

        results = [future.result() for future in futures]
    return results