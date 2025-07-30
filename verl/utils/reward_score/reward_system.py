import os

import backoff
import requests
from dotenv import load_dotenv
from ratelimit import limits, sleep_and_retry

load_dotenv()

@sleep_and_retry
@limits(calls=1, period=1)
@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=None)
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
        res = requests.post(API_URL, json=payload)
        res.raise_for_status()
        data = res.json()
        reward = data.get("final_eval_score")
        if reward is None:
            raise ValueError("Missing final_eval_score in API response")
        return reward
    except Exception as e:
        print(f"[RewardAPI Error] {e}")
        return 0.0  # Fallback reward
