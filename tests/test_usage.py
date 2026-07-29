"""app.usage: normalizing provider usage dicts into metering columns."""

from __future__ import annotations

from sqlalchemy import select

from app.admin.usage import _cost_expr
from app.orm import Client, Model, UsageLog
from app.proxy.anthropic import _usage_from_anthropic
from app.proxy.chat_completions import _chat_usage_to_azure_usage
from app.usage import tokens_from_usage


def test_anthropic_usage_carries_cache_fields():
    payload = {
        "usage": {
            "input_tokens": 100,
            "output_tokens": 40,
            "cache_creation_input_tokens": 2000,
            "cache_read_input_tokens": 30000,
        }
    }
    u = _usage_from_anthropic(payload)
    assert u["input_tokens"] == 100
    assert u["output_tokens"] == 40
    assert u["total_tokens"] == 140
    assert u["cache_creation_input_tokens"] == 2000
    assert u["cache_read_input_tokens"] == 30000


def test_anthropic_usage_without_cache_fields_is_none_not_zero():
    u = _usage_from_anthropic({"usage": {"input_tokens": 5, "output_tokens": 3}})
    assert u["cache_creation_input_tokens"] is None
    assert u["cache_read_input_tokens"] is None


def test_flat_shape_passes_through():
    tokens = tokens_from_usage(
        {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "cache_creation_input_tokens": 7,
            "cache_read_input_tokens": 8,
        }
    )
    assert tokens == {
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
        "cache_creation_input_tokens": 7,
        "cache_read_input_tokens": 8,
    }


def test_openai_responses_shape_nests_cached_tokens():
    # Raw Responses API usage, as captured from response.completed events and
    # non-streaming bodies: cached reads live in input_tokens_details.
    tokens = tokens_from_usage(
        {
            "input_tokens": 1000,
            "output_tokens": 50,
            "total_tokens": 1050,
            "input_tokens_details": {"cached_tokens": 900},
        }
    )
    assert tokens["cache_read_input_tokens"] == 900
    assert tokens["cache_creation_input_tokens"] is None


def test_openai_chat_shape_maps_prompt_token_details():
    tokens = tokens_from_usage(
        _chat_usage_to_azure_usage(
            {
                "prompt_tokens": 500,
                "completion_tokens": 25,
                "total_tokens": 525,
                "prompt_tokens_details": {"cached_tokens": 400},
            }
        )
    )
    assert tokens == {
        "input_tokens": 500,
        "output_tokens": 25,
        "total_tokens": 525,
        "cache_creation_input_tokens": None,
        "cache_read_input_tokens": 400,
    }


def test_empty_usage_yields_all_none():
    assert all(v is None for v in tokens_from_usage({}).values())


async def test_cost_expr_prices_every_token_bucket(session):
    client = Client(name="tester", kind="app")
    model = Model(
        deployment_name="claude-opus-5",
        target_uri="https://api.anthropic.com/v1/messages",
        cost_per_million_input=5,
        cost_per_million_output=25,
        cost_per_million_cache_write=6.25,
        cost_per_million_cache_read=0.5,
    )
    session.add_all([client, model])
    await session.flush()
    session.add(
        UsageLog(
            client_id=client.id,
            model_name="claude-opus-5",
            endpoint="messages",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_creation_input_tokens=1_000_000,
            cache_read_input_tokens=1_000_000,
            status_code=200,
        )
    )
    await session.commit()

    cost = (
        await session.execute(
            select(_cost_expr())
            .select_from(UsageLog)
            .outerjoin(Model, UsageLog.model_name == Model.deployment_name)
        )
    ).scalar_one()
    # 5 + 25 + 6.25 + 0.5 dollars — one million tokens in each bucket.
    assert float(cost) == 36.75


async def test_cost_expr_treats_null_cache_rates_as_zero(session):
    client = Client(name="tester", kind="app")
    model = Model(
        deployment_name="gpt",
        target_uri="https://api.openai.com/v1/responses",
        cost_per_million_input=2,
        cost_per_million_output=8,
    )
    session.add_all([client, model])
    await session.flush()
    session.add(
        UsageLog(
            client_id=client.id,
            model_name="gpt",
            endpoint="responses",
            input_tokens=500_000,
            output_tokens=250_000,
            cache_read_input_tokens=400_000,
            status_code=200,
        )
    )
    await session.commit()

    cost = (
        await session.execute(
            select(_cost_expr())
            .select_from(UsageLog)
            .outerjoin(Model, UsageLog.model_name == Model.deployment_name)
        )
    ).scalar_one()
    assert float(cost) == 3.0
