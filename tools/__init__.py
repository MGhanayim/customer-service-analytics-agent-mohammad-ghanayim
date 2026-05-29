"""Tool registry.

Assembles every tool into a single ``all_tools`` list that the agent binds to
the LLM in one line (`llm.bind_tools(all_tools)`). Consumers import from here:

    from tools import all_tools

Grouped by module so the layering stays legible: deterministic data/display
tools, the LLM-powered summary + recommender, and the profile (semantic memory)
tools.
"""

from tools.data_tools import (
    count_rows,
    filter_by_category,
    filter_by_intent,
    get_distribution,
    list_unique_values,
)
from tools.display_tools import find_instructions_by_keyword, show_examples
from tools.profile_tools import get_user_profile, update_user_profile
from tools.recommend_tools import suggest_query
from tools.summary_tools import summarize_responses

all_tools = [
    # data_tools (deterministic aggregates)
    count_rows,
    filter_by_category,
    filter_by_intent,
    get_distribution,
    list_unique_values,
    # display_tools (raw-row inspection)
    show_examples,
    find_instructions_by_keyword,
    # summary_tools (LLM-powered)
    summarize_responses,
    # profile_tools (semantic memory — Task 2b)
    get_user_profile,
    update_user_profile,
    # recommend_tools (Bonus B)
    suggest_query,
]
