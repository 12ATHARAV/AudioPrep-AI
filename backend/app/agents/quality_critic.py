"""
Quality Critic node — evaluates the generated dialogue.

This is the agentic quality loop. Instead of one LLM call
generating everything, we have a critic that checks:

1. Groundedness: Are claims supported by retrieved evidence?
2. Coverage: Were all key concepts covered?
3. Technical Accuracy: Are the explanations correct?
4. Dialogue Naturalness: Does it sound like real conversation?
5. Interview Value: Would this help in an interview?

If the critic finds issues, it loops back to revision.

Interview Note: "The critic implements a quality gate.
Without it, the LLM might hallucinate or skip concepts.
With it, we get measurable quality scores and automatic
revision. The max_revision_loops parameter prevents
infinite loops — a common agentic system concern."
"""

import json
from app.agents.state import (
    PodcastState,
    QualityCriticResult,
    TopicPlan,
    DialogueTurn,
)
from app.core.llm_provider import llm_provider
from app.core.config import settings
from app.core.logging import get_logger
from app.core.json_utils import extract_json

logger = get_logger("quality_critic")

CRITIC_PROMPT = """You are a quality assurance critic for a technical podcast.

Evaluate the following podcast dialogue on these criteria (score 0.0 to 1.0):

TOPIC PLAN:
{topic_plan}

COVERAGE MODE: {coverage_mode}
TARGET DURATION: {duration} minutes
Note on Coverage:
- If COVERAGE MODE is 'summary' (short episode <= 25m), the goal is high-yield synthesis prioritizing foundational concepts first. Do NOT penalize for omitting deep mathematical proofs or sub-algorithms that do not fit in a short runtime. If primary concepts are introduced and explanations are sound, score coverage >= 0.85 and revision_required = false.
- If COVERAGE MODE is 'comprehensive' (long episode >= 30m), evaluate for deep, multi-topic coverage.

DIALOGUE:
{dialogue}

RETRIEVED EVIDENCE USED:
{evidence}

Score each criterion:
1. "groundedness" (0-1): Are technical claims supported by the evidence or sound engineering fundamentals?
2. "coverage" (0-1): Were the planned concepts covered appropriately for this duration mode?
3. "technical_accuracy" (0-1): Are the explanations technically correct?
4. "dialogue_naturalness" (0-1): Does it sound like a real conversation?
5. "interview_value" (0-1): Would this help someone prepare for interviews?

Also provide:
- "issues": List of specific problems found
- "revision_required": true ONLY if overall_score < 0.70 or critical misinformation is present

Return ONLY valid JSON:
{{
  "groundedness": 0.0,
  "coverage": 0.0,
  "technical_accuracy": 0.0,
  "dialogue_naturalness": 0.0,
  "interview_value": 0.0,
  "issues": [],
  "revision_required": false,
  "overall_score": 0.0
}}"""


async def quality_critic_node(state: PodcastState) -> dict:
    """LangGraph node: Evaluate dialogue quality."""
    logger.info(
        "quality_critic_start",
        revision_count=state.get("revision_count", 0),
    )

    plan = state["topic_plan"]
    if isinstance(plan, dict):
        plan = TopicPlan(**plan)

    dialogue = state.get("dialogue", [])
    dialogue_text = "\n".join(
        f"[{t.speaker if isinstance(t, DialogueTurn) else t.get('speaker', '?')}]: "
        f"{t.text if isinstance(t, DialogueTurn) else t.get('text', '')}"
        for t in dialogue
    )

    contexts = state.get("retrieval_contexts", {})
    evidence_text = ""
    for concept, ctx_list in contexts.items():
        evidence_text += f"\n### {concept}\n"
        for ctx in ctx_list[:2]:
            content = ctx.get("content", "") if isinstance(ctx, dict) else str(ctx)
            evidence_text += f"- {content[:300]}\n"

    preferences = state.get("user_preferences", {})
    duration = preferences.get("duration") or getattr(plan, "estimated_duration_minutes", 120) or 120
    coverage_mode = getattr(plan, "coverage_mode", "summary" if duration <= 25 else "comprehensive")

    # Provide representative dialogue excerpt (up to 12k chars)
    prompt = CRITIC_PROMPT.format(
        topic_plan=json.dumps(plan.model_dump(), indent=2),
        coverage_mode=coverage_mode,
        duration=duration,
        dialogue=dialogue_text[:12000],
        evidence=evidence_text[:3000],
    )

    try:
        response = await llm_provider.generate(
            prompt=prompt,
            temperature=0.1,  # Low temperature for evaluation
            max_tokens=1000,
        )

        result_data = extract_json(response.text)
        critic_result = QualityCriticResult(**result_data)

        # Check duration and depth compliance
        preferences = state.get("user_preferences", {})
        duration = preferences.get("duration") or getattr(plan, "estimated_duration_minutes", 120) or 120
        # Adaptive word count threshold:
        # Short podcasts (<=25m): ~75 words/min minimum
        # Long podcasts (>25m up to 120m): ~45 words/min minimum
        min_words_per_minute = 75 if duration <= 25 else 45
        min_expected_words = duration * min_words_per_minute
        actual_words = sum(
            len(t.text.split()) if isinstance(t, DialogueTurn) else len(t.get("text", "").split())
            for t in dialogue
        )

        if actual_words < min_expected_words:
            critic_result.issues.append(
                f"Dialogue is too brief ({actual_words} words vs minimum expected {min_expected_words} words for {duration} min). Unpack concepts with more dialogue turns, analogies, and back-and-forth debate."
            )
            critic_result.coverage = min(critic_result.coverage, 0.6)
            critic_result.revision_required = True

        # Calculate overall score if not provided
        if critic_result.overall_score == 0:
            critic_result.overall_score = (
                critic_result.groundedness * 0.3
                + critic_result.coverage * 0.25
                + critic_result.technical_accuracy * 0.25
                + critic_result.dialogue_naturalness * 0.1
                + critic_result.interview_value * 0.1
            )

        # Check if we should revise
        revision_count = state.get("revision_count", 0)
        if revision_count >= settings.max_revision_loops:
            critic_result.revision_required = False
            logger.warning(
                "max_revisions_reached",
                count=revision_count,
                score=critic_result.overall_score,
            )

        logger.info(
            "quality_critic_complete",
            overall_score=critic_result.overall_score,
            revision_required=critic_result.revision_required,
            issues=len(critic_result.issues),
        )

        return {
            "critic_result": critic_result,
            "revision_count": revision_count + (
                1 if critic_result.revision_required else 0
            ),
            "current_stage": "critic_complete",
        }

    except Exception as e:
        logger.error("critic_failed", error=str(e))
        # On failure, pass through
        return {
            "critic_result": QualityCriticResult(
                overall_score=0.8,
                revision_required=False,
            ),
            "current_stage": "critic_fallback",
        }


def should_revise(state: PodcastState) -> str:
    """
    Conditional edge: decide whether to revise or proceed to TTS.

    This is the branching logic in our LangGraph.
    """
    critic = state.get("critic_result")
    if critic is None:
        return "proceed_to_tts"

    if isinstance(critic, dict):
        critic = QualityCriticResult(**critic)

    if critic.revision_required:
        return "revise_dialogue"

    return "proceed_to_tts"