"""
Topic Planner node — analyzes document and creates a structured plan.

This node uses an LLM to analyze the document content and produce
a structured TopicPlan with learning goals, dependencies, and
interview focus areas.

Interview Note: "The planner doesn't just say 'make a podcast
about Docker.' It produces a structured plan: what concepts to
cover, in what order (dependencies), what interview traps to
include, and estimated duration. This structure drives the
dialogue generator to produce coherent, progressive content."
"""

import json
from app.agents.state import PodcastState, TopicPlan, TopicPlanSegment
from app.core.config import settings
from app.core.llm_provider import llm_provider
from app.core.logging import get_logger
from app.core.json_utils import extract_json

logger = get_logger("topic_planner")


PLANNER_PROMPT = """You are the Lead Executive Producer and Content Architect for a top-tier educational technology podcast.

Analyze the entire document below and create a structured master plan for a {difficulty}-level podcast episode.
Target Duration: {duration} minutes.
Focus/Style: {focus}

DOCUMENT CONTENT:
---
{content}
---

PACING & SCOPE GUIDELINES BASED ON USER DURATION ({duration} minutes):
1. IF DURATION IS SHORT ({duration} <= 25 minutes, e.g. 10m or 20m) for a multi-topic document:
   - The user requested a summarized/condensed version.
   - Do NOT spend the entire episode on just the first 1-2 pages and run out of time!
   - Identify all major topics across the document, but prioritize the MOST IMPORTANT, core foundational concepts first.
   - Synthesize the remaining concepts into high-yield takeaways and key insights so the listener gains an executive, end-to-end understanding without running out of time.
   - Set "coverage_mode": "summary".
   - Group concepts into 3-4 high-yield priority segments.

2. IF DURATION IS LONG ({duration} >= 30 minutes, especially 60-120 minutes / 2 hours):
   - The user requested a comprehensive, in-depth deep-dive masterclass.
   - You MUST cover ALL major topics, sections, and chapters present in the entire document from start to finish (e.g. for OS: fundamentals, kernels, processes, threads, CPU scheduling, synchronization, deadlocks, memory management, virtual memory, page replacement, storage, rapid-fire questions).
   - Do NOT skip or omit any sections.
   - Set "coverage_mode": "comprehensive".
   - Divide the episode into 5-8 progressive sequential segments ("segments"), each dedicated to thoroughly exploring a cluster of concepts with deep technical mechanics, vivid real-world analogies, architectural trade-offs, and tricky interview questions.

OUTPUT FORMAT:
Return ONLY a valid JSON object:
{{
  "topic": "Comprehensive main title/topic of the document",
  "difficulty": "{difficulty}",
  "coverage_mode": "summary" or "comprehensive",
  "learning_goals": ["5-10 specific learning goals spanning the planned scope"],
  "dependencies": ["Ordering of concepts that must be explained before others"],
  "key_concepts": ["All important concepts to be covered in this episode"],
  "interview_focus": ["Tricky interview questions and common traps across these topics"],
  "estimated_duration_minutes": {duration},
  "segments": [
    {{
      "title": "Segment Title (e.g. Act 1: OS Core & Kernel Mechanics)",
      "concepts": ["Concept A", "Concept B"],
      "description": "What this segment covers and how it transitions"
    }}
  ]
}}"""


def _prepare_document_content(state: PodcastState) -> str:
    """
    Prepare complete document representation for the planner.
    Ensures that long documents are never prematurely truncated.
    """
    raw_text = state.get("raw_text", "")
    if not raw_text:
        return "No text available."

    # If within reasonable token budget (~12k tokens / 50k chars), pass the entire raw text
    if len(raw_text) <= 50000:
        return raw_text

    # For massive documents (>50k chars), assemble a complete structural outline
    # ensuring every single section, heading, and chapter across all pages is included
    chunks = state.get("chunks", [])
    if chunks:
        outline_parts = [
            f"DOCUMENT OVERVIEW (Total Chunks: {len(chunks)}, Total Characters: {len(raw_text)}):",
            "\n--- SECTION & HEADING OUTLINE ACROSS ALL PAGES ---"
        ]
        seen_headings = set()
        for c in chunks:
            hierarchy = " > ".join(c.heading_hierarchy) if c.heading_hierarchy else c.section_title or ""
            if hierarchy and hierarchy not in seen_headings:
                seen_headings.add(hierarchy)
                outline_parts.append(
                    f"\n[Page {c.page_number}] Section: {hierarchy}\nExcerpt: {c.content[:350]}..."
                )
            elif c.page_number % 2 == 0 and len(seen_headings) < 30:
                outline_parts.append(f"\n[Page {c.page_number}] Excerpt: {c.content[:200]}...")

        outline_parts.append("\n--- DOCUMENT INTRODUCTION ---")
        outline_parts.append(raw_text[:6000])
        outline_parts.append("\n--- DOCUMENT CONCLUSION / ADVANCED TOPICS ---")
        outline_parts.append(raw_text[-6000:])
        return "\n".join(outline_parts)

    # Fallback: head and tail
    return raw_text[:25000] + "\n\n...[middle sections omitted]...\n\n" + raw_text[-25000:]


async def topic_planner_node(state: PodcastState) -> dict:
    """
    LangGraph node: Analyze document and create topic plan.
    """
    logger.info("topic_planner_start", doc_id=state.get("doc_id"))

    preferences = state.get("user_preferences", {})
    difficulty = preferences.get("difficulty", "intermediate")
    duration = preferences.get("duration", settings.default_podcast_length)
    focus = preferences.get("focus", "general")

    # Use the full document representation
    content = _prepare_document_content(state)

    prompt = PLANNER_PROMPT.format(
        content=content,
        difficulty=difficulty,
        duration=duration,
        focus=focus,
    )

    try:
        response = await llm_provider.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=3000,
        )

        plan_data = extract_json(response.text)
        # Ensure user's requested duration is strictly enforced
        plan_data["estimated_duration_minutes"] = duration

        # Fallback for coverage mode if omitted
        if "coverage_mode" not in plan_data:
            plan_data["coverage_mode"] = "summary" if duration <= 25 else "comprehensive"

        # If LLM didn't return segments, build reasonable default segments from key concepts
        if not plan_data.get("segments") and plan_data.get("key_concepts"):
            concepts = plan_data["key_concepts"]
            step = 3 if duration <= 25 else 4
            segments = []
            for i in range(0, len(concepts), step):
                cluster = concepts[i : i + step]
                act_num = (i // step) + 1
                segments.append({
                    "title": f"Act {act_num}: {', '.join(cluster[:2])}",
                    "concepts": cluster,
                    "description": f"Deep dive and discussion on {', '.join(cluster)}."
                })
            plan_data["segments"] = segments

        topic_plan = TopicPlan(**plan_data)

        logger.info(
            "topic_plan_created",
            topic=topic_plan.topic,
            concepts=len(topic_plan.key_concepts),
            goals=len(topic_plan.learning_goals),
            segments=len(topic_plan.segments),
            duration=topic_plan.estimated_duration_minutes,
            coverage_mode=topic_plan.coverage_mode,
        )

        return {
            "topic_plan": topic_plan,
            "current_stage": "topic_planned",
            "status": "planning_complete",
        }

    except Exception as e:
        logger.error("topic_planner_failed", error=str(e))

        fallback_plan = TopicPlan(
            topic="Technical Concepts Deep Dive",
            difficulty=difficulty,
            learning_goals=["Master core architecture and mechanisms"],
            key_concepts=["System Fundamentals", "Core Mechanics", "Advanced Trade-offs"],
            estimated_duration_minutes=duration,
            coverage_mode="summary" if duration <= 25 else "comprehensive",
        )

        return {
            "topic_plan": fallback_plan,
            "current_stage": "topic_planned_fallback",
            "status": "planning_complete_with_fallback",
        }