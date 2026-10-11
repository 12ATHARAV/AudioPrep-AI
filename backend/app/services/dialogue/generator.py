"""
RAG-grounded dialogue generator.

Generates dynamic, natural podcast dialogue grounded in document evidence:
- Short episodes (<= 25m): Executive high-yield summary covering the most important concepts first.
- Long episodes (> 25m, up to 120m / 2 hours): Multi-segment progressive generation unpacking all document sections in deep detail.
"""

import json
from app.agents.state import PodcastState, DialogueTurn, TopicPlan
from app.core.llm_provider import llm_provider
from app.core.logging import get_logger
from app.core.json_utils import extract_json

logger = get_logger("dialogue_generator")

DIALOGUE_SYSTEM_PROMPT = """You are a master technical podcast producer and scriptwriter for an award-winning tech show (like 'Acquired' or 'Radiolab').
You write deeply engaging, high-energy, and natural conversations between two charismatic colleagues: HOST and EXPERT.

AUDIO PODCAST CHEMISTRY RULES:
1. NOT A DRY Q&A LECTURE: Do NOT make this a repetitive "Host asks question -> Expert defines term" pattern. This is a real, dynamic conversation between passionate colleagues.
2. HOST's PERSONALITY: Curiously skeptical, enthusiastic, relatable male co-host. The Host offers intuitive guesses, shares common misconceptions ("Wait, so in my head I always pictured it like..."), challenges claims, and has authentic "aha!" moments.
3. EXPERT's PERSONALITY: Highly articulate, confident, and warm female technical lead/architect. Master communicator, authoritative yet accessible, pragmatic. Uses vivid real-world analogies (e.g. shipping docks, apartment buildings, restaurant kitchens, airport gates) instead of dry textbook definitions.
4. TURN PACING: Keep each turn between 20 and 65 words. Break up complex explanations into natural back-and-forth volleys.
5. VOCAL MARKERS: Naturally weave in conversational tags: <short pause>, <laugh>, <chuckle>, <sigh>, and verbal affirmations ("100%!", "Exactly!", "Wait, hold on a second...").

OUTPUT FORMAT:
Return ONLY a valid JSON array of dialogue turns:
[
  {{"speaker": "HOST", "text": "...", "style": "curious"}},
  {{"speaker": "EXPERT", "text": "...", "style": "explaining"}},
  ...
]"""

SUMMARY_PROMPT = """Write a complete {duration}-minute executive summary podcast episode ({target_words} target words, at least {min_turns} turns).

TOPIC: {topic}
DIFFICULTY: {difficulty}
TARGET LENGTH: {duration} minutes (~{target_words} words, {min_turns}-{max_turns} turns)

KEY CONCEPTS ACROSS THE DOCUMENT:
{concepts}

LEARNING GOALS:
{learning_goals}

INTERVIEW TRAPS & TRICKY QUESTIONS:
{interview_focus}

SOURCE EVIDENCE (ground all technical specifics here):
{evidence}
{critic_feedback}

EPISODE ARC TO EXECUTE (HIGH-YIELD SUMMARY):
- ACT 1: COLD OPEN & HOOK (4-6 turns): A relatable production nightmare, real-world bug, or counterintuitive paradox that immediately hooks the listener.
- ACT 2: CORE FOUNDATIONAL MECHANICS (10-14 turns): Unpack the MOST IMPORTANT foundational topics first with vivid analogies and intuitive explanations.
- ACT 3: HIGH-YIELD SYNTHESIS OF ADVANCED CONCEPTS (10-14 turns): Synthesize the remaining advanced concepts across the document into practical architectural trade-offs so the entire document's arc is covered.
- ACT 4: THE INTERVIEW GAUNTLET (4-6 turns): Expert poses a classic tricky interview trap question; Host gives the intuitive common answer; Expert breaks down why that answer fails.
- ACT 5: TAKEAWAYS & WITTY OUTRO (4-6 turns): Lightning recap of key takeaways, signature podcast sign-off.

CRITICAL: Generate the full episode covering the document without stopping early (at least {min_turns} turns). Return ONLY valid JSON array of dialogue turns."""

SEGMENT_PROMPT = """You are writing Segment {segment_num} of {total_segments} for a comprehensive {duration}-minute masterclass podcast episode.

OVERALL TOPIC: {topic}
DIFFICULTY: {difficulty}
CURRENT SEGMENT: {segment_title}
SEGMENT DESCRIPTION: {segment_description}

KEY CONCEPTS FOR THIS SEGMENT:
{segment_concepts}

SOURCE EVIDENCE FOR THIS SEGMENT:
{evidence}
{critic_feedback}

{context_previous}

SEGMENT ROLE & INSTRUCTIONS:
{segment_role_instructions}
- Target length: {target_turns} turns (~{target_words} words).
- Thoroughly explore the mechanics, edge cases, real-world analogies, and engineering trade-offs.
- Do NOT rush. Keep the back-and-forth volley dynamic and engaging.
- Return ONLY a valid JSON array of dialogue turns for this segment."""


async def dialogue_generator_node(state: PodcastState) -> dict:
    """
    LangGraph node: Generate podcast dialogue.
    Adapts strategy based on requested duration:
    - Short (<= 25m): Single-pass executive summary prioritizing foundational concepts.
    - Long (> 25m, e.g. 60m or 120m): Progressive multi-segment generation covering all sections in depth.
    """
    logger.info("dialogue_generation_start")

    plan = state["topic_plan"]
    if isinstance(plan, dict):
        plan = TopicPlan(**plan)

    preferences = state.get("user_preferences", {})
    duration = preferences.get("duration") or getattr(plan, "estimated_duration_minutes", 120) or 120

    contexts = state.get("retrieval_contexts", {})

    # Extract critic feedback if revising
    critic = state.get("critic_result")
    issues = []
    if critic:
        if hasattr(critic, "issues"):
            issues = critic.issues
        elif isinstance(critic, dict):
            issues = critic.get("issues", [])

    critic_feedback = ""
    if issues:
        critic_feedback = "\n\nFEEDBACK FROM PREVIOUS CRITIC PASS (MUST RESOLVE):\n" + "\n".join(
            f"- {issue}" for issue in issues
        )

    try:
        if duration <= 25:
            # High-Yield Summary Mode
            dialogue = await _generate_summary_dialogue(plan, duration, contexts, critic_feedback)
        else:
            # Comprehensive Masterclass Mode (Multi-segment)
            dialogue = await _generate_segmented_dialogue(plan, duration, contexts, critic_feedback)

        # Build formatted script for TTS
        script = _format_script(dialogue)

        total_words = sum(len(t.text.split()) for t in dialogue)
        logger.info(
            "dialogue_generated",
            turns=len(dialogue),
            words=total_words,
            duration=duration,
        )

        return {
            "dialogue": dialogue,
            "dialogue_script": script,
            "current_stage": "dialogue_generated",
            "status": "dialogue_complete",
        }

    except Exception as e:
        logger.error("dialogue_generation_failed", error=str(e))
        return {
            "dialogue": [],
            "dialogue_script": "",
            "current_stage": "dialogue_failed",
            "status": "error",
            "error": f"Dialogue generation failed: {str(e)}",
        }


async def _generate_summary_dialogue(
    plan: TopicPlan,
    duration: int,
    contexts: dict,
    critic_feedback: str,
) -> list[DialogueTurn]:
    """Generate a single-pass high-yield summary episode."""
    target_words = duration * 140
    min_turns = max(24, int(duration * 3.2))
    max_turns = int(duration * 4.5)

    evidence_parts = []
    for concept, ctx_list in contexts.items():
        evidence_parts.append(f"\n### {concept}")
        for i, ctx in enumerate(ctx_list[:2]):
            content = ctx.get("content", "") if isinstance(ctx, dict) else str(ctx)
            evidence_parts.append(f"[Evidence {i+1}]: {content[:400]}")

    evidence_text = "\n".join(evidence_parts) if evidence_parts else "Use technical knowledge."

    prompt = SUMMARY_PROMPT.format(
        topic=plan.topic,
        difficulty=plan.difficulty,
        duration=duration,
        target_words=target_words,
        min_turns=min_turns,
        max_turns=max_turns,
        learning_goals="\n".join(f"- {g}" for g in plan.learning_goals),
        concepts="\n".join(f"- {c}" for c in plan.key_concepts),
        interview_focus="\n".join(f"- {q}" for q in plan.interview_focus),
        evidence=evidence_text,
        critic_feedback=critic_feedback,
    )

    response = await llm_provider.generate(
        prompt=prompt,
        system_prompt=DIALOGUE_SYSTEM_PROMPT,
        temperature=0.75,
        max_tokens=6000,
    )

    turns_data = extract_json(response.text)
    return [DialogueTurn(**turn) for turn in turns_data]


async def _generate_segmented_dialogue(
    plan: TopicPlan,
    duration: int,
    contexts: dict,
    critic_feedback: str,
) -> list[DialogueTurn]:
    """Generate long comprehensive podcast episode segment-by-segment."""
    # Determine segments to execute
    segments_to_run = []
    if plan.segments and len(plan.segments) >= 3:
        for s in plan.segments:
            segments_to_run.append({
                "title": s.title or "Deep Dive Segment",
                "concepts": s.concepts,
                "description": s.description or f"Exploring {', '.join(s.concepts)}",
            })
    else:
        # Partition key concepts into 5-6 progressive segments
        concepts = plan.key_concepts
        step = max(3, len(concepts) // 5)
        for i in range(0, len(concepts), step):
            cluster = concepts[i : i + step]
            act_num = (i // step) + 1
            segments_to_run.append({
                "title": f"Act {act_num}: {', '.join(cluster[:2])}",
                "concepts": cluster,
                "description": f"In-depth architectural analysis of {', '.join(cluster)}.",
            })

    total_segments = len(segments_to_run)
    # Target turns per segment
    target_turns = max(20, min(35, (duration * 3) // total_segments))
    target_words = target_turns * 38

    master_dialogue: list[DialogueTurn] = []

    for idx, seg in enumerate(segments_to_run):
        seg_num = idx + 1
        logger.info(
            "generating_segment",
            segment=seg_num,
            total=total_segments,
            title=seg["title"],
        )

        # Build context from previous turns for smooth conversational bridge
        context_previous = ""
        if master_dialogue:
            last_two = master_dialogue[-2:]
            context_previous = "PREVIOUS CONVERSATION (Pick up the dialogue seamlessly from here):\n"
            for t in last_two:
                context_previous += f'[{t.speaker}]: "{t.text}"\n'
            context_previous += f"\nNow transition naturally into the next section: '{seg['title']}'."

        # Filter evidence for this segment's concepts
        evidence_parts = []
        for c in seg["concepts"]:
            if c in contexts:
                evidence_parts.append(f"\n### {c}")
                for i, ctx in enumerate(contexts[c][:2]):
                    content = ctx.get("content", "") if isinstance(ctx, dict) else str(ctx)
                    evidence_parts.append(f"- {content[:400]}")

        # If no specific evidence, include general evidence
        if not evidence_parts:
            for c, ctx_list in list(contexts.items())[:3]:
                for ctx in ctx_list[:1]:
                    content = ctx.get("content", "") if isinstance(ctx, dict) else str(ctx)
                    evidence_parts.append(f"- {content[:300]}")

        evidence_text = "\n".join(evidence_parts) if evidence_parts else "Ground statements in technical accuracy."

        # Segment role instructions
        if seg_num == 1:
            role_inst = "OPENING ACT: Start with an attention-grabbing cold open/hook highlighting a production crisis or counterintuitive paradox before opening up the core system architecture."
        elif seg_num == total_segments:
            role_inst = f"FINALE & OUTRO: Deliver the final deep dive, conduct an intense interview gauntlet on tricky edge cases ({', '.join(plan.interview_focus[:2])}), provide a rapid recap of key takeaways, and give a signature podcast sign-off."
        else:
            role_inst = "CORE DEEP DIVE: Seamlessly continue the volley. Push deep into internal mechanisms, real-world analogies, failure modes, and architectural trade-offs."

        prompt = SEGMENT_PROMPT.format(
            segment_num=seg_num,
            total_segments=total_segments,
            duration=duration,
            topic=plan.topic,
            difficulty=plan.difficulty,
            segment_title=seg["title"],
            segment_description=seg["description"],
            segment_concepts=", ".join(seg["concepts"]),
            evidence=evidence_text,
            critic_feedback=critic_feedback if seg_num == 1 else "",
            context_previous=context_previous,
            segment_role_instructions=role_inst,
            target_turns=target_turns,
            target_words=target_words,
        )

        response = await llm_provider.generate(
            prompt=prompt,
            system_prompt=DIALOGUE_SYSTEM_PROMPT,
            temperature=0.75,
            max_tokens=4000,
        )

        turns_data = extract_json(response.text)
        seg_turns = [DialogueTurn(**t) for t in turns_data]
        master_dialogue.extend(seg_turns)

        logger.info(
            "segment_completed",
            segment=seg_num,
            turns=len(seg_turns),
            total_accumulated_turns=len(master_dialogue),
        )

    return master_dialogue


def _format_script(dialogue: list[DialogueTurn]) -> str:
    """Format dialogue turns into a readable script."""
    lines = []
    for turn in dialogue:
        lines.append(f"[{turn.speaker}]: {turn.text}")
    return "\n\n".join(lines)