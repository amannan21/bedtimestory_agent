import os, json
from typing import TypedDict, Optional, List
from openai import OpenAI
from langgraph.graph import StateGraph, END
from prompts import JUDGE_PROMPT, Story_PROMPT, REVISER_SYS, CLASSIFIER_SYS


"""
**Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:**   

If I had more time, I would have the similar system but I would have the explored diverse candidates at the story generation phase (for instance, pick 6 or so candidates with different temperatures and slightly different seeding instructions). Then I would have created one of these: 1. a new pairwise LLM Judge in charge of picking the best one of those candidates 2. Panel of LLm evaluators (as outlined here https://arxiv.org/pdf/2404.18796) 3.a compiler agent to take the best parts of each story and compile them into one (would have tested to determine which was better fit). The issue with relying on a solely single output judge is that scores drift, calibration can be fragile, and it can be gamed more easily by verbosity or keyword stuffing than a pairwise judge.

Alternatively, I would have also explored decoupling the revise agent into a few different agents (example: rhyme polisher, parental-controls, etc.) instead of just the single reviser model.

Over a longer period of 2 hours I would’ve also tried implementing automatic red-teaming using deepeval library for robustness.


"""

# classify → story → judge → (loop) → finalize.
#example_requests = "A story about a girl named Alice and her best friend Bob, who happens to be a cat."


CATEGORY_STRATEGY = {
    "bedtime_calm":      "Soothing cadence; tiny, fixable hiccup; quiet sensory images; end on safety/comfort.",
    "adventure":         "Curious, upbeat tone; 2–3 gentle obstacles; celebratory but safe discovery; happy homecoming.",
    "animal_fable":      "Distinct animal quirks; kind consequences; one clear moral emerging from actions.",
    "friendship":        "Name feelings; small misunderstanding; talk it out; apology/repair; warmer bond.",
    "problem_solving":   "On-page kid reasoning: notice → try → adjust → succeed; simple connectors (because/so).",
    "silly_fun":         "Bouncy rhythm; playful absurdity or light rhyme; surprises without meanness or gross-out.",
    "science_magic":     "Pick ONE: (a) simple real fact or (b) one soft magic rule; keep it concrete and fun.",
    "mystery_cozy":      "Gentle puzzle/clues; zero menace; friendly reveal; cocoa-warm ending.",
    "community_helping": "Kindness-in-action; small neighborhood need; teamwork; gratitude at the end.",
    "nature_wonder":     "Sensory awe (colors, textures, sounds); seasons/animals/sky; calm reflective close.",
    "sports_teamwork":   "Practice and teamwork over winning; fair play; bounce back from a fixable setback.",
    "arts_music":        "Creative expression; trying, messing up, trying again; pride in sharing; supportive crowd.",
    "custom_names":      "Center provided names/interests; gentle stakes; affirm strengths; proud, cozy finish."
}

_client = None  # lazy-initialized OpenAI client

def get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it in your shell: export OPENAI_API_KEY=\"sk-...\""
        )
    _client = OpenAI(api_key=api_key)
    return _client
    # redo this to the new way of doing things 



def call_openai(model: str, system: str, user_message: str, temperature: float = 0.2) -> str:
    """
   
    Returns plain text. 
    """
    resp = get_client().responses.create(
        model=model,
        temperature=temperature,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
    )
    return resp.output_text

class StoryState(TypedDict):
    request: str # the story itself
    category: Optional[str]
    story: Optional[str]
    judge_report: Optional[dict]
    rounds: int
    pass_mark: float
    max_rounds: int
    models: dict # for revise, judge, story, classifier


def classify_node(state:StoryState) -> StoryState:
    request = state["request"]
    
    output = call_openai(state["models"]["classify"], CLASSIFIER_SYS, state["request"])

    try:
        obj = json.loads(output)

    except Exception as e:
        state["category"] = "custom_names"
    return state


def story_node(state:StoryState) -> StoryState:
    cat = state["category"] or "custom_names"
    strat = CATEGORY_STRATEGY.get(cat, "General bedtime story.")
    user = f"Request: {state['request']}\nStrategy: {strat}"

    story = call_openai(state["models"]["story"], Story_PROMPT, user, temperature=0.4)
    state["story"] = story
    return state



def judge_node(state:StoryState) -> StoryState:

    # user message would be the story and the system message would be the prompt -> JUDGE_PROMPT
    user = f"Story: {state['story']}"
    payload = f"Story: {state['story']}\n\nEvaluate per rubric and return the JSON"
    out = call_openai(state["models"]["judge"], JUDGE_PROMPT, payload)

    try:
        state["judge_report"] = json.loads(out)

    except Exception as e:
        state["judge_report"] = {
            "weighted_score": 0.0,
            "fail_reasons": ["bad_json"],
            "fixes": ["Return valid JSON next time."],
            "praise": []
        }

    return state

def revise_node(state:StoryState) -> StoryState:
    report = state["judge_report"] or {}
    user = json.dumps(
        {
            "story":state["story"],
            "fixes": report.get("fixes", []),
            "praise": report.get("praise", []),

        }
    )
    state["story"] = call_openai(state["models"]["revise"], REVISER_SYS, user, temperature=0.23)

    state["rounds"] += 1
    return state


def should_continue(state:StoryState) -> str:
    rep = state["judge_report"] or {}
    s   = rep.get("scores", {})          # per-dimension
    fails = rep.get("critical_failures", [])
    weighted = float(rep.get("weighted_score", 0.0))

    #  prevent infinite loops
    if state["rounds"] >= state["max_rounds"]:
        return "stop"

    # Hard safety gate
    if fails:
        return "stop"  # or route to a "sanitize" node if you add one

    # Dimension floors 
    if s.get("age_appropriateness", 0)   < 4: return "revise"
    if s.get("inclusivity_safety", 0)    < 4: return "revise"

    # Overall bar
    if weighted < state["pass_mark"]:
        return "revise"

    return "stop"


def build_graph():
    graph = StateGraph(StoryState)
    graph.add_node("classify", classify_node) # calling this function
    graph.add_node("generate", story_node)
    graph.add_node("judge", judge_node)
    graph.add_node("revise", revise_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "generate")
    graph.add_edge("generate", "judge")
    # if returns revise -> go back to revise, otherwise end the cycle here 
    graph.add_conditional_edges("judge", should_continue, {"revise": "revise", "stop": END})
    graph.add_edge("revise", "judge")
    return graph.compile()



def main():
    user_input = input("What kind of story do you want to hear? ")
    response = call_openai(user_message=user_input,system=Story_PROMPT,model="gpt-3.5-turbo") 
    # how can I create the feedback loop here between judge and story model
    print("Story model response: ")
    print(response)
    models = {
        # Keep judge >= story. Swap to newer models when available.
        "story": "gpt-3.5-turbo",
        "judge": "gpt-3.5-turbo",
        "revise": "gpt-3.5-turbo",
        "classify": "gpt-3.5-turbo",
    }

    app = build_graph()
    initial_state: StoryState = {
        "request": user_input,
        "category": None,
        "story": None,
        "judge_report": None,
        "rounds": 0,
        "pass_mark": 4.5,
        "max_rounds": 7,
        "models": models,
    }
    final_state = app.invoke(initial_state)

    # Interactive feedback loop: keep asking for tweaks until user exits
    current_story = final_state["story"]
    while True:
        fb = input("\nTweaks? Enter to finish, or type change (q to quit): ").strip()
        if not fb or fb.lower() in {"q", "quit", "exit", "n", "no"}:
            break

        tmp_report = {"fixes": [fb], "praise": []}
        revised = call_openai(models["revise"], REVISER_SYS, json.dumps({
            "story": current_story,
            "fixes": tmp_report["fixes"],
            "keep": []
        }), temperature=0.2)
        judge_out = call_openai(models["judge"], JUDGE_PROMPT,
                                f"Story:\n<<<\n{revised}\n>>>\nEvaluate per rubric and return JSON.",
                                temperature=0.0)
        try:
            judge_json = json.loads(judge_out)
        except Exception:
            judge_json = {"weighted_score": 0.0, "fail_reasons":["bad_json"], "fixes":[], "praise":[]}

        print("\n=== REVISED STORY ===\n")
        print(revised)
        # print("\n=== REVISED JUDGE REPORT ===\n")
        # print(json.dumps(judge_json, indent=2))
        current_story = revised
    


if __name__ == "__main__":
    main()


