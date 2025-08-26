import json
import streamlit as st
import main as app
import prompts


st.set_page_config(page_title="Kids Story Agent", page_icon="📖", layout="centered")

if "story" not in st.session_state:
    st.session_state.story = None
if "judge_report" not in st.session_state:
    st.session_state.judge_report = None
if "request_text" not in st.session_state:
    st.session_state.request_text = ""
if "tweak_key" not in st.session_state:
    st.session_state.tweak_key = 0


st.title("📖 Kids Story Agent")
st.caption("Generate a bedtime-safe story (ages 5–10), then revise with a built-in judge.")


with st.form("request_form"):
    req = st.text_area("Story request", value=st.session_state.request_text, height=120, placeholder="A story about ...")
    submitted = st.form_submit_button("Generate Story")

if submitted:
    st.session_state.request_text = req
    models = {
        "story": "gpt-3.5-turbo",
        "judge": "gpt-3.5-turbo",
        "revise": "gpt-3.5-turbo",
        "classify": "gpt-3.5-turbo",
    }
    graph = app.build_graph()
    state = {
        "request": req,
        "category": None,
        "story": None,
        "judge_report": None,
        "rounds": 0,
        "pass_mark": 4.5,
        "max_rounds": 7,
        "models": models,
    }
    with st.spinner("Generating and judging..."):
        final_state = graph.invoke(state)
    st.session_state.story = final_state.get("story")
    st.session_state.judge_report = final_state.get("judge_report", {})


if st.session_state.story:
    st.subheader("Story")
    st.markdown(st.session_state.story)

    jr = st.session_state.judge_report or {}
    ws = jr.get("weighted_score")
    if ws is not None:
        st.metric("Judge score", f"{ws}")

    with st.expander("Judge details"):
        st.code(json.dumps(jr, indent=2))

    st.divider()
    st.subheader("Revise")
    tweak = st.text_input(
        "Describe a change (e.g., 'shorter ending', 'more dialogue')",
        key=f"tweak_text_{st.session_state.tweak_key}",
    )
    col1, col2 = st.columns(2)
    apply_btn = col1.button("Apply Tweak", use_container_width=True, key="apply_tweak")
    reset_btn = col2.button("Reset", use_container_width=True, type="secondary")

    if reset_btn:
        st.session_state.story = None
        st.session_state.judge_report = None
        st.session_state.request_text = ""
        st.rerun()

    if apply_btn:
        if not (tweak or "").strip():
            st.warning("Please enter a tweak before applying.")
        else:
            models = {
                "story": "gpt-3.5-turbo",
                "judge": "gpt-3.5-turbo",
                "revise": "gpt-3.5-turbo",
                "classify": "gpt-3.5-turbo",
            }
            try:
                with st.spinner("Revising and re-judging..."):
                    revised = app.call_openai(
                        models["revise"],
                        prompts.REVISER_SYS,
                        json.dumps({
                            "story": st.session_state.story,
                            "fixes": [tweak],
                            "keep": []
                        }),
                        temperature=0.2,
                    )
                    judge_out = app.call_openai(
                        models["judge"],
                        prompts.JUDGE_PROMPT,
                        f"Story:\n<<<\n{revised}\n>>>\nEvaluate per rubric and return JSON.",
                        temperature=0.0,
                    )
                    try:
                        jr2 = json.loads(judge_out)
                    except Exception:
                        jr2 = {"weighted_score": 0.0, "fail_reasons":["bad_json"], "fixes":[], "praise":[]}

                st.session_state.story = revised
                st.session_state.judge_report = jr2
                st.session_state.tweak_key += 1  # rotate widget key to clear input safely
                st.rerun()
            except Exception as e:
                st.error(f"Failed to apply tweak: {e}")


st.sidebar.caption("Run: streamlit run chatUI.py")


