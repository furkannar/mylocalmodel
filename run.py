"""Interactive command-line chat. Model, tools, and selection logic live in engine.py."""

from config import ChatConfig
from engine import ChatEngine


def print_candidate_scores(metadata: dict, best_of_n: int) -> None:
    runs = metadata.get("candidate_runs") or []
    if not runs:
        return
    run = runs[-1]
    print(f"\n--- Best-of-{best_of_n} Candidate Scores ({run['stage']}) ---")
    for candidate in run["candidates"]:
        marker = "*" if candidate["selected"] else " "
        score = candidate["score"]
        prob = candidate["geom_mean_prob"]
        score_text = f"{score:.4f}" if score is not None else "n/a"
        prob_text = f"{prob * 100:.1f}%" if prob is not None else "n/a"
        print(
            f" {marker} Candidate {candidate['index']}: Log-Likelihood = {score_text} | "
            f"Geom Mean Prob = {prob_text} ({candidate['token_count']} tokens)"
        )
    print("-" * 55)


def main():
    config = ChatConfig()
    engine = ChatEngine(config)
    history = []

    print("\n=== CLI Chat with Best-of-N Calibrated Qwen Instruct Model ===")
    print(
        f"Sampling Best-of-{config.best_of_n} candidates per turn "
        f"(T_gen={config.temperature}, T_calib={config.calibration_temperature})\n"
    )
    print("Type 'quit' or 'exit' to end the conversation\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if user_input.lower() in ("quit", "exit"):
            print("\nGoodbye!")
            break
        if not user_input:
            continue

        response, metadata = engine.respond(user_input, history)

        if metadata["rag_used"]:
            print(
                f"📚 Retrieved {len(metadata['rag_matches'])} reference item(s) from Wikipedia QA "
                f"(distance <= {config.rag.max_cosine_distance})"
            )
        for call in metadata["tool_calls"]:
            outcome = call.get("result") if call["success"] else f"error: {call.get('error')}"
            print(f"🧮 Calculator: {call['expression']} -> {outcome}")
        if config.show_candidate_scores:
            print_candidate_scores(metadata, config.best_of_n)

        print(f"AI: {response}\n")
        history.append((user_input, response))  # only the accepted answer is kept


if __name__ == "__main__":
    main()
