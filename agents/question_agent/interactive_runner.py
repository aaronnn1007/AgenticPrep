import sys
import time
import threading
import queue
from typing import Callable

# Import compiled_graph resiliently
try:
    from graph import compiled_graph
except Exception:
    from question_agent.graph import compiled_graph


def map_difficulty(difficulty: str) -> str:
    mapping = {
        "easy": "Fresher",
        "medium": "Mid",
        "hard": "Senior"
    }
    return mapping.get(difficulty, "Fresher")


def get_valid_input(prompt: str, validator: Callable[[str], bool], error_msg: str) -> str:
    while True:
        try:
            value = input(prompt).strip()
            if validator(value):
                return value
            else:
                print(error_msg)
        except KeyboardInterrupt:
            print("\nExiting...")
            raise
        except Exception as e:
            print(f"Error: {e}. Please try again.")


def _input_reader(q: queue.Queue, stop_event: threading.Event) -> None:
    """Read a single line from stdin and put it into the queue.

    This blocks on input() but runs in a daemon thread. If stop_event is set
    before input is received, the result will be ignored by the main thread.
    """
    try:
        line = sys.stdin.readline()
        if line is None:
            return
        q.put(line)
    except Exception:
        return


def ask_question_with_timer(question_text: str, time_limit: int) -> str:
    """Display question, show remaining time, and wait for N/S or timeout.

    Returns:
      'N' if user chose Next, 'S' if user chose Skip, 'T' if timed out.
    """
    q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    reader = threading.Thread(target=_input_reader, args=(q, stop_event), daemon=True)

    print("Question:")
    print(question_text)
    print()

    prompt_text = "Press N (Next) or S (Skip) and hit Enter: "

    reader.start()
    start = time.monotonic()
    end = start + float(time_limit)
    last_remaining = -1

    try:
        while True:
            now = time.monotonic()
            remaining = int(max(0, end - now))

            # Update remaining display only when it changes
            if remaining != last_remaining:
                print(f"Remaining time: {remaining}s", end="\r", flush=True)
                last_remaining = remaining

            # Check for user input
            try:
                line = q.get_nowait()
            except queue.Empty:
                line = None

            if line is not None:
                # normalize input
                choice = line.strip().upper()
                if len(choice) > 0:
                    c = choice[0]
                    if c == 'N':
                        print("\n")
                        stop_event.set()
                        return 'N'
                    elif c == 'S':
                        print("\n")
                        stop_event.set()
                        return 'S'
                    else:
                        # invalid input, inform user and continue
                        print("\nInvalid choice. Please enter N or S.")
                        # start a fresh reader to get next input
                        if not reader.is_alive():
                            reader = threading.Thread(target=_input_reader, args=(q, stop_event), daemon=True)
                            reader.start()

            if now >= end:
                print("\nTime up! Moving to next question.")
                stop_event.set()
                return 'T'

            # Sleep a short time to avoid busy-waiting
            time.sleep(0.2)
    except KeyboardInterrupt:
        stop_event.set()
        print("\nExiting...")
        raise


def main():
    print("Welcome to InterviewPA 4 - Interactive Question Generator")
    print("=" * 50)

    # Get role
    role = get_valid_input(
        "Enter role/topic: ",
        lambda x: len(x) > 0,
        "Role cannot be empty. Please enter a valid role."
    )

    # Get number of questions
    num_questions = int(get_valid_input(
        "Number of questions: ",
        lambda x: x.isdigit() and int(x) > 0,
        "Please enter a positive integer for number of questions."
    ))

    # Get difficulty
    difficulty = get_valid_input(
        "Difficulty (easy/medium/hard): ",
        lambda x: x.lower() in ['easy', 'medium', 'hard'],
        "Please enter 'easy', 'medium', or 'hard'."
    ).lower()

    # Get time limit (for guidance)
    time_limit = int(get_valid_input(
        "Time limit (seconds): ",
        lambda x: x.isdigit() and int(x) > 0,
        "Please enter a positive integer for time limit."
    ))

    # Get preferred question type
    preferred_question_type = get_valid_input(
        "Preferred question type (technical/behavioral/situational/mixed): ",
        lambda x: x.lower() in ['technical', 'behavioral', 'situational', 'mixed'],
        "Please enter 'technical', 'behavioral', 'situational', or 'mixed'."
    ).lower()

    print("\nStarting interview questions...\n")

    previous_answers = []

    for i in range(1, num_questions + 1):
        state = {
            "role": role,
            "experience_level": map_difficulty(difficulty),
            "question_index": i,
            "previous_answers": previous_answers,
            "prerequisites": {
                "number_of_questions": num_questions,
                "difficulty_level": difficulty,
                "time_limit_seconds": time_limit,
                "preferred_question_type": preferred_question_type
            }
        }

        try:
            result = compiled_graph.invoke(state)

            # Show question and start timer
            qtext = result["question_text"]
            qtime = int(result.get("time_limit_seconds", time_limit))

            choice = ask_question_with_timer(qtext, qtime)

            if choice == 'N':
                previous_answers.append("ANSWER_PENDING")
            # if 'S' or 'T' do not append anything (skipped)

            print()  # Blank line between questions

        except Exception as e:
            print(f"Error generating question {i}: {str(e)}")
            print("Continuing to next question...\n")

    print("Interview questions completed!")


if __name__ == "__main__":
    main()