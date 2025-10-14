import argparse
import pandas as pd
from tqdm import tqdm
import openai
import os


def adverb_diagnose(adverb, sentence):
    return f"""You are a linguistics expert trained to assign supersense labels to adverbs in context.
Given an adverb and the sentence in which it occurs, your task is to classify the adverb into one of the categories below. Use the paraphrase-based diagnostics and guiding questions provided for each type.

Return only one label that best fits the adverb's usage in the given sentence.

⸻

MANNER (adverb.manner)

Describes how the event or action is performed.

Paraphrase Test: Can it be rephrased as "in a [X] manner"?

Example:
Sentence: "He danced stupidly."
Paraphrase: "He danced in a stupid manner."

⸻

SUBJECT-ORIENTED (adverb.subject_oriented)

Attributes a property or attitude to the subject of the sentence in relation to the event.

Paraphrase Test: Can it be rephrased as "It was [X] of [SUBJECT] to [VERB]"?

Example:
Sentence: "Stupidly, he walked into traffic."
Paraphrase: "It was stupid of him to walk into traffic."

⸻

SPEAKER-ORIENTED (adverb.speaker_oriented)

Expresses the speaker's stance, evaluation, or attitude toward the proposition or the act of speaking.

Paraphrase Test: Can it be paraphrased as "I [believe / say / judge] that…" or "It is [unfortunate / fortunate / evident] that…"?

Examples:
	•	"Presumably, he missed the deadline." → "I presume that he missed the deadline."
	•	"Unfortunately, the project failed." → "It is unfortunate that the project failed."
	•	"Frankly, I disagree." → "I say this frankly."

⸻

FREQUENCY (adverb.frequency)

Describes how often the event occurs.

Question Test: "How often?"

Example:
Sentence: "She frequently visits her grandmother."
Answer: "Frequently." → Frequency adverb

⸻

TEMPORAL (adverb.temporal)

Specifies when the event happens or its duration.

Question Test: "When?" or "For how long?"

Examples:
	•	Sentence: "She arrived yesterday." → When? → Yesterday
	•	Sentence: "He stayed briefly." → For how long? → Briefly

⸻

SPATIAL (LOCATIVE) (adverb.spatial)

Specifies where the event takes place.

Question Test: "Where?"

Example:
Sentence: "He stood outside."
Answer: "Outside." → Spatial adverb

⸻

DEGREE (adverb.degree)

Describes the intensity, extent, or scalar position of another element (adjective, adverb, verb).

Question Test: "To what extent?" or "How much?"

Example:
Sentence: "She is very happy."
Question: "To what extent is she happy?"
Answer: "Very." → Degree adverb

⸻

DOMAIN (adverb.domain)

Limits the proposition to a particular semantic or disciplinary domain.

Paraphrase Test: "In a [X] sense" or "From a [X] perspective"

Example:
Sentence: "Politically, the policy is controversial."
Paraphrase: "In a political sense, the policy is controversial."

⸻

Output Format

Return the label only:
	•	adverb.manner
	•	adverb.subject_oriented
	•	adverb.speaker_oriented
	•	adverb.frequency
	•	adverb.temporal
	•	adverb.spatial
	•	adverb.degree
	•	adverb.domain

⸻

Examples

Adverb: wisely
Usage: "Wisely, he did not sign the contract."
Reasoning: The sentence can be paraphrased as "It was wise of him to not sign the contract," which assigns a property (wisdom) to the subject with respect to the action. This indicates a subject-oriented adverb.
Answer: adverb.subject_oriented

⸻

Adverb: frankly
Usage: "Frankly, I don't agree with that decision."
Reasoning: The sentence can be paraphrased as "I say this frankly," where the adverb qualifies the act of speaking, not the proposition itself. This indicates a speaker-oriented adverb.
Answer: adverb.speaker_oriented

⸻

Adverb: daily
Usage: "He exercises daily."
Reasoning: The adverb answers the question "How often?" with "daily," describing the frequency of the event.
Answer: adverb.frequency


Adverb: {adverb}
Usage: {sentence}"""


def process_prompt(client, prompt, model):
    """Process a single prompt with the OpenAI API"""
    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            reasoning={"effort": "medium"},
            text={"verbosity": "low"},
        )
        print(response.output_text)
        return response.output_text
    except Exception as e:
        print(f"Error processing prompt: {e}")
        return ""


def main(raw_args=None):
    parser = argparse.ArgumentParser(
        prog="simple.py",
        description="Adverb classification using OpenAI API",
        epilog="""Generate adverb classifications using OpenAI API.

    Usage:
        simple.py --api-key <openai_api_key> [options]

    Arguments:
        --api-key = OpenAI API key (or set OPENAI_API_KEY environment variable)
        --model = OpenAI model to use (default: gpt-3.5-turbo)
        --max-tokens = Maximum tokens to generate
    """,
    )

    parser.add_argument("--model", default="gpt-5-mini", help="OpenAI model to use")

    args = parser.parse_args(raw_args)

    # Set up OpenAI API key
    openai.api_key = os.environ["OPENAI_API_KEY"]

    # Initialize OpenAI client
    client = openai.OpenAI(api_key=openai.api_key)

    # Load data
    data = pd.read_csv("adverbs.csv")

    # Process prompts
    results = []
    index = -1

    try:
        for index, row in tqdm(data.iterrows(), total=len(data)):
            prompt = adverb_diagnose(row["adverb"], row["sentence"])
            result = process_prompt(client, prompt, args.model)
            results.append(result)
    except KeyboardInterrupt:
        print("stopped at index:", index)
    finally:
        # Save results
        n_rows = len(results)
        data["type"] = pd.NA
        data.iloc[:n_rows, data.columns.get_loc("type")] = results
        output_file = "adverb_os_llm.csv"
        data.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
