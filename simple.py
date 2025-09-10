import argparse
import pandas as pd
from pprint import pprint
from tqdm import tqdm
import guidance
from guidance import models, gen, select


@guidance(stateless=True)
def adverb_diagnose(lm, adverb, sentence):
    return (
        lm
        + f"""
You are a linguistics expert trained to assign supersense labels to adverbs in context.
Given an adverb and the sentence in which it occurs, your task is to classify the adverb into one of the categories below. Use the paraphrase-based diagnostics and guiding questions provided for each type.

Return only one label that best fits the adverb’s usage in the given sentence.

⸻

MANNER (adverb.manner)

Describes how the event or action is performed.

Paraphrase Test: Can it be rephrased as “in a [X] manner”?

Example:
Sentence: “He danced stupidly.”
Paraphrase: “He danced in a stupid manner.”

⸻

SUBJECT-ORIENTED (adverb.subject_oriented)

Attributes a property or attitude to the subject of the sentence in relation to the event.

Paraphrase Test: Can it be rephrased as “It was [X] of [SUBJECT] to [VERB]”?

Example:
Sentence: “Stupidly, he walked into traffic.”
Paraphrase: “It was stupid of him to walk into traffic.”

⸻

SPEAKER-ORIENTED (adverb.speaker_oriented)

Expresses the speaker’s stance, evaluation, or attitude toward the proposition or the act of speaking.

Paraphrase Test: Can it be paraphrased as “I [believe / say / judge] that…” or “It is [unfortunate / fortunate / evident] that…”?

Examples:
	•	“Presumably, he missed the deadline.” → “I presume that he missed the deadline.”
	•	“Unfortunately, the project failed.” → “It is unfortunate that the project failed.”
	•	“Frankly, I disagree.” → “I say this frankly.”

⸻

FREQUENCY (adverb.frequency)

Describes how often the event occurs.

Question Test: “How often?”

Example:
Sentence: “She frequently visits her grandmother.”
Answer: “Frequently.” → Frequency adverb

⸻

TEMPORAL (adverb.temporal)

Specifies when the event happens or its duration.

Question Test: “When?” or “For how long?”

Examples:
	•	Sentence: “She arrived yesterday.” → When? → Yesterday
	•	Sentence: “He stayed briefly.” → For how long? → Briefly

⸻

SPATIAL (LOCATIVE) (adverb.spatial)

Specifies where the event takes place.

Question Test: “Where?”

Example:
Sentence: “He stood outside.”
Answer: “Outside.” → Spatial adverb

⸻

DEGREE (adverb.degree)

Describes the intensity, extent, or scalar position of another element (adjective, adverb, verb).

Question Test: “To what extent?” or “How much?”

Example:
Sentence: “She is very happy.”
Question: “To what extent is she happy?”
Answer: “Very.” → Degree adverb

⸻

DOMAIN (adverb.domain)

Limits the proposition to a particular semantic or disciplinary domain.

Paraphrase Test: “In a [X] sense” or “From a [X] perspective”

Example:
Sentence: “Politically, the policy is controversial.”
Paraphrase: “In a political sense, the policy is controversial.”

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
Usage: “Wisely, he did not sign the contract.”
Reasoning: The sentence can be paraphrased as “It was wise of him to not sign the contract,” which assigns a property (wisdom) to the subject with respect to the action. This indicates a subject-oriented adverb.
Answer: adverb.subject_oriented

⸻

Adverb: frankly
Usage: “Frankly, I don’t agree with that decision.”
Reasoning: The sentence can be paraphrased as “I say this frankly,” where the adverb qualifies the act of speaking, not the proposition itself. This indicates a speaker-oriented adverb.
Answer: adverb.speaker_oriented

⸻

Adverb: daily
Usage: “He exercises daily.”
Reasoning: The adverb answers the question “How often?” with “daily,” describing the frequency of the event.
Answer: adverb.frequency


Adverb: {adverb}
Usage: {sentence}
Reasoning: {gen(max_tokens=512, stop="\n")}
Answer: {
            select(
                [
                    "adverb.manner",
                    "adverb.subject_oriented",
                    "adverb.epistemic",
                    "adverb.evaluative",
                    "adverb.speech_act",
                    "adverb.frequency",
                    "adverb.temporal",
                    "adverb.spatial",
                    "adverb.degree",
                    "adverb.domain",
                ],
                name="s1",
            )
        }
"""
    )


def main(raw_args=None):
    parser = argparse.ArgumentParser(
        prog="prompt_generate.py",
        description="What the program does",
        epilog="""Generate the prompt for querying an LLM.

    Usage:
        prompt_generate.py [-n] <c1> <c2> <targets>

    Arguments:

        <corpus> = corpus
        <instruction> = give examples on how to do the task
        <task> = task to generate the prompt
        <model> = prompt the model to reason before answering

    """,
    )
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", required=True)
    args = parser.parse_args(raw_args)
    model = models.LlamaCpp(
        args.model,
        n_ctx=args.ctx,
        # n_gpu_layers=-1,
        # flash_attn=True,
        # echo=False,
    )

    lm = model
    adverb = input("Adverb> ")
    sentence = input("Sentence> ")
    print(f"Adverb: {adverb}")
    print(f"Sentence: {sentence}")
    print("Generating prompt...")
    data = pd.read_csv("adverb_os.tsv")
    results = []
    for index, row in data.iterrows():
        lm += adverb_diagnose(row["adverb"], row["sentence"])
        results.append(str(lm))

    data["llm"] = results

    data.to_csv("adverb_os_llm.tsv", index=False)


if __name__ == "__main__":
    main()
