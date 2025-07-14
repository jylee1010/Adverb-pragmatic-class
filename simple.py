import argparse
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

SPEAKER-ORIENTED (Select one subtype)

These adverbs express the speaker’s stance toward the proposition or the speech act. Use the following subtypes:

a. Epistemic (adverb.epistemic)

Conveys degree of certainty or source of knowledge.

Paraphrase: “I [believe / presume / suspect] that…” or “It is [presumed / inferred] that…”

Example:
Sentence: “Presumably, he missed the deadline.”
Paraphrase: “I presume that he missed the deadline.”

⸻

b. Evaluative (adverb.evaluative)

Conveys speaker’s emotional or normative judgment of the proposition.

Paraphrase: “It is [unfortunate / amazing / regrettable] that…” or “From my perspective, it is X that…”

Example:
Sentence: “Unfortunately, the project failed.”
Paraphrase: “It is unfortunate that the project failed.”

⸻

c. Speech-Act Oriented (adverb.speech_act)

Modifies the act of speaking rather than the proposition itself.

Paraphrase: “I say this [frankly / honestly / confidentially]…”

Example:
Sentence: “Frankly, I disagree.”
Paraphrase: “I say this frankly.”

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
Sentence: “She arrived yesterday.” → When? → Yesterday
Sentence: “He stayed briefly.” → For how long? → Briefly

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
adverb.manner
adverb.subject_oriented
adverb.epistemic
adverb.evaluative
adverb.speech_act
adverb.frequency
adverb.temporal
adverb.spatial
adverb.degree
adverb.domain

⸻

Example: 

Adverb: wisely
Usage: “Wisely, he did not sign the contract.”
Reasoning: The sentence can be paraphrased as “It was wise of him to not sign the contract,” which assigns a property (wisdom) to the subject with respect to the action. This indicates a subject-oriented adverb.
Answer: adverb.subject_oriented
⸻
Adverb: frankly
Usage: “Frankly, I don’t agree with that decision.”
Reasoning: The sentence can be paraphrased as “I say this frankly,” where the adverb qualifies the act of speaking, not the proposition itself. This indicates a speech-act adverb.
Answer: adverb.speech_act

⸻
Adverb: daily
Usage: “He exercises daily.”
Reasoning: The adverb answers the question “How often?” with “daily,” describing the frequency of the event.
Answer: adverb.frequency
⸻

Adverb: {adverb}
Usage: {sentence}
Reasoning: {gen(max_tokens=512, stop='\n')}
Answer: {select([adverb.manner
adverb.subject_oriented
adverb.epistemic
adverb.evaluative
adverb.speech_act
adverb.frequency
adverb.temporal
adverb.spatial
adverb.degree
adverb.domain], name="s1", max_tokens=100)}
"""
    )


@guidance(stateless=True)
def rhetorical(lm, lemma, usage1, usage2):
    return (
        lm
        + f"""
    You are an expert linguist.
    You are presented with two sentences that both contain a shared word.
    Your task is to create and analyze zeugmas.
    Follow these steps to complete the task:
        Step 1. Rewrite the first sentence in a simplified form preserving the lemma.
        Step 2. Rewrite the second sentence in a simplified form preserving the lemma.
        Step 3. Write a sentence that joins both sentences using zeugma and the same shared word.
            If the construction doesn't make a bad pun, write same, otherwise, write different.
    Examples:
    Lemma: Plane
    Context 1: He loves planes and want to become a pilot.
    Context 2: The plane landed just now.
    <think>
    Here the lemma is plane, so I have to join both sentences using this word.
    First I will rewrite the sentences in a simplified form. For the first sentence, I can say:
    He loves planes.
    For the second sentence, I can say:
    The plane landed.
    Now I will join both sentences using zeugma:
    He loves planes, like the one that landed.
    The zeugma version doens't sound like a bad pun, so I will write 'same'.
    </think>
    1) He loves planes.
    2) The plane landed.
    3) He loves planes, like the one that landed.
    Conclusion: It doesn't make a bad pun.
    Answer: same
    ---
    Lemma: Cell
    Context 1: Anyone leaves a cell phone or handheld at home, many of them faculty members from nearby.
    Context 2: I just watch the dirty shadow the window bar makes across the wall of my cell.
    <think>
    Here the lemma is cell, so I have to join both sentences using this word.
    First I will rewrite the sentences in a simplified form. For the first sentence, I can say:
    Anyone leaves a cell phone at home.
    For the second sentence, I can say:
    The wall of my cell.
    Now I will join both sentences using zeugma:
    The wall of my cell which I leave at home.
    The zeugma version makes a bad pun, it doesn't sound right, so I will write 'different'.
    </think>
    1) Anyone leaves a cell phone at home.
    2) The wall of my cell.
    3) The wall of my cell which I leave at home.
    Conclusion: It makes a bad pun.
    Answer: different
    ---
    Lemma: {lemma}
    Context 1: {usage1}
    Context 2: {usage2}
    <think>
    {gen(stop='</think>')}
    </think>
    1) {gen("s1", max_tokens=100)}
    """
    )
# 2) {gen("s2", stop="\n")}
#     3) {gen("s3", stop="\n")}
#     Conclusion: {gen("conclude", stop="\n")}
#     Answer: {select(["same", "different"], "answer")}

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
    # lemma = input("Lemma> ")
    # usage1 = input("Usage1> ")
    # usage2 = input("Usage2> ")
    # print(f"Lemma: {lemma}")
    # print(f"Usage1: {usage1}")
    # print(f"Usage2: {usage2}")
    # print("Generating prompt...")
    # lm += rhetorical(lemma, usage1, usage2)
    adverb = input("Adverb> ")
    sentence = input("Sentence> ")
    print(f"Adverb: {adverb}")
    print(f"Sentence: {sentence}")
    print("Generating prompt...")
    lm += adverb_diagnose(adverb, sentence)
    print(str(lm))
    # output = {
    #     "s1": lm["s1"],
    #     "s2": lm["s2"],
    #     "s3": lm["s3"],
    #     "conclude": lm["conclude"],
    #     "answer": lm["answer"],
    # }
    # pprint(output)


if __name__ == "__main__":
    main()
